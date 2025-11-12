from turtle import forward
import torch
from torch.nn import Linear, ELU, Sequential, ModuleList
import torch.nn.functional as F
from torch_scatter import scatter_max
from utils import extract_subgraph
from itertools import chain


class Causal2X(torch.nn.Module):
    def __init__(self, trained_gnn, num_labels, hidden_size, device=None):
        # 调用父类的构造函数
        super(Causal2X, self).__init__()
        self.device = device

        # 将传入的模型保存到实例变量中
        # if self.device is not None:
        #     self.trained_gnn = trained_gnn.to(self.device)
        # else:
        #     self.trained_gnn = trained_gnn
        self.trained_gnn = trained_gnn
        # self.trained_gnn.eval()

        # 保存传入的标签数量
        self.num_labels = num_labels

        # 保存传入的隐藏层大小
        self.hidden_size = hidden_size

        # 创建边动作表示生成器
        self.edge_action_rep_generator = self.build_edge_action_rep_generator()

        # 构建边动作概率生成器
        self.edge_action_prob_generator = self.build_edge_action_prob_generator()

    def build_edge_action_rep_generator(self):
        # 创建一个Sequential模型，包含三个线性层和两个ELU激活函数层
        edge_action_rep_generator = Sequential(
            Linear(self.hidden_size * 2, self.hidden_size * 4),
            ELU(),
            Linear(self.hidden_size * 4, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size)
        ).to(self.device)
        return edge_action_rep_generator

    def build_edge_action_prob_generator(self):
        # 创建一个Sequential模型
        edge_action_prob_generator = Sequential(
            # 添加一个全连接层，输入维度为self.hidden_size，输出维度也为self.hidden_size
            Linear(self.hidden_size, self.hidden_size),
            # 添加一个ELU激活函数层
            ELU(),
            # 添加一个全连接层，输入维度为self.hidden_size，输出维度为self.num_labels
            Linear(self.hidden_size, self.num_labels)
        ).to(self.device)
        # 返回构建好的模型
        return edge_action_prob_generator

    def forward(self, graph, state, train_flag=False):
        """"
        Args:
            graph: 输入图数据
            state: 当前状态的索引，删除的为True，否则为False
        """
        graph = graph.to(self.device)

        # 已删除的边的索引与属性
        removed_edge_index = graph.edge_index[:, state]
        removed_edge_attr = graph.edge_attr[state]

        # 可用边的索引与属性
        available_edge_index = graph.edge_index[:, ~state]
        available_edge_attr = graph.edge_attr[~state]

        # 所有节点的嵌入表示
        all_node_reps = self.trained_gnn.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach()

        # 已删除的边的两个端点的嵌入表示
        removed_node_reps = self.trained_gnn.get_node_reps(graph.x, removed_edge_index, removed_edge_attr,
                                                             graph.batch).detach()

        available_node_reps = all_node_reps - removed_node_reps

        # 将可用边的两个端点的节点表示拼接起来，形成边的动作表示
        available_action_reps = torch.cat([available_node_reps[available_edge_index[0]],
                                           available_node_reps[available_edge_index[1]]], dim=1).to(self.device)

        # 通过边动作表示生成器生成最终的动作表示
        available_action_reps = self.edge_action_rep_generator(available_action_reps)

        # 获取可用边对应的节点批次
        available_action_batch = graph.batch[available_edge_index[0]]
        # 获取对应批次的目标值
        available_y_batch = graph.y[available_action_batch]

        # 预测边的动作概率
        available_action_probs = self.predict(available_action_reps, available_y_batch)

        # 获取每个批次中的最大动作概率和对应的动作
        # scatter_max函数用于获取每个批次中的最大动作概率和对应的动作
        remove_action_probs, remove_actions = scatter_max(available_action_probs, available_action_batch, dim=0)

        # 如果处于训练模式
        if train_flag:
            # 随机生成一个动作概率
            rand_action_probs = torch.rand(remove_action_probs.size()).to(self.device)
            _, rand_actions = scatter_max(rand_action_probs, available_action_batch)
            return available_action_probs, available_action_probs[rand_actions], rand_action_probs
        return available_action_probs, remove_action_probs, remove_actions

    def predict(self, ava_action_reps, target_y):
        # 生成动作概率，action_probs的大小为[batch_size, num_labels]
        action_probs = self.edge_action_prob_generator(ava_action_reps)

        # 根据目标标签选择相应的动作概率，action_probs的大小为[batch_size, 1]
        action_probs = action_probs.gather(1, target_y.unsqueeze(1))

        # 将选中的动作概率重塑为一维数组
        action_probs = action_probs.squeeze(1)

        # 对动作概率应用softmax函数，考虑可用动作批次
        action_probs = F.softmax(action_probs, dim=0)
        return action_probs

    def get_optimizer(self, lr=0.01, weight_decay=1e-5, scope='all'):
        if scope == 'all':
            params = self.parameters()
        else:
            params = chain(self.edge_action_rep_generator.parameters(), self.edge_action_prob_generator.parameters())
            # params = list(self.edge_action_rep_generator.parameters()) + \
            #          list(self.edge_action_prob_generator.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optimizer


class Causal2XPro(Causal2X):
    def __init__(self, trained_gnn, num_labels, hidden_size, device=None):
        super(Causal2XPro, self).__init__(trained_gnn, num_labels, hidden_size, device)
    
    def build_edge_action_prob_generator(self):
        edge_action_prob_generator = ModuleList()
        for i in range(self.num_labels):
            i_explainer = Sequential(
                Linear(self.hidden_size * 2, self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, out_features=1)
            ).to(self.device)
            edge_action_prob_generator.append(i_explainer)

        return edge_action_prob_generator
    
    def predict(self, graph_rep, removed_rep, ava_action_reps, target_y, ava_action_batch):
        action_graph_reps = graph_rep - removed_rep
        action_graph_reps = action_graph_reps[ava_action_batch]
        action_graph_reps = torch.cat([ava_action_reps, action_graph_reps], dim=1)

        action_probs = []
        for i_explainer in self.edge_action_prob_generator:
            i_action_probs = i_explainer(action_graph_reps)
            action_probs.append(i_action_probs)
        action_probs = torch.cat(action_probs, dim=1)

        action_probs = action_probs.gather(1, target_y.view(-1, 1))
        action_probs = action_probs.reshape(-1)
        return action_probs
    
    def forward(self, graph, state, train_flag=False):
        graph = graph.to(self.device)
        graph_rep = self.trained_gnn.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach()

        if len(torch.where(state)[0]) == 0:
            removed_rep = torch.zeros(graph_rep.size()).to(self.device)
        else:
            removed_subgraph = extract_subgraph(graph, selection=state)
            removed_rep = self.trained_gnn.get_graph_rep(removed_subgraph.x, removed_subgraph.edge_index, 
                                                            removed_subgraph.edge_attr, removed_subgraph.batch).detach()

        available_edge_index = graph.edge_index[:, ~state]
        available_edge_attr = graph.edge_attr[~state]
        available_node_reps = self.trained_gnn.get_node_reps(graph.x, available_edge_index, available_edge_attr, graph.batch).detach()

        available_action_reps = torch.cat([available_node_reps[available_edge_index[0]], 
                                           available_node_reps[available_edge_index[1]]], dim=1).to(self.device)
        available_action_reps = self.edge_action_rep_generator(available_action_reps)

        available_action_batch = graph.batch[available_edge_index[0]]
        available_y_batch = graph.y[available_action_batch]

        unique_batch, available_action_batch = torch.unique(available_action_batch, return_inverse=True)
        available_action_probs = self.predict(graph_rep, removed_rep, 
                                              available_action_reps, 
                                              available_y_batch, 
                                              available_action_batch)
        
        if train_flag:
            rand_action_probs = torch.rand(available_action_probs.size()).to(self.device)
            _, rand_actions = scatter_max(rand_action_probs, available_action_batch)
            remove_action_probs = available_action_probs[rand_actions]
            remove_actions = rand_actions
        else:
            remove_action_probs, remove_actions = scatter_max(available_action_probs, available_action_batch)

        return available_action_probs, remove_action_probs, remove_actions, unique_batch