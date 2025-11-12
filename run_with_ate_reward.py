import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.data
from tqdm import tqdm

from graph_perturbation import gen_perturbed_graphs, id2euv, eindex2bool
from utils import extract_subgraph, extract_subgraph_backup

EPS = 1e-15


def test_policy(gnn_explainer, trained_gnn, test_loader):
    device = gnn_explainer.device
    
    topK_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_count_arr = np.zeros(len(topK_ratio_list))

    with torch.no_grad():
        for graph in iter(test_loader):
            graph = graph.to(device)
            max_budget = graph.num_edges

            check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
            
            valid_budget = max(int(0.9 * max_budget), 1)
            state = torch.zeros(max_budget, dtype=torch.bool)
            
            for budget in range(1, valid_budget+1):
                available_actions = state[~state].clone()

                _, _, remove_actions, _ = gnn_explainer(graph=graph, state=state, train_flag=False)

                available_actions[remove_actions] = True
                state[~state] = available_actions.clone()

                if budget not in check_budget_list:
                    continue
                check_idx = check_budget_list.index(budget)
                keep_mask = ~state

                subgraph = extract_subgraph_backup(graph=graph, selection=keep_mask)
                if subgraph.num_nodes is 0: break
                # if subgraph is None: continue

                subgraph_pred = trained_gnn(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

                acc = sum(graph.y == subgraph_pred.argmax(dim=1))
                # print(f'budget: {budget}, num of removed edges: {state.sum()}  ACC: {round(acc.item()/len(test_loader), 4)}')
                acc_count_arr[check_idx] += acc

    acc_count_arr[-1] = len(test_loader)
    acc_count_arr = acc_count_arr / len(test_loader)
    print('\nACC-AUC: %.4f' % acc_count_arr.mean())
    return acc_count_arr.mean(), acc_count_arr


def train_policy(gnn_explainer, 
                 trained_gnn, 
                 train_loader, 
                 test_loader, 
                 optimizer,
                 topK_ratio=0.1, 
                 batch_size=32, 
                 num_samples=10,
                 alpha=0.2,
                 beta=1):
    # 设置训练的总回合数
    num_episodes = 15
    device = gnn_explainer.device

    best_acc_auc, best_acc_curve = 0, 0

    previous_baseline_list = []
    current_baseline_list = []
    for ep in range(1, num_episodes + 1): 
        loss = 0.
        avg_reward = []

        for graph in tqdm(train_loader, total=len(train_loader)):
            # 将图数据移动到指定设备
            graph = graph.to(device)

            # 根据topK_ratio计算有效的预算
            if topK_ratio < 1:
                valid_budget = max(int(topK_ratio * graph.num_edges / batch_size), 1)
            else:
                valid_budget = topK_ratio

            batch_loss = torch.tensor(0., dtype=torch.float32, requires_grad=True).to(device)

            # 初始化当前状态，记录历史上被删除的边，删除的为True，否则为False
            current_state = torch.zeros(graph.num_edges, dtype=torch.bool)
            next_state = current_state.clone()

            pre_reward = torch.zeros(graph.y.size()).to(device)
            num_beam = 2
            
            for budget in range(valid_budget):
                # 获取可用的动作
                available_action = current_state[~current_state].clone()
                beam_reward_list = []
                beam_action_list = []
                beam_action_probs_list = []

                for beam in range(num_beam):
                    beam_available_action = current_state[~current_state].clone()
                    beam_next_state = current_state.clone()

                    if beam is 0:
                        _, remove_action_probs, remove_actions, unique_batch = gnn_explainer(graph, current_state, train_flag=False)
                    else:
                        _, remove_action_probs, remove_actions, unique_batch = gnn_explainer(graph, current_state, train_flag=True)

                    beam_available_action[remove_actions] = True
                    beam_next_state[~current_state] = beam_available_action
                    
                    reward = get_ate_reward(trained_gnn, graph, beam_next_state, graph.y, num_samples=num_samples,
                                            max_virtual_nodes=2, edge_drop_rate=0.1, alpha=alpha, beta=beta)
                    if reward is None: continue
                
                    reward = reward[unique_batch]

                    if len(previous_baseline_list) - 1 < budget:
                        baseline_reward = 0.
                    else:
                        baseline_reward = previous_baseline_list[budget]

                    if len(current_baseline_list) - 1 < budget:
                        current_baseline_list.append([torch.mean(reward)])
                    else:
                        current_baseline_list[budget].append(torch.mean(reward))

                    reward -= baseline_reward
                    avg_reward += reward.tolist()
                
                    beam_reward_list.append(reward)
                    beam_action_list.append(remove_actions)
                    beam_action_probs_list.append(remove_action_probs)
    
                if len(beam_reward_list) == 0:
                    continue

                beam_reward_list = torch.stack(beam_reward_list).T
                beam_action_list = torch.stack(beam_action_list).T
                beam_action_probs_list = torch.stack(beam_action_probs_list).T
                beam_action_probs_list = F.softmax(beam_action_probs_list, dim=1)
                batch_loss += torch.mean(-torch.log(beam_action_probs_list + EPS) * beam_reward_list)
                
                max_reward, max_reward_idx = torch.max(beam_reward_list, dim=1)
                max_actions = beam_action_list[range(beam_action_list.size()[0]), max_reward_idx]

                # 更新状态
                available_action[max_actions] = True
                next_state[~current_state] = available_action
                current_state = next_state.clone()
                pre_reward[unique_batch] = max_reward
        
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 累加损失
            loss += batch_loss

        # 计算平均奖励
        avg_reward = torch.mean(torch.FloatTensor(avg_reward))

        # 打印当前回合的信息
        print('Episode: %d, loss: %.6f, average rewards: %.6f' % (ep, loss.detach(), avg_reward.detach()))

        # gnn_explainer.eval()
        ep_acc_auc, ep_acc_curve = test_policy(gnn_explainer, trained_gnn, test_loader)
        # gnn_explainer.train()

        if ep_acc_auc >= best_acc_auc:
            best_acc_auc = ep_acc_auc
            best_acc_curve = ep_acc_curve

        previous_baseline_list = [torch.mean(torch.stack(cur_baseline)) for cur_baseline in current_baseline_list]
        current_baseline_list = []
    return gnn_explainer, best_acc_auc, best_acc_curve


def get_ite_reward_backup(controlled_pred, treated_pred, true_y, mode='mutual_info', pre_reward=0.):
    # 根据模式选择计算奖励的方式
    if mode == 'mutual_info':
        # 计算基于互信息的奖励
        reward = torch.sum(controlled_pred * torch.log(treated_pred + EPS), dim=1)
        reward += 2 * (true_y == treated_pred.argmax(dim=1)).float() - 1.

    elif mode == 'binary':
        # 计算基于二分类的奖励
        reward = (true_y == treated_pred.argmax(dim=1)).float()
        reward = 2. * reward - 1.

    elif mode == 'cross_entropy':
        # 计算基于交叉熵的奖励
        reward = torch.log(treated_pred + EPS)[:, true_y]

    # 将先前的奖励乘以0.9后加到当前奖励上
    # reward += pre_reward
    reward += 0.9 * pre_reward

    return reward


def calc_mutual_reward(controlled_pred, treated_pred):
    reward = torch.sum(controlled_pred * torch.log(treated_pred + EPS), dim=1)
    return reward


def calc_binary_reward(treated_pred, true_y):
    reward = (true_y == treated_pred.argmax(dim=1)).float()
    reward = 2. * reward - 1.
    return reward


def calc_cross_entropy_reward(treated_pred, true_y):
    reward = torch.log(treated_pred + EPS)[:, true_y]
    return reward


def calc_predicted_probability(trained_gnn, graph):
    pred = trained_gnn(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
    prob = F.softmax(pred, dim=0).detach()
    return prob


def get_ate_reward(trained_gnn: torch.nn.Module,
                   graph: torch_geometric.data.Data,
                   state: torch.BoolTensor,
                   true_y: torch.Tensor,
                   num_samples: int,
                   max_virtual_nodes: int,
                   edge_drop_rate: float,
                   alpha: float,
                   beta: float
                   ):
    """

    计算ATE奖励（Average Treatment Effect Reward）。

    Args:
        trained_gnn (nn.Module): 已训练的图神经网络模型。
        graph (torch_geometric.data.Data): 输入的图数据。
        state (torch.BoolTensor): gnn_explainer预测的需要切除的边张量，需要切除为True，保留为False，且存储了历史的记录。
        true_y (torch.Tensor): 目标标签张量。
        num_samples (int): 生成的扰动图数量。
        max_virtual_nodes (int): 图中虚拟节点的最大数量。
        edge_drop_rate (float): 边删除率。
        # alpha (float): 保留边惩罚系数。
        beta (float): ATE奖励的标准差惩罚系数。

    Returns:
        torch.Tensor: 计算得到的ATE奖励。

    """
    # 生成随机扰动图 -------------------------------------------------------------------------------
    focused_eindex = id2euv(graph, state)
    control_group = gen_perturbed_graphs(graph,
                                         focused_eindex,
                                         num_samples,
                                         edge_drop_rate,
                                         max_virtual_nodes)

    # 使用模型对图进行预测，并获取预测结果---------------------------------------------------------------
    mutual_rewards = []
    for index, controlled in enumerate(control_group):
        # 对照组
        controlled_pred = calc_predicted_probability(trained_gnn, controlled)

        # 实验组
        removed_mask = eindex2bool(controlled, focused_eindex)
        keep_mask = ~removed_mask
        # keep_mask = removed_mask
        treated = extract_subgraph_backup(controlled, keep_mask)
        # if treated is None: continue

        treated_pred = calc_predicted_probability(trained_gnn, treated)

        if index is 0:
            try:
                binary_reward = calc_binary_reward(treated_pred, true_y)
            except RuntimeError as e:
                # print(e)
                binary_reward = torch.tensor(-2, dtype=torch.float32).to(graph.x.device)
        try:
            mutual_reward = calc_mutual_reward(controlled_pred, treated_pred)
        except RuntimeError as e:
            # print(e)
            continue
        mutual_rewards.append(mutual_reward.unsqueeze(0))
    
    if len(mutual_rewards) == 0:
        return None

    mutual_rewards = torch.cat(mutual_rewards, dim=0).to(device=graph.x.device)
    mutual_reward_mean = mutual_rewards.mean(dim=0)
    mutual_reward_std = mutual_rewards.std(dim=0)
    
    if mutual_reward_std.isnan().any():
        mutual_reward_std = torch.zeros_like(mutual_reward_std, dtype=mutual_reward_std.dtype, device=mutual_reward_std.device)

    final_reward = alpha * (mutual_reward_mean - beta * mutual_reward_std) + (1-alpha) * binary_reward
    # return final_reward, mutual_reward_mean, mutual_reward_std, binary_reward
    return final_reward