import numpy as np
import torch
import torch.nn.functional as F
from utils import extract_subgraph
from torch_scatter import scatter_max
from tqdm import tqdm

EPS = 1e-15
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_policy(rc_explainer, trained_model, test_loader, topN=None):
    rc_explainer.eval()
    trained_model.eval()

    # 定义topK的比例列表
    topK_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # 初始化准确率的计数器列表
    acc_count_list = np.zeros(len(topK_ratio_list))

    # 初始化topN的精确度和召回率计数器
    precision_topN_count = 0.
    recall_topN_count = 0.

    with torch.no_grad():
        for graph in iter(test_loader):
            graph = graph.to(DEVICE)
            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            # 计算每个topK比例的预算值
            check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
            valid_budget = max(int(0.9 * max_budget), 1)

            for budget in range(valid_budget):
                # 遍历给定的预算范围
                available_actions = state[~state].clone()
                # 获取当前状态的非零元素，并克隆这些元素作为可用动作

                # 使用rc_explainer模型做出决策
                _, _, make_action_id = rc_explainer(graph=graph, state=state, train_flag=False)

                # 更新状态，标记已选择的动作
                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()

                # 如果当前预算在check_budget_list中，则计算准确率
                if (budget + 1) in check_budget_list:
                    check_idx = check_budget_list.index(budget + 1)
                    subgraph = extract_subgraph(graph, state)
                    subgraph_pred = trained_model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

                    acc_count_list[check_idx] += sum(graph.y == subgraph_pred.argmax(dim=1))

                # 如果设置了topN，并且当前预算等于topN-1，则计算精确度和召回率
                if topN is not None and budget == topN - 1:
                    precision_topN_count += torch.sum(state * graph.ground_truth_mask[0]) / topN
                    recall_topN_count += torch.sum(state * graph.ground_truth_mask[0]) / sum(graph.ground_truth_mask[0])

    # 将最后一个准确率计数设置为测试集大小
    acc_count_list[-1] = len(test_loader)
    # 将准确率计数列表转换为numpy数组并除以测试集大小
    acc_count_list = np.array(acc_count_list) / len(test_loader)

    # 计算topN的精确度和召回率
    precision_topN_count = precision_topN_count / len(test_loader)
    recall_topN_count = recall_topN_count / len(test_loader)

    # 打印结果
    if topN is not None:
        print('ACC-AUC: %.4f, Precision@5: %.4f, Recall@5: %.4f\n' %
              (acc_count_list.mean(), precision_topN_count, recall_topN_count))
    else:
        print('ACC-AUC: %.4f\n' % acc_count_list.mean())
    print(acc_count_list)

    # 返回准确率计数列表除以测试集大小的numpy数组
    return np.array(acc_count_list) / len(test_loader)


def normalize_reward(reward_pool):
    # 将reward_pool中的奖励值堆叠成一个张量
    reward_pool = torch.stack(reward_pool)

    # 如果reward_pool的第一个维度（样本数量）不等于1
    if reward_pool.shape[0] != 1:
        # 计算reward_pool的平均值
        reward_mean = torch.mean(reward_pool)
        # 计算reward_pool的标准差，并添加一个小的正数EPS以避免除以零
        reward_std = torch.std(reward_pool) + EPS
        # 对reward_pool进行标准化处理：每个元素减去平均值后除以标准差
        reward_pool = (reward_pool - reward_mean) / reward_std
    # 返回处理后的reward_pool
    return reward_pool


def bias_detector(model, graph, valid_budget):
    pred_bias_list = []

    for budget in range(valid_budget):
        num_repeat = 2

        i_pred_bias = 0.
        for i in range(num_repeat):
            # 初始化 bias_selection 为全零布尔张量，长度为 graph 的边数
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool)

            # 获取 graph 的 batch 索引
            ava_action_batch = graph.batch[graph.edge_index[0]]
            # 生成与 ava_action_batch 尺寸相同的随机张量，并将其移至设备
            ava_action_probs = torch.rand(ava_action_batch.size()).to(DEVICE)
            # 使用 scatter_max 函数选择具有最大概率的动作
            _, added_actions = scatter_max(ava_action_probs, ava_action_batch)

            # 将 selected_actions 对应的 bias_selection 设置为 True
            bias_selection[added_actions] = True
            # 根据 bias_selection 重新标记图
            bias_subgraph = extract_subgraph(graph, bias_selection)
            # 使用模型对 bias_subgraph 进行预测
            bias_subgraph_pred = model(bias_subgraph.x, bias_subgraph.edge_index,
                                       bias_subgraph.edge_attr, bias_subgraph.batch).detach()

            # 将 bias_subgraph_pred 累加到 i_pred_bias 中
            i_pred_bias += bias_subgraph_pred / num_repeat

        # 将 i_pred_bias 添加到 pred_bias_list 中
        pred_bias_list.append(i_pred_bias)

    return pred_bias_list


def train_policy(rc_explainer, trained_model, train_loader, test_loader, optimizer,
                 topK_ratio=0.1, debias_flag=False, topN=None, batch_size=32):
    # 设置训练的总回合数
    num_episodes = 100

    # 在测试集上测试策略
    # test_policy(rc_explainer, trained_gnn, test_loader, topN)
    ep = 0

    while ep < num_episodes:
        # 设置rc_explainer为训练模式
        rc_explainer.train()
        # 设置trained_model为评估模式
        trained_model.eval()

        loss = 0.
        avg_reward = []

        for graph in tqdm(iter(train_loader), total=len(train_loader)):
            # 将图数据移动到指定设备
            graph = graph.to(DEVICE)

            # 根据topK_ratio计算有效的预算
            if topK_ratio < 1:
                valid_budget = max(int(topK_ratio * graph.num_edges / batch_size), 1)
            else:
                valid_budget = topK_ratio

            batch_loss = 0.

            # 使用模型对图进行预测，并获取预测结果---------------------------------------------------------------
            full_subgraph_pred = trained_model(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            full_subgraph_pred = F.softmax(full_subgraph_pred).detach()
            # --------------------------------------------------------------------------------------------

            # 初始化当前状态
            current_state = torch.zeros(graph.num_edges, dtype=torch.bool)

            # 如果需要去除偏差，调用偏差检测器
            if debias_flag:
                pred_bias_list = bias_detector(trained_model, graph, valid_budget)

            pre_reward = 0.
            for budget in range(valid_budget):
                # 获取可用的动作
                available_action = current_state[~current_state].clone()

                # 使用rc_explainer获取添加的动作及其概率
                _, added_action_probs, added_actions = rc_explainer(graph, current_state, train_flag=True)

                # 更新状态
                new_state = current_state.clone()
                try:
                    available_action[added_actions] = True
                except:
                    pass

                new_state[~current_state] = available_action

                # 重新标记图，并获取新的子图预测结果--------------------------------------------------------------
                new_subgraph = extract_subgraph(graph, new_state)
                new_subgraph_pred = trained_model(new_subgraph.x, new_subgraph.edge_index,
                                                  new_subgraph.edge_attr, new_subgraph.batch)

                # 如果需要去除偏差，则调整预测结果
                if debias_flag:
                    new_subgraph_pred = F.softmax(new_subgraph_pred - pred_bias_list[budget]).detach()
                else:
                    new_subgraph_pred = F.softmax(new_subgraph_pred).detach()

                # 计算奖励
                reward = get_reward(full_subgraph_pred, new_subgraph_pred, graph.y, mode='binary',
                                    pre_reward=pre_reward)
                pre_reward = reward
                # -----------------------------------------------------------------------------------------

                # 更新损失和平均奖励
                batch_loss += torch.mean(- torch.log(added_action_probs + EPS) * reward)
                avg_reward += reward.tolist()

                # 更新当前状态
                current_state = new_state.clone()

            # 反向传播并更新参数
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 累加损失
            loss += batch_loss

        # 计算平均奖励
        avg_reward = torch.mean(torch.FloatTensor(avg_reward))

        ep += 1
        # 打印当前回合的信息
        print('Episode: %d, loss: %.4f, average rewards: %.4f' % (ep, loss.detach(), avg_reward.detach()))

        # 在测试集上测试策略
        # test_policy(rc_explainer, trained_gnn, test_loader, topN)

        # 设置rc_explainer为训练模式
        # rc_explainer.train()
    return rc_explainer


def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, mode='mutual_info', pre_reward=0.):
    # 根据模式选择计算奖励的方式
    if mode == 'mutual_info':
        # 计算基于互信息的奖励
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1)
        reward += 2 * (target_y == new_subgraph_pred.argmax(dim=1)).float() - 1.

    elif mode == 'binary':
        # 计算基于二分类的奖励
        reward = (target_y == new_subgraph_pred.argmax(dim=1)).float()
        reward = 2. * reward - 1.

    elif mode == 'cross_entropy':
        # 计算基于交叉熵的奖励
        reward = torch.log(new_subgraph_pred + EPS)[:, target_y]

    # 将先前的奖励乘以0.9后加到当前奖励上
    # reward += pre_reward
    reward += 0.9 * pre_reward

    return reward
