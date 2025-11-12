# encoding: utf-8

import torch
from torch import LongTensor, Tensor
import numpy as np
from torch_geometric.data import Data


def id2euv(graph: Data, edge_ids: Tensor) -> LongTensor:
    """
    将边ids转换为边的起始节点和终止节点uv

    """
    device = graph.edge_index.device
    edge_ids = edge_ids.ravel().to(device)
    eindex = graph.edge_index[:, edge_ids]
    if eindex.dim() == 1:
        return eindex.unsqueeze(-1)
    return eindex


def eindex2bool(graph: Data, eindex: LongTensor) -> Tensor:
    """
    将边的起始节点和终止节点uv转换为边id
    """
    device = graph.edge_index.device
    bool_ = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
    eindex = eindex.to(device)

    for i in range(eindex.size(1)):
        src_bool, tgt_bool = graph.edge_index == eindex[:, i].unsqueeze(-1)
        bool_ |= src_bool & tgt_bool
    return bool_


def gen_edge_perturbed_graph(
        graph: Data,
        focused_eindex: LongTensor,
        edge_drop_rate: float = 0.1) -> Data:
    perturbed_graph = graph.clone()  # 克隆原始图
    num_edges = perturbed_graph.num_edges

    focused_bool = eindex2bool(graph, focused_eindex)
    drop_candidate_bool = ~focused_bool

    num_to_drop = min(int(edge_drop_rate * num_edges) or 1, drop_candidate_bool.sum())  # 计算需要删除的边数
    drop_candidate_index = drop_candidate_bool.nonzero().ravel()
    drop_index = drop_candidate_index[torch.randperm(drop_candidate_index.size(0))][:num_to_drop]

    # 构建掩码，用于筛选保留的边
    keep_mask = torch.ones(num_edges, dtype=torch.bool, device=graph.edge_index.device)
    keep_mask[drop_index] = False

    perturbed_graph.edge_index = perturbed_graph.edge_index[:, keep_mask]
    perturbed_graph.edge_attr = perturbed_graph.edge_attr[keep_mask]
    return perturbed_graph


def gen_node_perturbed_graph(
        graph: Data,
        focused_eindex: LongTensor,
        max_virtual_nodes: int = 2) -> Data:
    """
    生成节点扰动图

    """
    x = graph.x
    num_feats = x.size(1)
    device = x.device
    node_ids = torch.arange(x.size(0), device=device)
    perturbed_graph = graph.clone()
    focused_nodes = focused_eindex.unique().to(device)

    perturbed_xs = []
    perturbed_batches = []
    perturbed_edge_indexes = []
    perturbed_edge_attrs = []
    num_virtual_per_batch = [0]

    for batch_index in graph.batch.unique():
        batch_mask = graph.batch == batch_index
        sub_x = x[batch_mask]
        num_sub_nodes = sub_x.size(0)

        # 插入虚拟节点
        num_virtual = np.random.randint(1, max_virtual_nodes + 1)
        num_virtual_per_batch.append(num_virtual)
        perturbed_x = torch.cat([sub_x, torch.zeros(num_virtual, num_feats, device=device)], dim=0)
        perturbed_xs.append(perturbed_x)

        batch = torch.cat([graph.batch[batch_mask], torch.LongTensor([batch_index]*num_virtual)], dim=0)
        perturbed_batches.append(batch)

        # 生成虚拟边（不连接到目标端点）
        allowed_nodes = node_ids[batch_mask]

        allowed_bool = (allowed_nodes.unsqueeze(-1) != focused_nodes.unsqueeze(0)).all(dim=-1)
        allowed_nodes = allowed_nodes[allowed_bool]

        new_edges = []
        for s in range(num_sub_nodes, num_sub_nodes + num_virtual):
            num_connections = np.random.randint(1, 2)
            connections = allowed_nodes[torch.randperm(allowed_nodes.size(0))[:num_connections]]
            new_edges.extend([[s, t] for t in connections])

        if new_edges:
            virtual_edges = torch.tensor(new_edges, device=device, dtype=torch.long).T
            edge_mask = (graph.edge_index[0].unsqueeze(-1) == node_ids[batch_mask].unsqueeze(0)).any(dim=-1)
            new_edge_index = torch.cat([graph.edge_index[edge_mask], virtual_edges], dim=-1)
            virtual_attr = torch.zeros((virtual_edges.size(1), graph.edge_attr.size(1)), device=device, dtype=graph.edge_attr.dtype)
            new_edge_attr = torch.cat([graph.edge_attr[edge_mask], virtual_attr], dim=0)
        else:
            new_edge_index = graph.edge_index
            new_edge_attr = graph.edge_attr
        perturbed_edge_indexes.append(new_edge_index)
        perturbed_edge_attrs.append(new_edge_attr)

    # perturbed_graph.edge_index = new_edge_index
    # perturbed_graph.edge_attr =
    return perturbed_graph


def _worker(
        seed: int,
        graph: Data,
        focused_eindex: LongTensor,
        edge_drop_rate: float,
        max_virtual_nodes: int,
):
    r"""
    生成扰动图
    Args:
        graph: 原始图
        focused_eindex:
        edge_drop_rate: 边删除率
        max_virtual_nodes: 最大虚拟节点数
        seed: 随机种子
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    # decide_flag = np.random.random()

    # if 0 <= decide_flag < 0.4:
    #     print("生成边扰动图")
    #     perturbed_graph = gen_edge_perturbed_graph(graph, focused_eindex, edge_drop_rate)
    #
    # elif 0.4 <= decide_flag < 0.7:
    #     print("生成节点扰动图")
    #     perturbed_graph = gen_node_perturbed_graph(graph, focused_eindex, max_virtual_nodes)
    #
    # else:
    #     print("生成边和节点扰动图")
    #     perturbed_graph = gen_edge_perturbed_graph(graph, focused_eindex, edge_drop_rate)
    #     perturbed_graph = gen_node_perturbed_graph(perturbed_graph, focused_eindex, max_virtual_nodes)
    perturbed_graph = gen_edge_perturbed_graph(graph, focused_eindex, edge_drop_rate)
    return perturbed_graph


# 用循环生成多个扰动图
def gen_perturbed_graphs(
        graph: Data,
        focused_eindex: LongTensor,
        num_samples: int = 10,
        edge_drop_rate: float = 0.2,
        max_virtual_nodes: int = 3) -> list:
    
    # perturbed_graphs = [graph]
    
    # for i in range(num_samples-1):
    for i in range(num_samples):
        if i is 0:
            yield graph
        
        else:
            perturbed_graph = _worker(
                seed=np.random.randint(0, 2 ** 10 - 1),
                graph=graph,
                focused_eindex=focused_eindex,
                edge_drop_rate=edge_drop_rate,
                max_virtual_nodes=max_virtual_nodes
            )
            yield perturbed_graph
        # perturbed_graphs.append(perturbed_graph)

    # return perturbed_graphs


if __name__ == "__main__":
    # 创建示例图
    g = Data(
        x=torch.randn(20, 8),
        edge_index=torch.tensor([[0, 1, 2, 3, 1, 0], [1, 0, 3, 2, 2, 3]], dtype=torch.long)
    )
