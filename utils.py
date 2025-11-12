import torch
from torch import Tensor
from torch_geometric.data import Data


def extract_subgraph(
    graph: Data, 
    selection: Tensor):
    """
    从原始图中选择解释性子图，并重新映射其中的节点到新ID
    Args:
        graph (Data): 原始图
        selection (Tensor): 解释性子图的选择
    """
    subgraph = graph.clone()

    # 检索解释性子图的属性
    subgraph.edge_index = graph.edge_index[:, selection]
    subgraph.edge_attr = graph.edge_attr[selection]
    return subgraph


def extract_subgraph_backup(
    graph: Data, 
    selection: Tensor,
    batch_size :int = 32
    ):
    """
    从原始图中选择解释性子图，并重新映射其中的节点到新ID
    Args:
        graph (Data): 原始图
        selection (Tensor): 解释性子图的选择
    """
    subgraph = graph.clone()

    # 检索解释性子图的属性
    subgraph.edge_index = graph.edge_index[:, selection]
    subgraph.edge_attr = graph.edge_attr[selection]
    
    sub_nodes = torch.unique(subgraph.edge_index)
    # 节点特征
    subgraph.x = graph.x[sub_nodes]
    subgraph.batch = graph.batch[sub_nodes]

    # if subgraph.batch.unique().size(0) < batch_size:
    #     return None
    
    src_nodes, _ = graph.edge_index
    
    # 重新映射解释性子图中的节点到新ID

    node_idx = src_nodes.new_full((graph.num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=src_nodes.device)
    subgraph.edge_index = node_idx[subgraph.edge_index]
    return subgraph



if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data

    # 1. 创建原始图
    data = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long),
        edge_attr=torch.randn(6, 2)
    )
    data.batch = torch.zeros(4, dtype=torch.long)
    print("Original Graph:")
    print(data)

    # 2. 选择解释性子图
    selection = torch.tensor([True, False, True, False, False, False])
    subgraph = extract_subgraph(data, selection)

    print("\nExplanatory Subgraph:")
    print(subgraph)
