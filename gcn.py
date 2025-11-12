import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.nn import functional as F


# 设置常量
RANDOM_STATE = 42
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 设置随机种子函数
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# GCN模型定义
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 评估模型函数
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        preds = model(data).argmax(dim=1)
        true_labels = data.y[mask]
        predicted_labels = preds[mask]
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        accuracy = (true_labels == predicted_labels).sum().item() / len(true_labels)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


# 训练GCN函数
def train_gnn(num_epochs, data, model, optimizer, criterion):
    model.train()
    val_metrics = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # 验证集评估
        val_metrics.append(evaluate(model, data, data.val_mask))

        # 每10个epoch输出一次训练损失和验证集评估指标
        if (epoch + 1) % 10 == 0:
            val_metrics_last = val_metrics[-1]
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, '
                  f'Val Accuracy: {val_metrics_last["accuracy"]:.4f}, '
                  f'Val Precision: {val_metrics_last["precision"]:.4f}, '
                  f'Val Recall: {val_metrics_last["recall"]:.4f}, '
                  f'Val F1 Score: {val_metrics_last["f1_score"]:.4f}')

    return val_metrics


def main():
    # 设置随机种子
    set_seeds(RANDOM_STATE)

    # 加载数据
    elliptic_txs_features = pd.read_csv('elliptic_txs_features.csv', header=None)
    elliptic_txs_classes = pd.read_csv('elliptic_txs_classes.csv')
    elliptic_txs_edgelist = pd.read_csv('elliptic_txs_edgelist.csv')

    # 数据预处理
    elliptic_txs_features.columns = ['txId'] + [f'V{i}' for i in range(1, 167)]
    tx_id_mapping = {tx_id: idx for idx, tx_id in enumerate(elliptic_txs_features['txId'])}
    edges_with_features = elliptic_txs_edgelist[
        (elliptic_txs_edgelist['txId1'].isin(tx_id_mapping.keys())) &
        (elliptic_txs_edgelist['txId2'].isin(tx_id_mapping.keys()))
        ]
    edges_with_features['Id1'] = edges_with_features['txId1'].map(tx_id_mapping)
    edges_with_features['Id2'] = edges_with_features['txId2'].map(tx_id_mapping)

    edge_index = torch.tensor(edges_with_features[['Id1', 'Id2']].values.T, dtype=torch.long)
    node_features = torch.tensor(elliptic_txs_features.drop(columns=['txId']).values, dtype=torch.float)

    le = LabelEncoder()
    class_labels = le.fit_transform(elliptic_txs_classes['class'])
    node_labels = torch.tensor(class_labels, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, y=node_labels).to(DEVICE)

    # 创建训练集、验证集和测试集掩码
    known_mask = (data.y != -1)  # 假设-1表示未知标签
    num_known_nodes = known_mask.sum().item()
    permutations = torch.randperm(num_known_nodes)
    train_size = int(0.8 * num_known_nodes)
    val_size = int(0.1 * num_known_nodes)
    test_size = num_known_nodes - train_size - val_size

    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_indices = known_mask.nonzero(as_tuple=True)[0][permutations[:train_size]]
    val_indices = known_mask.nonzero(as_tuple=True)[0][permutations[train_size:train_size + val_size]]
    test_indices = known_mask.nonzero(as_tuple=True)[0][permutations[train_size + val_size:]]

    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True

    # 初始化模型、优化器和损失函数
    model = GCN(num_node_features=data.num_features, num_classes=len(le.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    train_val_metrics = train_gnn(NUM_EPOCHS, data, model, optimizer, criterion)

    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_metrics = evaluate(model, data, data.test_mask)
        print(
            f'Test Acc: {test_metrics["accuracy"]:.4f} - Prec: {test_metrics["precision"]:.4f} - Rec: {test_metrics["recall"]:.4f} - F1: {test_metrics["f1_score"]:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'trained_model.pth')
    print("GCN trained_model saved as trained_model.pth")


if __name__ == "__main__":
    main()