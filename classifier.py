from torch import nn
from utils import *
from sklearn.model_selection import train_test_split# , confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, input_dim=4096, num_classes=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(128, num_classes)


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.classifier(x)
        return out
    
def load_and_split_data(data_file, batch_size=64):
    data = np.load(data_file, allow_pickle=True).item()
    X = np.array(data["hidden_states"], dtype=np.float32)
    y = np.array(data["labels"], dtype=np.int64)

    print(f"Total samples: {len(y)}, Feature dim: {X.shape[1]}")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, epochs, device=torch.device("cuda:0")):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # 记录
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
              
    return history

# def evaluate_on_test(model, test_loader, device='cuda'):
#     """
#     在测试集上评估模型性能，并绘制混淆矩阵
#     """
#     model.eval()  # 切换到评估模式 (关闭 Dropout, 锁定 BatchNorm)
    
#     all_preds = []
#     all_labels = []
    
#     print(f"Start testing on {device}...")
    
#     # 1. 推理 (Inference)
#     with torch.no_grad():  # 不需要计算梯度，节省显存
#         for inputs, labels in test_loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
            
#             outputs = model(inputs)
            
#             # 获取预测类别 (0 或 1)
#             # outputs shape: [batch_size, 2] -> 取 max 的索引
#             _, preds = torch.max(outputs, 1)
            
#             # 收集结果用于后续计算
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # 2. 计算指标
#     cm = confusion_matrix(all_labels, all_preds)
    
#     # 3. 可视化混淆矩阵
#     plt.figure(figsize=(8, 6))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
#     disp.plot(cmap=plt.cm.Blues, values_format='d')
#     plt.title('Confusion Matrix (Test Set)')
#     plt.savefig('data/test_confusion_matrix.png')
    
#     return float(np.mean(np.array(all_preds) == np.array(all_labels)))


if __name__ == "__main__":
    
    dataset_file = 'data/hidden_states_PS_O.npy'  # <--- 请修改这里为你的文件名
    BATCH_SIZE = 64
    EPOCHS = 20
    DEVICE = torch.device('cuda:0')

    train_loader, val_loader, test_loader = load_and_split_data(data_file=dataset_file, batch_size=BATCH_SIZE)   
    model = Classifier(input_dim=4096, num_classes=2)
    
    # 3. 开始训练
    history = train(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
    
    # 4. 可视化损失变化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], color='orange', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig('data/training_history.png')

    # final_acc = evaluate_on_test(model, test_loader, device=DEVICE)
    # print(f"Final Test Accuracy: {final_acc * 100:.2f}%")


