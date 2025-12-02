from torch import nn
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
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
        torch.save(model.state_dict(), f'ckpt/classifier_Instruct/classifier_model_epoch{epoch}.pth')
              
    return history

def evaluate_model(model, test_loader, device=torch.device("cuda:0")):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"正在使用设备: {device} 进行评估...")
    
    model = model.to(device)
    model.eval()  # 切换到评估模式 (关闭 Dropout, BN 使用移动平均值)
    
    all_preds = []
    all_labels = []
    
    # 2. 禁用梯度计算以节省显存并加速
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 获取预测结果
            # 假设输出是 (Batch_Size, 2) 的 Logits，使用 argmax 获取类别索引 (0 或 1)
            # 如果你的模型输出层是 Sigmoid (Batch_Size, 1)，请将下行改为: preds = (outputs > 0.5).float()
            _, preds = torch.max(outputs, 1)
            
            # 将结果移回 CPU 并转为 numpy 用于 sklearn 计算
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 3. 计算指标
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    # 提取 TN, FP, FN, TP (仅适用于二分类)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # 防止某些极端情况（如测试集中只有一类数据）导致解包失败
        tn, fp, fn, tp = 0, 0, 0, 0 
    
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    
    # 4. 打印结果
    print("\n" + "="*40)
    print("           模型评估报告           ")
    print("="*40)
    
    print(f"\n[混淆矩阵 Confusion Matrix]")
    print(f"{'':<10} {'预测 0':<10} {'预测 1':<10}")
    print(f"{'真实 0':<10} {cm[0,0]:<10} {cm[0,1]:<10}")
    print(f"{'真实 1':<10} {cm[1,0]:<10} {cm[1,1]:<10}")
    
    if cm.shape == (2, 2):
        print(f"\n详情:")
        print(f"  True Negatives (TN): {tn}")
        print(f"  False Positives (FP): {fp}")
        print(f"  False Negatives (FN): {fn}")
        print(f"  True Positives (TP): {tp}")

    print("\n[核心指标 Key Metrics]")
    print(f"  准确率 (Accuracy) : {acc:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}  (查准率)")
    print(f"  召回率 (Recall)   : {recall:.4f}  (查全率/敏感度)")
    print(f"  F1 分数 (F1-Score): {f1:.4f}")
    
    print("\n[详细分类报告 Classification Report]")
    print(classification_report(all_labels, all_preds, target_names=['Class 0', 'Class 1'], digits=4))
    print("="*40)

    # 返回指标字典，方便后续记录日志
    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":
    
    run_train = False
    dataset_file = 'data/hidden_states_PS_O-Instruct.npy'  # <--- 请修改这里为你的文件名
    ckpt_path = 'ckpt/classifier_Instruct/classifier_model_epoch19.pth'
    evaluate_result_fig = "data/training_history_Instruct.png"
    BATCH_SIZE = 128
    EPOCHS = 20
    DEVICE = torch.device('cuda:0')

    train_loader, val_loader, test_loader = load_and_split_data(data_file=dataset_file, batch_size=BATCH_SIZE)   
    model = Classifier(input_dim=4096, num_classes=2)

    if not run_train:
        # 2. 加载预训练模型权重
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        evaluate_model(model, test_loader, device=DEVICE)

    else:
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

        plt.savefig(evaluate_result_fig)

