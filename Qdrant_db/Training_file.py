from SEResNet18 import ThinSEResNet18
from ThinResNet18 import ThinResNet18
from ThinCBAMResNet18 import ThinCBAMResNet18
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from CustomDataset import Cocodataset, SampledDataset, SampledDataset_Origin


# ---------------- 定義要訓練的模型 ----------------
num_classes = 11 

# ThinSEResNet18 with SE blocks
# models_to_train = {
#     "ThinSEResNet18_ss":lambda: ThinSEResNet18(num_classes=num_classes, channels=[2,4,8,16],se_reduction=2),
#     "ThinSEResNet18_s":lambda: ThinSEResNet18(num_classes=num_classes, channels=[4,8,16,32],se_reduction=2),
#     "ThinSEResNet18_m": lambda: ThinSEResNet18(num_classes=num_classes, channels=[8,16,32,64],se_reduction=2),
#     "ThinSEResNet18_l": lambda: ThinSEResNet18(num_classes=num_classes, channels=[16,32,64,128],se_reduction=2),
#     "ThinSEResNet18_h": lambda: ThinSEResNet18(num_classes=num_classes, channels=[32,64,128,256],se_reduction=2)
# }

# ThinResNet18 without SE blocks
# models_to_train = {
#     "ThinResNet18_ss":lambda: ThinResNet18(num_classes=num_classes, channels=[2,4,8,16]),
#     "ThinResNet18_s":lambda: ThinResNet18(num_classes=num_classes, channels=[4,8,16,32]),
#     "ThinResNet18_m": lambda: ThinResNet18(num_classes=num_classes, channels=[8,16,32,64]),
#     "ThinResNet18_l": lambda: ThinResNet18(num_classes=num_classes, channels=[16,32,64,128]),
#     "ThinResNet18_h": lambda: ThinResNet18(num_classes=num_classes, channels=[32,64,128,256])
# }

# ThinCBAMResNet18 with CBAM blocks
models_to_train = {
    "ThinCBAMResNet18_ss":lambda: ThinCBAMResNet18(num_classes=num_classes, channels=[2,4,8,16],cbam_reduction=2),
    "ThinCBAMResNet18_s":lambda: ThinCBAMResNet18(num_classes=num_classes, channels=[4,8,16,32],cbam_reduction=2),
    "ThinCBAMResNet18_m": lambda: ThinCBAMResNet18(num_classes=num_classes, channels=[8,16,32,64],cbam_reduction=2),
    "ThinCBAMResNet18_l": lambda: ThinCBAMResNet18(num_classes=num_classes, channels=[16,32,64,128],cbam_reduction=2),
    "ThinCBAMResNet18_h": lambda: ThinCBAMResNet18(num_classes=num_classes, channels=[32,64,128,256],cbam_reduction=2)
}
# ---------------- 範例: 使用方式 ----------------
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

extra_transform = transforms.Compose([                                              # 對於擴增資料實施 水平、垂直、顏色抖動等翻轉
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0.2,hue=0.1)
])


# 假設 train_dataset 已建立
target_per_class = {i: 400 for i in range(1, num_classes+1)}
train_json = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train\_annotations.coco.json"
train_dir = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train"
train_dataset = Cocodataset(train_json, train_dir, transform=resize_transform)

sampled_dataset_base = SampledDataset_Origin(
    original_dataset=train_dataset,
    target_per_class=target_per_class,
    extra_transform=None
)

sampled_dataset = SampledDataset(
    original_dataset=train_dataset,
    target_per_class=target_per_class,
    extra_transform=extra_transform
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
valid_json = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\valid\_annotations.coco_valid.json"
valid_dir = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\valid"
valid_dataset = Cocodataset(valid_json, valid_dir, transform=resize_transform)

# ---------------- DataLoader ----------------
batch_size = 32
train_loader = DataLoader(sampled_dataset_base,batch_size=batch_size,shuffle=True)
# train_loader_aug = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=100, dataset_name="Train"):
    train_history = {"loss": [], "acc": [], "val_acc": []}
    for epoch in range(num_epochs):
        # -------- 訓練模式 --------
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        # -------- 驗證模式 --------
        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_corrects += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_corrects / val_total
        # 儲存歷史
        train_history["loss"].append(epoch_loss)
        train_history["acc"].append(epoch_acc)
        train_history["val_acc"].append(val_acc)
        print(f"[{dataset_name}] Epoch {epoch+1}/{num_epochs} | " f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
    return train_history

# 建立模型
num_classes = len(train_dataset.cat_id_to_idx)
num_epochs = 50
num_runs = 5
lr = 1e-4

save_root = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results"
os.makedirs(save_root, exist_ok=True)
os.makedirs(os.path.join(save_root, "models"), exist_ok=True)

all_history = []

for model_name, model_fn in models_to_train.items():
    print(f"\n==============================")
    print(f"🚀 Model: {model_name}")
    print(f"==============================")
    for run in range(1, num_runs + 1):
        print(f"\n▶ Run {run}/{num_runs}")
        # --- 建立模型 ---
        model = model_fn().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # --- 訓練 ---
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            dataset_name=f"{model_name}_run{run}"
        )
        # --- 紀錄 history ---
        df = pd.DataFrame(history)
        df["model"] = model_name
        df["run"] = run
        df["epoch"] = df.index + 1
        all_history.append(df)
        # --- 儲存模型 ---
        model_path = os.path.join(
            save_root,
            "models",
            f"{model_name}_run{run}.pth"
        )
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved: {model_path}")

# ================= 匯出 CSV =================
all_history_df = pd.concat(all_history, ignore_index=True)
csv_path = os.path.join(save_root, "multi_run_history_test.csv")
all_history_df.to_csv(csv_path, index=False)

print(f"\n✅ All training history saved to:\n{csv_path}")
















