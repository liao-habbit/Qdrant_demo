import os
import ctypes
from collections import Counter, defaultdict
from CustomDataset import Cocodataset
import platform

if platform.system() == "Windows":
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision import transforms
import torch.nn as nn
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2
from ThinResNet18 import ThinResNet18
from SEResNet18 import ThinSEResNet18

def plot_accuracy(csv_path, fig_save_path, column_name="acc",show_plot=False):
    df = pd.read_csv(csv_path)
    summary = (
        df
        .groupby(["model", "epoch"])[column_name]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / summary["count"]**0.5

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 6))

    for model_name, g in summary.groupby("model"):
        plt.plot(g["epoch"], g["mean"], label=model_name)
        plt.fill_between(
            g["epoch"],
            g["mean"] - g["se"],
            g["mean"] + g["se"],
            alpha=0.25
        )

    plt.xlabel("Epoch")
    if column_name == "acc":
        plt.ylabel("Training Accuracy")
        plt.title("Model Training Accuracy (Mean ± SE)")
    else:
        plt.ylabel("Validation Accuracy")
        plt.title("Model Validation Accuracy (Mean ± SE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_save_path)
    if show_plot:
        plt.show()
    plt.close()

# example usage:
# plot_accuracy(
#     csv_path=r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\multi_run_history.csv",
#     fig_save_path=r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\test.png",
#     column_name="val_acc",
#     show_plot=False
# )

train_json = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train\_annotations.coco.json"
img_dir = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train"
train_dataset = Cocodataset(train_json, img_dir)

# 計算訓練資料集中各類別的出現次數
# 儲存每張圖的類別
all_labels = []
for idx in range(len(train_dataset)):
    _, label = train_dataset[idx]  # train_dataset[idx] 回傳 (image, label)
    all_labels.append(label.item())

# 計算每個類別出現次數
label_counts = Counter(all_labels)

# 對照中文名稱
label_counts_chinese = {
    train_dataset.cat_id_to_chinese_name.get(idx+1, str(idx+1)): count
    for idx, count in label_counts.items()
}

# 排序
label_counts_chinese = dict(sorted(label_counts_chinese.items(), key=lambda x: x[1], reverse=True))

font_path = r"C:\Windows\Fonts\msjh.ttc"  # 微軟正黑體
font_prop = font_manager.FontProperties(fname=font_path)

rcParams["font.family"] = font_prop.get_name()
rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題
df_counts = pd.DataFrame(list(label_counts_chinese.items()), columns=["類別", "數量"])
print(df_counts)
plt.figure(figsize=(10, 8))
bars = plt.barh(df_counts["類別"], df_counts["數量"], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("出現次數")
plt.ylabel("病害類別")
plt.title("訓練資料集病害出現次數分布")
for bar in bars:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    plt.text(
        width + 5,      # 文字往右一點
        y,
        f"{int(width)}",
        va="center",
        fontsize=10
    )

plt.tight_layout()
plt.show()





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 假設是 ResNet18，num_classes = 你的分類數
num_classes = 11
model = resnet18(num_classes=num_classes)
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

batch_size = 32
valid_json = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\valid\_annotations.coco_valid.json"
valid_dir = r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\valid"
valid_dataset = Cocodataset(valid_json, valid_dir, transform=resize_transform)
val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
valid_dataset.cat_id_to_chinese_name

def show_cm(checkpoint_path, dataset, model, save=False, save_path=False,show_plot=False):
    """
    顯示 confusion matrix，x/y 軸對應中文類別名稱。

    checkpoint_path: 模型權重路徑
    dataset: Cocodataset 物件
    cat_id_to_name: {cat_id: 中文名稱} 對應字典
    save: 是否儲存圖
    save_path: 儲存路徑
    """
    all_labels = []
    all_preds = []
    # 初始化模型
    model = model
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 調整最後一層
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 計算 confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # -------------------------------
    # 準備類別名稱
    idx_to_cat_id = {idx: cat_id for cat_id, idx in dataset.cat_id_to_idx.items()}
    labels_names = [
        dataset.cat_id_to_chinese_name.get(idx_to_cat_id[i], str(idx_to_cat_id[i]))
        for i in range(len(idx_to_cat_id))
    ]

    # -------------------------------
    # 繪圖
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel("預測標籤", fontsize=12)
    plt.ylabel("真實標籤", fontsize=12)
    plt.title("混淆矩陣", fontsize=14)
    plt.xticks(np.arange(len(labels_names)), labels_names, rotation=90, ha="right", fontsize=10)
    plt.yticks(np.arange(len(labels_names)), labels_names, fontsize=10)

    # 在格子中顯示數字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

# 使用範例
# checkpoint_path = r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\resnet18_400\resnet18_run5_400.pth"
# show_cm(checkpoint_path, valid_dataset, model = model)
# show_cm(checkpoint_path, valid_dataset, model = model , save=True,save_path=r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\figure\ResNet18_400\ResNet18_run_400.png")



def batch_cm(base_model_dir, base_fig_dir, model_fn, sizes, runs):
    cm_jobs = []

    for size in sizes:
        for run in runs:
            cm_jobs.append({
                "checkpoint": os.path.join(
                    base_model_dir,
                    f"ThinResNet18_{size}_run{run}.pth"
                ),
                "out_dir": base_fig_dir,
                "model": model_fn(size),
                "filename": f"ThinResNet18_{size}_run{run}.png"
            })
            
    for job in cm_jobs:
        out_path = os.path.join(job["out_dir"], job["filename"])
        show_cm(
            checkpoint_path=job["checkpoint"],
            dataset=valid_dataset,
            model=job["model"],
            save=True,
            save_path=out_path
        )
# 使用範例
# sizes = ["ss", "s", "m", "l", "h"]
# runs = [1, 2, 3, 4, 5]
# base_model_dir = r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinSEResNet18"
# base_fig_dir   = r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\figure\ThinSEResNet18_200"
# model_fn = lambda size: ThinSEResNet18(num_classes=num_classes, channels={
#     "ss": [2, 4, 8, 16],
#     "s":  [4, 8, 16, 32],
#     "m":  [8, 16, 32, 64],
#     "l":  [16, 32, 64, 128],
#     "h":  [32, 64, 128, 256]
# }[size], se_reduction=2)
# batch_cm(base_model_dir, base_fig_dir, model_fn, sizes, runs)

# misclassified = []  # 存 (image, true_label, pred_label)
# model.eval()
# with torch.no_grad():
#     for images, labels in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         preds = torch.argmax(outputs, dim=1)
#         for i in range(len(labels)):
#             if preds[i] != labels[i]:
#                 misclassified.append((
#                     images[i].cpu(),
#                     labels[i].item(),
#                     preds[i].item()
#                 ))
# import matplotlib.pyplot as plt

# def show_images(samples, idx_to_cat, n=5):
#     plt.figure(figsize=(15, 3))
#     for i in range(n):
#         img, true, pred = samples[i]
#         img = img.permute(1, 2, 0)  # C,H,W → H,W,C

#         plt.subplot(1, n, i+1)
#         plt.imshow(img)
#         plt.axis("off")
#         plt.title(f"T: {idx_to_cat[true]}\nP: {idx_to_cat[pred]}")
#     plt.show()

# show_images(misclassified, valid_dataset.cat_id_to_idx, n=5)

# 假設已經有 idx_to_cat
# idx_to_cat = {0: "Brown Spot", 1: "Leaf Blight", 2: "Healthy", ...}

# target_true = 10  # 真實類別 index
# target_pred = 6  # 預測錯誤的類別 index

# pair_errors = []

# model.eval()
# with torch.no_grad():
#     for images, labels in val_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         preds = torch.argmax(outputs, dim=1)

#         for i in range(len(labels)):
#             if labels[i] == target_true-1 and preds[i] == target_pred-1:
#                 pair_errors.append(images[i].cpu())

# # 顯示前 5 張錯誤影像
# plt.figure(figsize=(12, 8))
# for i in range(min(5, len(pair_errors))):
#     img = pair_errors[i].permute(1, 2, 0)  # C,H,W -> H,W,C
#     plt.subplot(1, 5, i+1)
#     plt.imshow(img)
#     plt.axis("off")
# plt.show()
import random
import torch.nn.functional as F
# 隨機抽一張 validation dataset


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def __call__(self, x, class_idx):
        self.model.zero_grad()
        output = self.model(x)
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients      # [B,C,H,W]
        acts = self.activations     # [B,C,H,W]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (x.size(3), x.size(2)))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

def overlay_cam(img, cam):
    # 1. Tensor -> numpy, [H,W,C]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    
    # 2. 將值縮放到 0~255
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # 3. 生成 heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 4. 疊圖
    overlay = 0.7 * img + 0.3 * heatmap
    return overlay.astype(np.uint8)

models = {
    "ThinResNet18_ss":lambda: ThinResNet18(num_classes=num_classes, channels=[2,4,8,16]),
    "ThinResNet18_s":lambda: ThinResNet18(num_classes=num_classes, channels=[4,8,16,32]),
    "ThinResNet18_m": lambda: ThinResNet18(num_classes=num_classes, channels=[8,16,32,64]),
    "ThinResNet18_l": lambda: ThinResNet18(num_classes=num_classes, channels=[16,32,64,128]),
    "ThinResNet18_h": lambda: ThinResNet18(num_classes=num_classes, channels=[32,64,128,256]),
    "ResNet18": lambda: resnet18(num_classes=num_classes)
}

ckpt_paths = {
    "ThinResNet18_ss": r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet_200\ThinResNet18_ss_run4.pth",
    "ThinResNet18_s":  r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet_200\ThinResNet18_s_run4.pth",
    "ThinResNet18_m":  r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet_200\ThinResNet18_m_run4.pth",
    "ThinResNet18_l":  r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet_200\ThinResNet18_l_run4.pth",
    "ThinResNet18_h":  r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet_200\ThinResNet18_h_run4.pth",
    "ResNet18":        r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\resnet18_run4.pth",
}


def load_model(model_fn, ckpt_path, device):
    model = model_fn().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt  # 直接是 state_dict
    model.load_state_dict(state_dict)
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

loaded_models = {}

for name, model_fn in models.items():
    ckpt_path = ckpt_paths[name]
    if not os.path.exists(ckpt_path):
        print(f"❌ Missing checkpoint: {name}")
        continue
    loaded_models[name] = load_model(model_fn, ckpt_path, device)
    print(f"✅ Loaded {name}")

target_layer_map = {
    "ResNet18": lambda m: m.layer4[-1].conv2,
    "ThinResNet18_ss": lambda m: m.layer[-1].conv1,
    "ThinResNet18_s":  lambda m: m.layer1[-1].conv1,
    "ThinResNet18_m":  lambda m: m.layer4[-1].conv2,
    "ThinResNet18_l":  lambda m: m.layer1[-1].conv1,
    "ThinResNet18_h":  lambda m: m.layer1[-1].conv1,
}
plt.figure(figsize=(16, 4))
from collections import defaultdict

class_to_indices = defaultdict(list)

for i, (_, label) in enumerate(valid_dataset):
    label = int(label)        
    class_to_indices[label].append(i)

def plot_cam_for_class(target_class):
    if target_class not in class_to_indices or len(class_to_indices[target_class]) == 0:
        print(f"No image found for class {target_class}")
    else:
        idx = random.choice(class_to_indices[target_class])
        img, true_label = valid_dataset[idx]   
    input_tensor = img.unsqueeze(0).to(device)
    plt.figure(figsize=(4*len(loaded_models), 4))
    for i, (name, model) in enumerate(loaded_models.items()):
        # 計算 CAM
        cam = GradCAM(model, target_layer_map[name](model))
        cam_map = cam(input_tensor, target_class)
        # 疊圖
        overlay = overlay_cam(img, cam_map)
        # 預測類別
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_class = outputs.argmax(dim=1).item()
        plt.subplot(1, len(loaded_models), i + 1)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"{name}\nT: {true_label}, P: {pred_class}")
    return plt.show()

model = loaded_models["ThinResNet18_l"]
layer = target_layer_map["ThinResNet18_l"](model)
def visualize_layer_channels(model, layer, channels=16, steps=1000, lr=0.001):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    activations = {}
    def hook_fn(module, input, output):
        activations["feat"] = output
    handle = layer.register_forward_hook(hook_fn)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    for k in range(channels):
        img = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([img], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            _ = model(img)
            loss = -activations["feat"][0, k].mean()
            loss.backward()
            optimizer.step()
            img.data.clamp_(0, 1)
        vis = img.detach().cpu()[0].permute(1, 2, 0)
        axes[k].imshow(vis)
        axes[k].set_title(f"ch {k}")
        axes[k].axis("off")
    handle.remove()
    plt.tight_layout()
    plt.show()

import torchvision.models as models
import torchvision.transforms as T


# ---------------------------
# 1️⃣ 選擇模型
# ---------------------------
model = models.resnet18(pretrained=True)
model.eval()

# ---------------------------
# 2️⃣ 選擇目標層與通道
# ---------------------------
target_layer = model.layer3[0].conv1  # 例如 ResNet layer3 的第一個 block 的 conv1
target_channel = 10                    # 想要可視化的通道

# ---------------------------
# 3️⃣ 建立 forward hook 捕捉中間層輸出
# ---------------------------
activations = {}

def hook_fn(module, input, output):
    activations['feat'] = output

handle = target_layer.register_forward_hook(hook_fn)

# ---------------------------
# 4️⃣ 初始化輸入影像
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)

# ---------------------------
# 5️⃣ 梯度上升參數
# ---------------------------
lr = 0.1
iterations = 50

# ---------------------------
# 6️⃣ 梯度上升
# ---------------------------
for i in range(iterations):
    model.zero_grad()
    
    _ = model(image)  # forward pass，hook 會抓到 target_layer
    
    # 取目標通道平均作為 loss
    loss = -activations['feat'][0, target_channel].mean()  # 負號做梯度上升
    loss.backward()
    
    # 更新影像
    image.data -= lr * image.grad.data
    
    # 梯度清零
    image.grad.data.zero_()
    
    # 簡單正則化：限制像素在 0~1
    image.data = torch.clamp(image.data, 0, 1)

# ---------------------------
# 7️⃣ 移除 hook
# ---------------------------
handle.remove()

# ---------------------------
# 8️⃣ 顯示生成影像
# ---------------------------
img = image.detach().squeeze().permute(1, 2, 0).cpu().numpy()
plt.imshow(img)
plt.axis('off')
plt.title(f'ResNet18 Layer3[0].conv1 Channel {target_channel}')
plt.show()
