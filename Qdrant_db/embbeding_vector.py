import torch
from torchvision import transforms
from ThinResNet18 import ThinResNet18  # 你的模型
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from ThinResNet18 import ThinResNet18
from CustomDataset import Cocodataset
import pandas as pd
# 1. 定義 transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# 2. 初始化 dataset
dataset = Cocodataset(coco_json=r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train\_annotations.coco.json", img_dir=r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\train", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 3. 載入模型
model = ThinResNet18(channels=[16,32,64,128])
model.load_state_dict(torch.load(r"C:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\models\ThinResNet18_200\ThinResNet18_l_run4.pth", map_location="cpu"))
model.eval()

# 去掉 fc 層，只保留 embedding
embedding_model = torch.nn.Sequential(
    model.conv1, model.bn1, model.relu, model.maxpool,
    model.layer1, model.layer2, model.layer3, model.layer4,
    model.avgpool, torch.nn.Flatten()
)

# 4. 生成 embedding
embeddings = []
payloads = []

with torch.no_grad():
    for idx in range(len(dataset)):
        image, label_idx = dataset[idx]

        # embedding
        vec = embedding_model(image.unsqueeze(0)).squeeze().numpy()  # shape=(128,)

        # 取得 COCO JSON info
        img_id = dataset.img_ids[idx]
        img_info = dataset.coco.loadImgs(img_id)[0]
        
        # 找該圖片的 annotation
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        
        if len(anns) > 0:
            largest_ann = max(anns, key=lambda x: x['area'])
            cat_id = largest_ann['category_id']
            cat_name = dataset.cat_id_to_name[cat_id]
        else:
            cat_id = 0
            cat_name = "unknown"
        payload = {
            "file_name": img_info['file_name'],
            "height": img_info['height'],
            "width": img_info['width'],
            "category_id": cat_id,
            "category_name": cat_name
        }
        embeddings.append(vec)
        payloads.append(payload)

df = pd.DataFrame(payloads)
df = df[df['category_id'] != 0]

df["category_chinese_name"] = df["category_id"].map(dataset.cat_id_to_chinese_name)
df.drop(columns=["id"],inplace=True)

df.columns
df.to_csv(r"C:\Users\user\Desktop\謙恩的文件\Tea_Disease_Project\Qdrant_db\ThinResNet18_run4_payloads.csv",index=False)
vectors = pd.Series(embeddings)
vectors = vectors[df['category_id'] != 0]
vectors.to_csv(r"C:\Users\user\Desktop\謙恩的文件\Tea_Disease_Project\Qdrant_db\ThinResNet18_run4_vectors.csv",index=False)