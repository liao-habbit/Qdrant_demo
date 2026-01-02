# CustomDataset.py
# resolve Windows DLL loading issue for PyTorch
import os
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

# python code for custom COCO dataset handling
from PyQt6.QtWidgets import QApplication
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import random

class Cocodataset(Dataset):
    def __init__(self, coco_json, img_dir, transform=None):
        self.coco = COCO(coco_json)                                                                       #讀取json
        self.img_dir = img_dir                                                                            #讀取影像資料夾
        self.transform = transform                                                                        #資料轉換
        self.img_ids = self.coco.getImgIds()                                                              #建立影像編號 
        self.cat_ids = [cat['id'] for cat in self.coco.loadCats(self.coco.getCatIds()) if cat['id'] != 0] #建立類別編號
        self.cat_id_to_idx = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_id_to_name={cat['id']: cat['name'] for cat in self.coco.loadCats(self.cat_ids)}
        self.cat_id_to_chinese_name={
            1: "藻類",
            2: "赤葉枯病",
            3: "藻斑病",
            4: "枝枯病",
            5: "網餅病",
            6: "茶餅病",
            7: "地衣",
            8: "茶髮狀病",
            9: "苔癬",
            10: "輪班病",
            11: "煤煙病"
        }
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            label = 0
        else:
            largest_ann = max(anns, key=lambda x: x['area'])
            label = self.cat_id_to_idx[largest_ann['category_id']]
            # 選擇 label 的 標準可以自行定義
            # 1. 使用第一個 annotation 的類別
            # cat_id = anns[0]['category_id']
            # label = self.cat_id_to_idx[cat_id]
            # 2. 使用面積最大的 annotation 的類別 (如上)
            # 3. 挑選面積最大的 object 類別作為 label
            # 4. 多標籤分類 (需修改 model 輸出與 loss function)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
    
class SampledDataset_Origin(Dataset):
    def __init__(self, original_dataset, target_per_class, extra_transform=None):
        """
        original_dataset: 原本的 CocoDataset
        target_per_class: dict {cat_id: 每個類別需要的照片數量}
        extra_transform: 額外 transform（選擇性）
        """
        self.original_dataset = original_dataset
        self.extra_transform = extra_transform
        self.sampled_img_ids = self.sample_images(target_per_class)
    def sample_images(self, target_per_class):
        # 建立 per-class → img_ids 映射表
        category_to_img_ids = {}
        for img_id in self.original_dataset.img_ids:
            ann_ids = self.original_dataset.coco.getAnnIds(imgIds=img_id)
            anns = self.original_dataset.coco.loadAnns(ann_ids)
            if len(anns) == 0:
                continue
            cat_id = anns[0]['category_id']
            if cat_id == 0:
                continue
            category_to_img_ids.setdefault(cat_id, []).append(img_id)
        # 按照 target_per_class 抽取，不足則全部使用
        sampled_img_ids = []
        for cat_id, img_ids in category_to_img_ids.items():
            target = target_per_class.get(cat_id, len(img_ids))
            # 如果資料量 ≥ 目標 → random.sample
            if len(img_ids) >= target:
                sampled = random.sample(img_ids, target)
            # 如果資料量不足 → 全部使用 (不做 oversample)
            else:
                sampled = img_ids.copy()
            sampled_img_ids.extend(sampled)
        return sampled_img_ids
    def __len__(self):
        return len(self.sampled_img_ids)
    def __getitem__(self, idx):
        img_id = self.sampled_img_ids[idx]
        img_info = self.original_dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.original_dataset.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        # annotation
        ann_ids = self.original_dataset.coco.getAnnIds(imgIds=img_id)
        anns = self.original_dataset.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            raise ValueError("Image has no annotation")
        cat_id = anns[0]['category_id']
        label = self.original_dataset.cat_id_to_idx[cat_id]
        # 原 transform（如 resize）
        if self.original_dataset.transform:
            image = self.original_dataset.transform(image)
        # 額外 transform（給重抽樣資料）
        if self.extra_transform:
            image = self.extra_transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
class SampledDataset(Dataset):
    def __init__(self, original_dataset, target_per_class, extra_transform=None):
        """
        original_dataset: 原本的 Cocodataset
        target_per_class: dict {cat_id: 每個類別目標數量}
        extra_transform: 針對重抽樣資料的 transform
        """
        self.original_dataset = original_dataset
        self.extra_transform = extra_transform
        self.sampled_img_ids = self.oversample(target_per_class)
    def oversample(self, target_per_class):
        # 建立類別對應的圖片 ID
        category_to_img_ids = {}
        for img_id in self.original_dataset.img_ids:
            ann_ids = self.original_dataset.coco.getAnnIds(imgIds=img_id)
            anns = self.original_dataset.coco.loadAnns(ann_ids)
            if len(anns) > 0:
                cat_id = anns[0]['category_id']
                if cat_id == 0:
                    continue
                category_to_img_ids.setdefault(cat_id, []).append(img_id)
        # 重抽樣補足每個類別
        sampled_img_ids = []
        for cat_id, img_ids in category_to_img_ids.items():
            target = target_per_class.get(cat_id, len(img_ids))
            if len(img_ids) >= target:
                sampled = random.sample(img_ids, target)
            else:
                sampled = img_ids.copy()
                while len(sampled) < target:
                    sampled.extend(random.choices(img_ids, k=target - len(sampled)))
            sampled_img_ids.extend(sampled)
        return sampled_img_ids
    def __len__(self):
        return len(self.sampled_img_ids)
    def __getitem__(self, idx):
        img_id = self.sampled_img_ids[idx]
        img_info = self.original_dataset.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.original_dataset.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        ann_ids = self.original_dataset.coco.getAnnIds(imgIds=img_id)
        anns = self.original_dataset.coco.loadAnns(ann_ids)
        if len(anns) == 0:
            raise ValueError("Image has no annotation")
        cat_id = anns[0]['category_id']
        label = self.original_dataset.cat_id_to_idx[cat_id]
        # 先套用原 Dataset 的 transform (resize)
        if self.original_dataset.transform:
            image = self.original_dataset.transform(image)
        # 再套用額外的重抽樣 transform
        if self.extra_transform:
            image = self.extra_transform(image)
        return image, torch.tensor(label, dtype=torch.long)
