import os
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        if (spec := find_spec("torch")) and spec.origin and os.path.exists(
            dll_path := os.path.join(os.path.dirname(spec.origin), "lib", "c10.dll")
        ):
            ctypes.CDLL(os.path.normpath(dll_path))
    except Exception:
        pass

from PyQt6.QtWidgets import QApplication
import torch
import torch.nn as nn
import pandas as pd
from thop import profile
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from SEResNet18 import ThinSEResNet18,SEBasicBlock,SEBlock
from ThinResNet18 import ThinResNet18
from ThinCBAMResNet18 import ThinCBAMResNet18
# 假設 SimpleCNN 已定義好

def analyze_model(model, input_size=(1, 3, 224, 224), device="cpu"):
    model = model.to(device)
    model.eval()

    dummy = torch.randn(*input_size).to(device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=False)

    flops = macs * 2  # MACs → FLOPs
    return params, flops

model_configs = {
    "ThinResNet18_ss": lambda: ThinResNet18(num_classes=11, channels=[2,4,8,16]),
    "ThinResNet18_s":  lambda: ThinResNet18(num_classes=11, channels=[4,8,16,32]),
    "ThinResNet18_m":  lambda: ThinResNet18(num_classes=11, channels=[8,16,32,64]),
    "ThinResNet18_l":  lambda: ThinResNet18(num_classes=11, channels=[16,32,64,128]),
    "ThinResNet18_h":  lambda: ThinResNet18(num_classes=11, channels=[32,64,128,256]),
    "ThinSEResNet18_ss": lambda: ThinSEResNet18(num_classes=11, channels=[2,4,8,16],se_reduction=2),
    "ThinSEResNet18_s": lambda: ThinSEResNet18(num_classes=11, channels=[4,8,16,32],se_reduction=2),
    "ThinSEResNet18_m": lambda: ThinSEResNet18(num_classes=11, channels=[8,16,32,64],se_reduction=2),
    "ThinSEResNet18_l": lambda: ThinSEResNet18(num_classes=11, channels=[16,32,64,128],se_reduction=2),
    "ThinSEResNet18_h": lambda: ThinSEResNet18(num_classes=11, channels=[32,64,128,256],se_reduction=2),
    "ThinCBAMResNet18_ss": lambda: ThinCBAMResNet18(num_classes=11, channels=[2,4,8,16],cbam_reduction=2),
    "ThinCBAMResNet18_s": lambda: ThinCBAMResNet18(num_classes=11, channels=[4,8,16,32],cbam_reduction=2),
    "ThinCBAMResNet18_m": lambda: ThinCBAMResNet18(num_classes=11, channels=[8,16,32,64],cbam_reduction=2),
    "ThinCBAMResNet18_l": lambda: ThinCBAMResNet18(num_classes=11, channels=[16,32,64,128],cbam_reduction=2),
    "ThinCBAMResNet18_h": lambda: ThinCBAMResNet18(num_classes=11, channels=[32,64,128,256],cbam_reduction=2),
    "ResNet18": None
}

def params_flops(model_configs, save_path):
    records = []
    for name, model_fn in model_configs.items():
        if name == "ResNet18":
            model = resnet18(weights=None, num_classes=11)
        else:
            model = model_fn()
        params, flops = analyze_model(model)
        records.append({
            "Model": name,
            "Params (M)": params / 1e6,
            "FLOPs (G)": flops / 1e9 / 2,
        })  
    df = pd.DataFrame(records)
    if save_path:
        df.to_csv(save_path, index=False)
    return df

save_path = r"c:\Users\user\Downloads\Tree disease MLC model.v7i.coco\results\model_flops_params.csv"
df = params_flops(model_configs, save_path=save_path)
