import torch

from Models.Sample import Sample
from Models.SampleCha import SampleCha
from Models.StandardSarCNN import StandardSarCNN
from Models.VGG19 import VGG19
from Models.VGG19BN import VGG19BN
from Models.VGG19BNLite import VGG19BNLite


def get_model(name: str, num_classes: int) -> torch.nn.Module:
    name = name.lower()
    if name == "MatNet".lower():
        return StandardSarCNN(num_classes)
    elif name == "VGG19".lower():
        return VGG19(num_classes)
    elif name == "VGG19BN".lower():
        return VGG19BN(num_classes)
    elif name == "VGG19BNLite".lower():
        return VGG19BNLite(num_classes)
    elif name == "Sample".lower():
        return Sample(num_classes)
    elif name == "SampleCha".lower():
        return SampleCha(num_classes)
    raise ModuleNotFoundError(f"Model {name} not found")
