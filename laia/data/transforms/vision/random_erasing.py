# erase_transform.py
import torchvision.transforms as T
import torch

# random erasing with implemented torch erasing function
def get_random_erasing(p=0.5):
    return T.RandomErasing(p=p, scale=(0.01, 0.2), ratio=(0.005,0.01), value=0)

# data wrapper
class AugmentedImageTransform(torch.nn.Module):
    def __init__(self, base_transform, apply_erasing=True, erasing_transform=None):
        super().__init__()
        self.base_transform = base_transform
        self.erasing = erasing_transform if apply_erasing else None

    def forward(self, img):
        img = self.base_transform(img)
        if self.erasing:
            img = self.erasing(img)
        return img
