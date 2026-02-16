
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms


import numpy as np
from io import BytesIO
from random import random, choice
import random as rd



def compute_complexity(patch):
    weight, height = patch.size
    m = weight
    res = 0
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()


def select_patch(img, patch_size, height):
    img_width, img_height = img.size
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    for i in range(num_patch):
        patch_list.append(rp(img))
    patch_list.sort(key=lambda x: compute_complexity(x), reverse=False)
    new_img = patch_list[0]

    return new_img

def select_patch_val(img, patch_size, height):
    img_width, img_height = img.size
    # num_patch = (height // patch_size) * (height // patch_size)
    num_patch = 256
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    for i in range(num_patch):
        patch_list.append(rp(img))
    patch_list.sort(key=lambda x: compute_complexity(x), reverse=False)
    new_img = patch_list[0]

    return new_img



trans_patch = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

trans_dwt = transforms.Compose([
        transforms.RandomCrop([256, 256], pad_if_needed=True),
        transforms.ToTensor(),
    ])

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list):
        self.images_path = images_path
        self.images_class = images_class


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        #img已经打开了 已经是一个PLT类型了
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert('RGB')
            #raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]

        patch = select_patch(img=img, patch_size=32, height=256)

        # imgSAFM = imgSAFM
        img_dwt = trans_dwt(img)

        patch = trans_patch(patch)


        return patch, img_dwt, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        patch, img_dwt, label = tuple(zip(*batch))

        patch = torch.stack(patch, dim=0)
        img_dwt = torch.stack(img_dwt, dim=0)

        label = torch.as_tensor(label)
        return patch, img_dwt, label


# 验证/测试时的 transforms（无额外扰动参数输入的情况下）
transform_val_dwt = transforms.Compose([
    transforms.CenterCrop([256, 256]),
    transforms.ToTensor(),
])


class MyValDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list):
        self.images_path = images_path
        self.images_class = images_class
        self.transform_val_dwt = transform_val_dwt

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        #img已经打开了 已经是一个PLT类型了
        # 允许加载截断的图像（可选）
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert('RGB')
            #raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        label = self.images_class[item]

        # 原始数据
        img_dwt = self.transform_val_dwt(img)

        patch = select_patch_val(img=img, patch_size=32, height=256)

        # imgSAFM = imgSAFM
        img_dwt = img_dwt

        patch = trans_patch(patch)


        return patch, img_dwt, label
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        patch, img_dwt, label = tuple(zip(*batch))

        patch = torch.stack(patch, dim=0)
        img_dwt = torch.stack(img_dwt, dim=0)

        label = torch.as_tensor(label)
        return patch, img_dwt, label
