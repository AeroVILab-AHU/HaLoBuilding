from .transform import *
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image, ImageOps
import random

CLASSES = ('background', 'building')
PALETTE = [[0, 0, 0], [255, 255, 255]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.25),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([
        RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=255, nopad=False)
    ])
    img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [albu.Normalize()]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class HaLo_HTrainDataset(Dataset):
    def __init__(self, data_root='//data/sff/LWGANet-main/segmentation/data/HaLo_H/train_val', img_dir='img', mosaic_ratio=0.25,
                 mask_dir='m', img_suffix='.png', mask_suffix='.png',
                 transform=train_aug, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mosaic_ratio = mosaic_ratio

        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_ids = self.get_img_ids()

    def get_img_ids(self):
        img_path = osp.join(self.data_root, self.img_dir)
        return [id.split('.')[0] for id in os.listdir(img_path)]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        p_ratio = random.random()
        img, mask = self.load_img_and_mask(index)
        if p_ratio < self.mosaic_ratio:
            img, mask = self.load_mosaic_img_and_mask(index)
        if self.transform:
            img, mask = self.transform(img, mask)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        return {'img': img, 'gt_semantic_seg': mask, 'img_id': img_id}

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root,  self.img_dir, img_id + self.img_suffix)
        mask_path = osp.join(self.data_root,  self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        imgs, masks = zip(*(self.load_img_and_mask(i) for i in indexes))

        imgs = [np.array(img) for img in imgs]
        masks = [np.array(mask) for mask in masks]

        h, w = self.img_size
        start_x = w // 4
        start_y = h // 4
        offset_x = random.randint(start_x, w - start_x)
        offset_y = random.randint(start_y, h - start_y)

        crop_sizes = [
            (offset_x, offset_y),
            (w - offset_x, offset_y),
            (offset_x, h - offset_y),
            (w - offset_x, h - offset_y)
        ]

        crops = [albu.RandomCrop(width=cw, height=ch)(image=im, mask=ma)
                 for (im, ma), (cw, ch) in zip(zip(imgs, masks), crop_sizes)]

        img_top = np.concatenate((crops[0]['image'], crops[1]['image']), axis=1)
        img_bot = np.concatenate((crops[2]['image'], crops[3]['image']), axis=1)
        mask_top = np.concatenate((crops[0]['mask'], crops[1]['mask']), axis=1)
        mask_bot = np.concatenate((crops[2]['mask'], crops[3]['mask']), axis=1)

        img = np.concatenate((img_top, img_bot), axis=0)
        mask = np.concatenate((mask_top, mask_bot), axis=0)

        return Image.fromarray(img), Image.fromarray(mask)

HaLo_H_val_dataset = HaLo_HTrainDataset(data_root='/data/sff/LWGANet-main/segmentation/data/HaLo_H/val', mosaic_ratio=0.0,
                                        transform=val_aug)
class HaLo_HTestDataset(Dataset):
    def __init__(self, data_root='/data/sff/LWGANet-main/segmentation/data/HaLo_H/test', img_dir='img',
                 img_suffix='.png', mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_ids = self.get_img_ids()

    def get_img_ids(self):
        img_path = osp.join(self.data_root,  self.img_dir)
        return [id.split('.')[0] for id in os.listdir(img_path)]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img = self.load_img(index)
        img = np.array(img)
        img = albu.Normalize()(image=img)['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        return {'img': img, 'img_id': img_id}

    def load_img(self, index):
        img_id = self.img_ids[index]
        img_path = osp.join(self.data_root,  self.img_dir, img_id + self.img_suffix)
        return Image.open(img_path).convert('RGB')
