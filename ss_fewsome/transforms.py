import torch.utils.data as data
from PIL import Image
import torch
import random
import os
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn

RECEPTIVE_FIELD = 137

def crop_and_paste_patch(image, patch_w, patch_h, transform, rotation=False):
    org_w, org_h = image.shape
    mask = None

    patch_left, patch_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    patch_right, patch_bottom = patch_left + patch_w, patch_top + patch_h
    patch =image[patch_top:patch_bottom, patch_left:patch_right]
    if transform:
        patch= transform(patch)

    if rotation:
        random_rotate = random.uniform(*rotation)
        patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
        mask = patch.split()[-1]

    paste_left, paste_top = random.randint(0, org_w - patch_w), random.randint(0, org_h - patch_h)
    aug_image = image.copy()
    aug_image[paste_top:(paste_top+patch.shape[0]), paste_left:(paste_left+patch.shape[1])]=patch


    mask = np.zeros(image.shape)
    assert mask.shape == (224,224)
    mask[paste_top:(paste_top+patch.shape[0]), paste_left:(paste_left+patch.shape[1])]=1
    mask=nn.MaxPool2d(RECEPTIVE_FIELD, stride = 32, ceil_mode =True)(torch.FloatTensor(mask.reshape(1,224,224)))


    return aug_image, mask

def transform_function(augmentations, im):

    mask= 1
    i = random.sample(range(0, len(augmentations)), 1)[0]
    augmentations = [augmentations[i] ]


#    print(augmentations)
    for aug in augmentations:


        if aug == 'crop' :
            im = torch.FloatTensor(im) / 255
            im = torch.stack((im,im,im),0)

            return transforms.RandomResizedCrop(size=(224,224), scale=(0.08, 0.25))(im)


        if aug == 'bright':
            return transforms.ColorJitter(brightness=(0.4,1), contrast=0, saturation=0, hue=0)(im)

        if aug == 'jitter':
            return transforms.ColorJitter(brightness=(0.4,1), contrast=(0.4,1), saturation=(0.4,1), hue=(0.5, 0.5))(im)


        if aug == 'sharp':

            return transforms.functional.adjust_sharpness(im, sharpness_factor=4)



        if aug =='cutpaste':
            area_ratio = (0.2, 0.3)
            aspect_ratio = ((0.3, 1) , (1, 3.3))
            img_area = im.shape[0] * im.shape[1]
            patch_area = random.uniform(*area_ratio) * img_area
            patch_aspect = random.choice([random.uniform(*aspect_ratio[0]), random.uniform(*aspect_ratio[1])])
            patch_w  = int(np.sqrt(patch_area*patch_aspect))
            patch_h = int(np.sqrt(patch_area/patch_aspect))
            im, mask = crop_and_paste_patch(im, patch_w, patch_h, transform=False, rotation=False)
            im = torch.FloatTensor(im) / 255
            im = torch.stack((im,im,im),0)

            return im, mask
