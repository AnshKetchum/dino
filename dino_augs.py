import torch.nn as nn

import random
import torchvision.transforms as T
from PIL import Image, ImageFilter


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class DINOCrops(nn.Module):
    def __init__(self, local_crops_scale, global_crops_scale, num_local_crops=8):
        super(DINOCrops, self).__init__()

        # Define our local crops
        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

       # first global crop
        self.global_transformation_1 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale,
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transformation_2 = T.Compose([
            T.RandomResizedCrop(224, scale=global_crops_scale,
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.5),
            T.RandomSolarize(128),
            normalize,
        ])

        # Transformation for the local small crops
        self.num_local_crops = num_local_crops
        self.local_transformation = T.Compose([
            T.RandomResizedCrop(96, scale=local_crops_scale,
                                interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.5),
            normalize,
        ])

    def forward(self, batch):

        # Collect all of our local transformations
        local_crops = []
        for i in range(self.num_local_crops):
            local_crops.append(self.local_transformation(batch))

        # Collect our global transformations
        global_crops = [self.global_transformation_1(
            batch), self.global_transformation_2(batch)]

        return local_crops, global_crops
