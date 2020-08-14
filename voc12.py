import torch
import cv2
import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch.utils.data as data
import random

class Voc12(data.Dataset):
    def __init__(self, root, split, ignore_label, mean_bgr, augment, base_size, crop_size, scales, flip, year=2012):
        super(Voc12, self).__init__()
        self.year = year
        self.flip = flip
        self.scales = scales
        self.crop_size = crop_size
        self.base_size = base_size
        self.augment = augment
        self.mean_bgr = np.array(mean_bgr)         # Used to pad padding for image
        self.ignore_label = ignore_label # Used to pad padding for label, ignore_label is to ignore the label without calculating loss
        self.split = split
        self.root = os.path.join(root, "VOC%d" %self.year)

        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.label_dir = os.path.join(self.root, "SegmentationClass")

        if self.split in ["train", "trainval", "val", "test"]:
            file_list = os.path.join(self.root, "ImageSets/Segmentation/%s.txt" %self.split)
            print('--file_list---', file_list)
            file_list = tuple(open(file_list, "r"))
            file_list = [i.rstrip() for i in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: %s" %self.split)
        
