"""Pascal VOC Dataset Segmentation Dataloader"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import rasterio

VOC_CLASSES = ('background',  # always index 0
               "Mixed water", "Wakes", "Cloud shadows", "Waves",
               "Shallow water", "Turbid water", "Foam", "Sediment-laden water",
               "Marine water", "Clouds", "Ship", "Natural Organic Material", "Sparse Sargassum",
               "Dense Sargassum", "Marine debris")
"""
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')"""

NUM_CLASSES = len(VOC_CLASSES)



class MarineDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".tif"
        self.mask_extension = ".tif"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }

        return data


    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for root, dirs, files in os.walk(self.mask_root_dir):
            for file in files:
                file_path = os.path.join(root, file)
                ds_cl = rasterio.open(file_path)
                IM_CL = ds_cl.read().reshape(256, 256)
                for i in range(NUM_CLASSES):
                    counts[i] += np.sum(IM_CL == i)
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        src = rasterio.open(path)
        data = src.read()

        rgb_bands = [4, 3, 2]
        rgb_data = data[rgb_bands, :, :]

        rgb_data = (rgb_data / rgb_data.max()) * 255
        rgb_data = rgb_data.astype(np.uint8)
        imx_t = np.array(rgb_data, dtype=np.float32) / 255.0

        return imx_t

    # raw_image = Image.open(path)
    #     raw_image = np.transpose(raw_image.resize((256, 256)), (2,1,0))
    #     imx_t = np.array(raw_image, dtype=np.float32)/255.0
    #
    #     return imx_t

    def load_mask(self, path=None):
        ds = rasterio.open(path)
        temp = ds.read().squeeze()
        return temp
        # raw_image = Image.open(path)
        # raw_image = raw_image.resize((256, 256))
        # imx_t = np.array(raw_image)
        # # border
        # imx_t[imx_t==255] = len(VOC_CLASSES)

        # return imx_t


if __name__ == "__main__":
    data_root = os.path.join("data", "VOCdevkit", "VOC2007")
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationObject")


    objects_dataset = MarineDataset(list_file=list_file_path,
                                       img_dir=img_dir,
                                       mask_dir=mask_dir)

    print(objects_dataset.get_class_probability())

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    image.transpose_(0, 2)

    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()

