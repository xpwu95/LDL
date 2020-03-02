import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.img_path = data_path
        self.transform = transform
        # reading img file from file
        img_filepath = img_filename
        fp = open(img_filepath, 'r')

        self.img_filename = []
        self.labels = []
        self.lesions = []
        for line in fp.readlines():
            filename, label, lesion = line.split()
            self.img_filename.append(filename)
            self.labels.append(int(label))
            self.lesions.append(int(lesion))
        # self.img_filename = [x.strip() for x in fp]
        fp.close()
        self.img_filename = np.array(self.img_filename)
        self.labels = np.array(self.labels)#.reshape(-1, 1)
        self.lesions = np.array(self.lesions)#.reshape(-1, 1)

        if 'NNEW_trainval' in img_filename:
            ratio = 1.0#0.1
            import random
            random.seed(42)
            indexes = []
            for i in range(4):
                index = random.sample(list(np.where(self.labels == i)[0]), int(len(np.where(self.labels == i)[0]) * ratio))
                indexes.extend(index)
            self.img_filename = self.img_filename[indexes]
            self.labels = self.labels[indexes]
            self.lesions = self.lesions[indexes]

        # reading labels from file
        # label_filepath = os.path.join(data_path, label_filename)
        # labels = np.loadtxt(label_filepath, dtype=np.int64)

        # self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        name = self.img_filename[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        lesion = torch.from_numpy(np.array(self.lesions[index]))
        return img, label, lesion#, name
    def __len__(self):
        return len(self.img_filename)

