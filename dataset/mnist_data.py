import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split
from utils import transforms
import random

class Mnist_Data(Dataset):
    def __init__(self, data_path, split="train"):
        super(Mnist_Data, self).__init__()
        self.data_path = data_path
        self.split = split
        self.data, self.labels = self.get_data()

        if self.split == "train":
            self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(self.data, self.labels, test_size=0.3, random_state = 123)
            print("Total {} images in {} data".format(len(self.train_X), self.split))
        elif self.split == "val":
            self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(self.data, self.labels, test_size=0.3, random_state = 123)
            print("Total {} images in {} data".format(len(self.val_X), self.split))
        else:
            self.test_X, self.test_Y = self.data, self.labels
            print( "Total {} images in {} data".format(len(self.test_X), self.split))

    def get_data(self):
        data_path = self.data_path
        df = pd.read_csv(data_path)
        # print(df.info())
        if self.split == "train" or self.split == "val":
            labels = df["label"].to_numpy()
            df.drop("label", axis=1, inplace = True)
        else:
            data = df.to_numpy()
            labels = np.zeros((len(data)))
        data = df.to_numpy()
        return data, labels

    def __len__(self):
        if self.split == "train":
            return (len(self.train_X))
        elif self.split == "val":
            return (len(self.val_X))
        else:
            return (len(self.test_X))


    def __getitem__(self, index):
        # print("Index = ", index)
        if self.split == "train":
            img, label = self.train_X[index], self.train_Y[index]
            img = img.reshape(28,28)
            img = np.uint8(img)
            # print(random.random())
            # if random.random() < 0.5 or True:
            #     img = transforms.flip(img).copy()
            if random.random() < 0.5:
                img = transforms.random_blur(img)
                img = transforms.rotation(img, [-10, 10])
            if random.random() < 0.5 :
                random_contrast = random.randint(50, 150) * 1.0
                random_brightness = random.randint(-50, 50)
                img = cv2.convertScaleAbs(img, alpha=random_contrast / 100.0, beta=random_brightness)
            
        elif self.split == "val":
            img, label = self.val_X[index], self.val_Y[index]
            img = img.reshape(28,28)
            img = np.uint8(img)
        else:
            img, label = self.test_X[index], self.test_Y[index]
            img = np.uint8(img)
        # print(img)
        img_tensor = torch.from_numpy(img).float().div(255)
        label_tensor =  label
        return img_tensor, label_tensor


if __name__ == "__main__" :
    dataset = Mnist_Data("./dataset/train.csv", "train")
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1)
    for i, (img, label) in enumerate(dataloader):
        print(i, label)
        if i == 15:
            print(img.shape)
            plt.imshow(img[0])
            plt.plot()
            plt.show()
            exit(0)