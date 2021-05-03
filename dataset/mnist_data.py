import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
from sklearn.model_selection import train_test_split

class Mnist_Data(Dataset):
    def __init__(self, data_path, isTrain=True):
        super(Mnist_Data, self).__init__()
        self.data_path = data_path
        self.isTrain = isTrain
        self.data, self.labels = self.get_data()
        self.train_X, self.val_X, self.train_Y, self.val_Y = train_test_split(self.data, self.labels, test_size=0.3, random_state = 123)
        if isTrain == True:
            print("Total {} images in {} data".format(len(self.train_X), isTrain))
        else:
            print("Total {} images in {} data".format(len(self.val_X), isTrain))

    def get_data(self):
        data = []
        data_path = self.data_path
        df = pd.read_csv(data_path)
        print(df.info())

        labels = df["label"].to_numpy()
        df.drop("label", axis=1, inplace = True)
        
        data = df.to_numpy()
        return data, labels

    def __len__(self):
        if(self.isTrain == True):
            return (len(self.train_X))
        else:
            return (len(self.val_X))

    def __getitem__(self, index):
        if self.isTrain == True:
            img, label = self.train_X[index], self.train_Y[index]
        else:
            img, label = self.val_X[index], self.val_Y[index]
        img = img.reshape(28,28)
        
        img_tensor = torch.from_numpy(img).float().div(255)
        label_tensor =  label
        return img_tensor, label_tensor


if __name__ == "__main__" :
    dataset = Mnist_Data("./dataset/train.csv", False)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 1)
    for i, (img, label) in enumerate(dataloader):
        print(i, label)
        if i == 16:
           
            print(img.shape)
            plt.imshow(img[0])
            plt.plot()
            plt.show()
            exit(0)