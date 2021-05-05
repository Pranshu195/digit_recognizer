import torch
import argparse
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from dataset.mnist_data import Mnist_Data
from model.mnist_model import CNNModel
from tools.evaluate import evaluate
from tqdm import tqdm


# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file = open("submission.csv", 'w')
file.write("ImageId,Label\n")

test_data = Mnist_Data("./dataset/test.csv", "test")
test_loader = DataLoader(test_data, batch_size=1, shuffle= False )

model = CNNModel()
model.load_state_dict(torch.load("saved_models/best_epoch_digit_recognizer_mnist_data_pytorch1.6_lr_1e-2_transform_24.pth"))
model.to(device)

criterion = nn.CrossEntropyLoss()
model.eval()
for i, (images, label) in enumerate(tqdm(test_loader)):
    images = Variable(images.view(1,1,28,28)).to(device)
    labels = Variable(label).to(device)
    # print(images.shape, labels.shape)
    output = model(images)
    
    ## -----------Finding accuracy
    # print(output.shape)
    out = torch.max(output)
    out_index = torch.argmax(output).cpu()
    print(out_index.numpy())
    # print(out)
    # exit(0)
    file.write(str(i+1) + "," + str(out_index.numpy()) + "\n")

