import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

# use gpu if cuda can be detected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate(model, val_dataset, criterion):
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    model.to(device)
    model.eval()
    correct = 0.0
    total = 0.0
    val_loss = 0.0
    for i, (images, label) in enumerate(tqdm(val_loader)):
        images = Variable(images.view(1,1,28,28)).to(device)
        labels = Variable(label).to(device)
        # print(images.shape, labels.shape)
        output = model(images)
        
        # print(label.size(), output.size())
        loss = criterion(output, labels)
        val_loss += loss.item()

        ## -----------Finding accuracy
        # print(output.shape)
        out = torch.max(output)
        outi = torch.argmax(output)
        # print(out)
        if(out > 0.5):
            pred = 1
            if(labels==outi):
                correct += 1
        else:
            pred = 0
        
        total += 1

    return correct/total, val_loss/(len(val_loader))