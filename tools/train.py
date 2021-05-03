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

def train(args):
    train_data = Mnist_Data("./dataset/train.csv", "train")
    val_data = Mnist_Data("./dataset/train.csv", "val")
    train_loader = DataLoader(train_data, batch_size=100, shuffle= True )

    model = CNNModel()
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    epochs = 30 #'''(int) iterations / len(train_data) / batch_size'''
    best_val_acc = None
    best_val_loss = None
    with torch.no_grad():
       best_val_acc, best_val_loss = evaluate(model, val_data, criterion)
    print('Best Validation Accuracy : {}'.format(best_val_acc))
    print('Best Validation Loss : {}'.format(best_val_loss))
    

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = Variable(images.view(100,1,28,28)).to(device)
            
            labels = Variable(labels).to(device)
            # print(images.shape, labels.shape)
            # print(labels.size())
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('Epoch {}/{} : Step {}/{}, Loss: {:.4f}'
                        .format(epoch + 1, 30, i + 1, len(train_loader), loss.item()))
        with torch.no_grad():
            validation_acc, val_loss = evaluate(model, val_data, criterion)
        model.train()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = validation_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "saved_models/best_epoch_digit_recognizer_mnist_data_pytorch1.6_lr_1e-2_{}.pth".format(epoch + 1))
        # torch.save(model.state_dict(), "weights_epoch_{}.pth".format(epoch + 1))
        print('Best Validation Loss : {}'.format(best_val_loss))
        print('Best Validation Accuracy : {}'.format(best_val_acc))
        print('Best Epoch: {}'.format(best_epoch + 1))
        print('Epoch {}/{} Done | Train Loss : {:.4f} | Validation Loss : {:.4f} | Validation Accuracy : {:.4f}'
              .format(epoch + 1, 30, train_loss / len(train_loader), val_loss, validation_acc))
    return best_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_step', type=int, default=2000)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1E-4)
    parser.add_argument('--freeze_layers', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1111)
    
    args = parser.parse_args()
    print(args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    
    best_val_loss = train(args=args)
    print(best_val_loss)

    