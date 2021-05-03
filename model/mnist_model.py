import torch
import torch.nn as nn



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Conv layer #1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool layer #1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Conv layer #2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool layer #2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layer #1
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
    
    def forward(self, x):
        # Conv layer #1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool layer #1
        out = self.maxpool1(out)
        
        # Conv layer #2
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool layer #2
        out = self.maxpool2(out)
        
        # Flattening
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out


if __name__ == "__main__":
    model = CNNModel()
    
