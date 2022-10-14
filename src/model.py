import torch.nn as nn
import torch.nn.functional as F
import torch
from torchinfo import summary

class CustomNet(nn.Module):
    def __init__(self, first_conv_out = 10, first_fc_out = 120):
        super(CustomNet, self).__init__() # the input shape is 32x32
        
        self.first_conv_out = first_conv_out
        self.first_fc_out = first_fc_out

        # All Conv layers.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.first_conv_out, kernel_size=5, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(in_channels=self.first_conv_out, out_channels=self.first_conv_out*2, kernel_size=5, stride = 1, padding = 0)

        # All fully connected layers.
        self.fc1 = nn.Linear(in_features=self.first_conv_out*2*5*5, out_features=self.first_fc_out)
        self.fc2 = nn.Linear(in_features=self.first_fc_out, out_features=self.first_fc_out//2)
        self.fc3 = nn.Linear(in_features=self.first_fc_out//2, out_features=10)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)

    def forward(self, x):    
        # Passing though convolutions.
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    net = CustomNet(first_conv_out = 10, first_fc_out = 120)
    print(net)
    #net.forward()    
    #batch_size = 16
    summary(net, (16, 3, 32, 32))