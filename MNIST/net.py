import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class CNNforMNIST(nn.Module):
    def __init__(self, channels=[32, 64]):
        super(CNNforMNIST, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=channels[0],
                      kernel_size=5,
                      padding=2,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels[0],
                      out_channels=channels[0],
                      kernel_size=5,
                      padding=2,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25, inplace=True)
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=channels[0],
                      out_channels=channels[1],
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels[1],
                      out_channels=channels[1],
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=0.25, inplace=True)
        )
        self.linear2 = nn.Linear(in_features=7*7*64, out_features=256)
        self.linear3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(F.relu(self.linear2(x), inplace=True),
                      p=0.5, inplace=True, training=self.training)
        logits = self.linear3(x)

        return logits

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
