import torch.nn as nn
import torch.nn.functional as F

class GlassNet(nn.Module):
    def __init__(self, image_size, num_channels, num_filters, num_classes):
        super(GlassNet, self).__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters, num_filters*2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_filters*8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )
        self.clf = nn.Sequential(
            nn.Linear(128*7*7, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, self.num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.feature(x)
        out1 = out1.view(out1.size(0), -1) # flatten output
        out = self.clf(out1)
        return out

