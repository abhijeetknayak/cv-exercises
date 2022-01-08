import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        # START TODO #################
        # see model description in exercise pdf
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding='same')

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding='same')

        self.linear1 = nn.Linear(8 * 8 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

        # END TODO #################

    def forward(self, x):
        # START TODO #################
        # see model description in exercise pdf
        x = self.conv1(x)
        nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        nn.functional.relu(x, inplace=True)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.shape[0], -1)

        x = self.linear1(x)
        nn.functional.relu(x, inplace=True)

        x = self.linear2(x)
        nn.functional.relu(x, inplace=True)

        x = self.linear3(x)

        return x

        # END TODO #################
