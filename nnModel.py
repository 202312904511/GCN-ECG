from torch import nn
from torch.nn import functional as F
from gcnModel import GCN_layer
from resnet import resnet

class zhnn1(nn.Module):
    def __init__(self, input_shape, A):
        super(zhnn1, self).__init__()
        self.h_n = input_shape[0]
        A = A.cuda()
        self.A = nn.Parameter(A, requires_grad=True)

        self.resnet1 = resnet()

        self.gconv2 = GCN_layer((60, 30), bias=True)
        self.norm2 = nn.BatchNorm2d(64)
        self.ELU2 = nn.ELU(inplace=True)
        self.drop2 = nn.Dropout2d(0.25)

        self.dconv3 = nn.Conv2d(64, 16, kernel_size=(30, 1), stride=(1, 1), groups=16)
        self.pconv3 = nn.Conv2d(16, 32, kernel_size=(1, 1), groups=1)
        self.norm3 = nn.BatchNorm2d(32)
        self.ELU3 = nn.ELU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop3 = nn.Dropout2d(0.25)

        self.flatten = nn.Flatten()

        # Define the fully connected layers with further reduced out_features
        self.fc1 = nn.Sequential(
            nn.Linear(224, 8),  # Further reduced out_features to 8
        )
        self.fc2 = nn.Linear(8, 5)  # Further reduced out_features to 5

    def forward(self, input):
        input = input.view(-1, 1, 1, 300)
        x = F.pad(input, pad=(7, 8, 0, 0))
        x = self.resnet1(x)

        x = self.gconv2(self.A, x)
        x = self.norm2(x)
        x = self.ELU2(x)
        x = self.drop2(x)

        x = self.dconv3(x)
        x = self.pconv3(x)
        x = self.norm3(x)
        x = self.ELU3(x)
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x