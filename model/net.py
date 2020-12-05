import torch
import torch.nn as nn


class SmartNet(nn.Module):
    '''
    SmartNet class. Uses a 3-block CNN with leaky ReLU and dropout layers.
    Average pooling is used.
    Args:
        n_features: the number of total features in the dataset.
        k: the number of time series to use. Each row is one time slice.
    '''

    # pylint: disable=too-many-instance-attributes
    def __init__(self, n_features: int, k: int = 3):
        super(SmartNet, self).__init__()
        self.n_features = n_features
        self.k = k
        self.conv1 = nn.Conv2d(in_channels=self.n_features,
                               out_channels=128,
                               kernel_size=(self.n_features, self.k))
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(self.n_features/2, self.k))
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(self.n_features/4, self.k))  # check this.

        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(32, 5)
        self.flatten = torch.flatten()
        self.soft = nn.Softmax()

    def forward(self, x):
        '''
        Feed forward pass.
        Args:
            x: the input data.
        '''
        # Block 1
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.lrelu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.pool(x)

        # Output
        x = self.Dropout(x)
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.soft(x)

        return x
