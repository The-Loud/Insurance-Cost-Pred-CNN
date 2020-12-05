import torch
import torch.nn as nn
'''

'''


class SmartNet(nn.Module):
    '''
    SmartNet class. Uses a 3-block CNN with leaky ReLU and dropout layers.
    Average pooling is used.
    Args:
        n_features: the number of total features in the dataset.
        k: the number of time series to use. Each row is one time slice
    '''

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

        self.LReLU = nn.LeakyReLU()
        self.Dropout = nn.Dropout()
        self.pool = nn.AvgPool2d(kernel_size=2)


        # add dense

    def forward(self, x):
        '''
        do stuff here.
        '''
        pass
