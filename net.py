import math

import torch
import torch.nn as nn

class WSDDN(nn.Module):
    def __init__(self):
        super(WSDDN, self).__init__()
        
        # pool5 for extracting features
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(192, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(384, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)),
        )
        
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        
    def forward(self, x, ssw_output):
        x = self.features(x)
        x = self.spp_layer(x, ssw_output)
    
    def spp_layer(self, x, ssw):
        #x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        """
        x.shape: [BATCH_SIZE, 512, 14, 14]
        ssw_output = [BATCH_SIZE, r, 4]
        y.shape = [BATCH_SIZE, r, 4096]
        """
        for i in range(BATCH_SIZE):
            for j in range(ssw.size(1)):
        
        

def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    
    for i in range(len(out_pool_size)):
        h_wid = math.ceil(previous_conv_size[0] / out_pool_size[i])
        w_wid = math.ceil(previous_conv_size[1] / out_pool_size[i])
        h_pad = min(math.floor((h_wid*out_pool_size[i])))
        