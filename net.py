import math

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 1

class WSDDN(nn.Module):
    def __init__(self):
        super(WSDDN, self).__init__()
        
        # VGG11
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            
        )
        """
        # Alexnet
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
        """
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8c = nn.Linear(4096, 20)
        self.fc8d = nn.Linear(4096, 20)
        
    def forward(self, x, ssw_output):
        x = self.features(x)
        x = self.spp_layer(x, ssw_output)
        
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x_clf = F.relu(self.fc8c(x))
        x_dct = F.relu(self.fc8d(x))
        sigma_clf = F.softmax(x_clf, dim=2)
        sigma_dct = F.softmax(x_dct, dim=1)
        
        x = sigma_clf * sigma_dct
        x = torch.sum(x, dim=1)
        
        return x, sigma_clf, sigma_dct
        
    def spp_layer(self, x, ssw):
        #x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        """
        x.shape: [BATCH_SIZE, 512, 14, 14]
        ssw_output = [BATCH_SIZE, r, 4]
        y.shape = [BATCH_SIZE, r, 4096]
        """
        for i in range(BATCH_SIZE):
            for j in range(ssw.size(1)):
                feature_map_piece = torch.unsqueeze(x[i, :, math.floor(ssw[i, j, 0]) : math.floor(ssw[i, j, 0] + ssw[i, j, 2]),
                                               math.floor(ssw[i, j, 1]) : math.floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                feature_map_piece = spatial_pyramid_pool(previous_conv=feature_map_piece,
                                                         num_sample=1,
                                                         previous_conv_size = [feature_map_piece.size(2), feature_map_piece.size(3)],
                                                         out_pool_size=[2, 2])

                if j == 0:
                    y_piece = feature_map_piece
                else:
                    y_piece = torch.cat((y_piece, feature_map_piece))
            
            if i == 0:
                y = torch.unsqueeze(y_piece, 0)
            else:
                y = torch.cat((y, torch.unsqueeze(y_piece, 0)))
        
        return y
        
        
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
        h_pad = min(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2), math.floor(h_wid/2))
        w_pad = min(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2), math.floor(w_wid/2))
        
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        
        x = maxpool(previous_conv)
        
        if(i == 0):
            spp = x.view(num_sample, -1)
        else:
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        
        return spp