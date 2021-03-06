import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import BoxReshape

BATCH_SIZE = 1

class WSDDN(nn.Module):
    def __init__(self, args):
        super(WSDDN, self).__init__()
        
        self.args = args
        self.block_transform = BoxReshape()
        # VGG11
        if self.args.backbone_network == 'vgg11':
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
        
        elif self.args.backbone_network == 'alexnet':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
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
        
        self.classifier = nn.Sequential(
            nn.LeakyReLU(1, inplace=True),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        
        self.score_cls = nn.Linear(in_features=4096, out_features=20)
        self.score_det = nn.Linear(in_features=4096, out_features=20)
        self.cls_softmax = nn.Softmax(dim=2)
        self.det_softmax = nn.Softmax(dim=1)
        
    def forward(self, x, ssw_block, x_width, x_height):
        if self.args.backbone_network == 'vgg11':
            x = self.features(x)
            conv_width = x.shape[2]
            conv_height = x.shape[3]
            ssw_output = self.block_transform(ssw_block, x_width, x_height, conv_width, conv_height)
            #print('x', x.shape)
            #print('ssw_output', ssw_output.shape)
            x = self.spp_layer(x, ssw_output)
            #print('x', x.shape)
            x = F.relu(self.fc6(x))
            x = F.relu(self.fc7(x))
            x_clf = F.relu(self.fc8c(x))
            x_det = F.relu(self.fc8d(x))
            sigma_clf = F.softmax(x_clf, dim=2)
            sigma_det = F.softmax(x_det, dim=1)
            
            x = sigma_clf * sigma_det
            x = torch.sum(x, dim=1)
            
            return x, sigma_clf, sigma_det

        elif self.args.backbone_network == 'alexnet':
            x = self.features(x)
            x = self.spp_layer(x, ssw_output)
            x = self.classifier(x)
            
            score_cls = self.score_cls(x)
            score_det = self.score_det(x)
            sigma_clf = self.cls_softmax(score_cls)
            sigma_det = self.det_softmax(score_det)
            
            x = sigma_clf * sigma_det
            x = torch.sum(x, dim=1)
            
            return x, sigma_clf, sigma_det
        
    def spp_layer(self, x, ssw):
        #x.shape = [BATCH_SIZE, 512, 14, 14] ssw_get.shape = [BATCH_SIZE, R, 4] y.shape = [BATCH_SIZE, R, 4096]
        """
        vgg11
        x.shape: [BATCH_SIZE, 512, 14, 14]
        ssw_output = [BATCH_SIZE, r, 4]
        y.shape = [BATCH_SIZE, r, 4096]
        
        alexnet
        x.shape = [BATCH, 256, 29, 29]
        ssw_output = [BATCH, r, 4]
        """
        #print(ssw.shape)
        
        for i in range(BATCH_SIZE):
            for j in range(ssw.size(1)):
                #print('size', ssw.size(1))
                #print('xywh2', math.floor(ssw[i, j, 0]), math.floor(ssw[i, j, 1]), math.floor(ssw[i,j,2]), math.floor(ssw[i, j, 3]))
                #print('xywh', ssw[i, j, 0], ssw[i, j, 1], ssw[i, j, 0] + ssw[i, j, 2], ssw[i, j, 1] + ssw[i, j, 3])
    
                feature_map_piece = torch.unsqueeze(x[i, :, math.floor(ssw[i, j, 0]) : math.floor(ssw[i, j, 0] + ssw[i, j, 2]), \
                                                    math.floor(ssw[i, j, 1]) : math.floor(ssw[i, j, 1] + ssw[i, j, 3])], 0)
                #print('unsq', feature_map_piece.shape)
                
                """
                if feature_map_piece.size(2) == 0:
                    print('feature_map')
                    print(x.shape)
                    print(feature_map_piece.shape)
                    print(ssw[i, j, :])
                """
                feature_map_piece = spatial_pyramid_pool(previous_conv=feature_map_piece, \
                                                         num_sample=1, \
                                                         previous_conv_size = [feature_map_piece.size(2), feature_map_piece.size(3)], \
                                                         out_pool_size=[2, 2])
                #print('spa', feature_map_piece.shape)
                #print(j)
                #print('xx', x.shape)
                if j == 0:
                    y_piece = feature_map_piece
                    #print('y_piece', y_piece.shape)
                else:
                    #print('y',y_piece.shape)
                    #print('f',feature_map_piece.shape)
                    y_piece = torch.cat((y_piece, feature_map_piece))
                    #print('y_piece2', y_piece.shape)
            
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
    #print(previous_conv_size)
    spp = torch.tensor([]).cuda()
    for i in range(len(out_pool_size)):
        
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        
        if h_wid >= out_pool_size[i] and w_wid >= out_pool_size[i]:
            #h_pad = math.floor((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            #w_pad = math.floor((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            h_pad = min(math.floor((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2), math.floor(h_wid/2))
            w_pad = min(math.floor((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2), math.floor(w_wid/2))
        
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            #print('info', h_wid, w_wid, h_pad, w_pad)
            #print('x', x.shape)
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
            #print('sppiter', spp.size())
            """
            try:
                x = maxpool(previous_conv)
                print('previous', previous_conv_size)
                print('x', x.shape)
                print('reshape', torch.cat((spp, x.view(num_sample, -1)), 1))
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
                print('spp', spp.size())
            except RuntimeError as e:
                print(e)
                #print(previous_conv_size)
                
                break
            """
        
        else:
            break
    #print('spp', spp.size())
        
    return spp
