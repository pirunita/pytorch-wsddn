#coding=utf-8
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class WSDDNDataset(data.Dataset):
    def __init__(self, args):
        # base setting
        self.args = args
        self.mode = args.datamode
        
        self.root = args.dataroot
        self.data_list = args.data_list
        self.ssw_list = args.ssw_list
        
        self.transform = transforms.Compose([  \
            transforms.Resize([480, 480]), \
            transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), \
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])
        
        self.imgs = []
        
        with open(os.path.join(self.root, self.data_list), 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                words = line.split()
                
                if self.mode == 'train':
                    if not(words[0][0:4] == '2007' or words[0][0:4] == '2008'):
                        label_cur = [0 for i in range(20)]
                        for i in range(1, len(words)):
                            label_cur[int(words[i])] = 1