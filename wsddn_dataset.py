#coding=utf-8
import json
import logging
import math
import os

import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

# Set logger
logger = logging.getLogger('DataLoader')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

class WSDDNDataset(data.Dataset):
    def __init__(self, args):
        # base setting
        self.args = args
        self.mode = args.datamode
        
        self.root = args.dataroot
        #self.text_path = args.text_path
        self.jpeg_path = args.jpeg_path
        self.image_label_path = args.image_label_path
        self.ssw_path = args.ssw_path
        self.text_path = args.text_path
        
        
        self.transform = transforms.Compose([  \
            transforms.Resize([480, 480]), \
            transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), \
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])
        
        self.imgs = []
        self.ssw_list = sio.loadmat(os.path.join(self.root, self.mode, self.ssw_path))['boxes'][0] 
        
            
        # Json, trainval 5011
        with open(os.path.join(self.root, self.mode, self.image_label_path), 'r') as jr:
            self.image_label_list = json.load(jr)
        
        # logging
        logger.info('Selective Search Work List' + str(self.ssw_list.shape))
        #logger.info('Image label List' + str(self.image_label_list.shape))
        
        with open(os.path.join(self.root, self.mode, self.text_path), 'r') as text_f:
            for idx, file_name in enumerate(text_f.readlines()):
                file_name = file_name.rstrip()
                
                # image_label parsing
                image_label_current = [0 for i in range(20)]
                image_label_list = self.image_label_list[file_name]
                for i in range(0, len(image_label_list)):
                    image_label_current[i] = 1
                ssw_info = self.ssw_list[idx]
                ssw_block = torch.Tensor(math.floor((ssw_info.shape[0] - 1) / 4), 4)
                
                for i in range(math.floor((ssw_info.shape[0] - 1) / 4)):
                    print(ssw_info[i*4 + 3])
                    w = max(int(ssw_info[i*4 + 3]), 2)
                    h = max(int(ssw_info[i*4 + 4]), 2)
                    ssw_block[i, 0] = (30 - w if (int(ssw_info[i*4 + 1]) + w >= 31) else int(ssw_info[i*4 + 1]))
                    ssw_block[i, 1] = (30 - h if (int(ssw_info[i*4 + 2]) + h >= 31) else int(ssw_info[i*4 + 2]))
                    ssw_block[i, 2] = w
                    ssw_block[i, 3] = h
                    
                self.imgs.append([file_name, ssw_block, label_current])
            

    def __getitem__(self, index):
        current_img = Image.open(os.path.join(self.root, self.mode, self.jpeg_path, self.imgs[index][0] + '.jpg'))
        file_name = self.imgs[index][0]
        data_once = self.transform(current_img)
        ssw_block = self.imgs[index][1]
        label_once = self.imgs[index][2]
        
        return file_name, data_once, ssw_block, torch.Tensor(label_once)

    def __len__(self):
        return len(self.imgs)
        
