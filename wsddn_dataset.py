#coding=utf-8
import json
import math
import os


import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

class WSDDNDataset(data.Dataset):
    def __init__(self, args):
        # base setting
        self.args = args
        self.mode = args.datamode
        
        self.root = args.dataroot
        #self.text_path = args.text_path
        self.jpeg_path = args.jpeg_path
        self.json_path = args.json_path
        self.ssw_path = args.ssw_path
        
        
        self.transform = transforms.Compose([  \
            transforms.Resize([480, 480]), \
            transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), \
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])
        
        self.imgs = []
        
        with open(os.path.join(self.root, self.mode, self.json_path), 'r') as jr:
            self.json_list = json.load(jr)
        
        
        
        with open(os.path.join(self.root, self.mode, self.ssw_path), 'r') as f:
            for ssw_line in f.readlines():
                ssw_info = ssw_line.rstrip()
                ssw_list = ssw_info.split()
                file_name = os.path.splitext(ssw_list[0])[0]
    
                #if self.mode == 'train':
                # JSON parsing
                label_current = [0 for i in range(20)]
                label_list = self.json_list[file_name]
                for i in range(0, len(label_list)):
                    label_current[i] = 1
                
                # Selective Search Data parsing
                ssw_block = torch.Tensor(math.floor((len(ssw_list) - 1) / 4), 4)
                
                for i in range(math.floor((len(ssw_list) - 1) / 4)):
                    w = max(int(ssw_list[i*4 + 3]), 2)
                    h = max(int(ssw_list[i*4 + 4]), 2)
                    ssw_block[i, 0] = (30 - w if (int(ssw_list[i*4 + 1]) + w >= 31) else int(ssw_list[i*4 + 1]))
                    ssw_block[i, 1] = (30 - h if (int(ssw_list[i*4 + 2]) + h >= 31) else int(ssw_list[i*4 + 2]))
                    ssw_block[i, 2] = w
                    ssw_block[i, 3] = h
                
                self.imgs.append([file_name, ssw_block, label_current])
                
                """
                elif self.mode == 'test':
                    label_current = [0 for i in range(20)]
                    for i in range(1, len(words)):
                        label_current[int(words[i])] = 1
                    for ssw in self.ssw_list:
                        ssw = ssw.rstrip()
                        words_ssw = ssw.split()
                        if word_ssw[0] == words[0]:
                            ssw_block = torch.Tensor(math.floor((len(word_ssw) - 1) / 4), 4)
                            for i in range(math.floor((len(word_ssw) - 1) / 4)):
                                w = max(int(word_ssw[i*4 + 3]), 2)
                                h = max(int(word_ssw[i*4 + 4]), 2)
                                ssw_block[i, 0] = (30 - w if (int(word_ssw[i*4 + 1]) + w >= 31) else int(word_ssw[i*4 + 1]))
                                ssw_block[i, 1] = (30 - h if (int(word_ssw[i*4 + 2]) + h >= 31) else int(word_ssw[i*4 + 2]))
                                ssw_block[i, 2] = w
                                ssw_block[i, 3] = h
                        else:
                            ssw_block = torch.tensor([0, 0, 2, 2])
                    
                    self.imgs.append([words[0], ssw_block, label_current])
                """
    def __getitem__(self, index):
        current_img = Image.open(os.path.join(self.root, self.mode, self.jpeg_path, self.imgs[index][0] + '.jpg'))
        file_name = self.imgs[index][0]
        data_once = self.transform(current_img)
        ssw_block = self.imgs[index][1]
        label_once = self.imgs[index][2]
        
        return file_name, data_once, ssw_block, torch.Tensor(label_once)

    def __len__(self):
        return len(self.imgs)
        