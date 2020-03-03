#coding=utf-8
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
        self.mode = args.mode
        
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
                        label_current = [0 for i in range(20)]
                        for i in range(1, len(words)):
                            label_current[int(words[i])] = 1
                        for ssw in self.ssw_list:
                            ssw = ssw.rstrip()
                            word_ssw = ssw.split()
                            if word_ssw[0] == words[0]:
                                ssw_block = torch.Tensor(math.floor((len(word_ssw) - 1) / 4), 4)
                                for i in range(math.floor((len(word_ssw) - 1) / 4)):
                                    w = max(int(word_ssw[i*4 + 3]), 2)
                                    h = max(int(word_ssw[i*4 + 4]), 2)
                                    ssw_block[i, 0] = (30 - w if (int(word_ssw[i*4 + 1]) + w >= 31) else int(word_ssw[i*4 + 1]))
                                    ssw_block[i, 1] = (30 - h if (int(word_ssw[i*4 + 2]) + h >= 31) else int(word_ssw[i*4 + 2]))
                                    ssw_block[i, 2] = w
                                    ssw_block[i, 3] = h
                                break
                            else:
                                ssw_block = torch.tensor([0, 0, 2, 2])
                        
                        self.imgs.append([words[0], ssw_block, label_current])
                elif self.mode == 'test':
                    if words[0][0:4] == '2007' or words[0][0:4] == '2008':
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
                                break
                            else:
                                ssw_block = torch.tensor([0, 0, 2, 2])
                        
                        self.imgs.append([words[0], ssw_block, label_current])
                        
    def __getitem__(self, index):
        current_img = Image.open(self.root + self.imgs[index][0] + '.jpg')
        data_once = self.transform(current_img)
        ssw_block = self.imgs[index][1]
        label_once = self.imgs[index][2]
        
        return data_once, ssw_block, torch.Tensor(label_once)

    def __len__(self):
        return len(self.imgs)
        