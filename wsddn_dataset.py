#coding=utf-8
import json
import logging
import math
import os

import scipy.io as sio
import torch
import torch.utils.data as data
import torchvision
import tqdm

from PIL import Image, ImageDraw
from utils import Augmentation, Normalize
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
        self.augmentation = Augmentation()
        self.image_transform = Normalize()
        #self.block_transform = BoxReshape()
        
        self.imgs = []
        self.ssw_list = sio.loadmat(os.path.join(self.root, self.mode, self.ssw_path))['boxes'][0] 
        
            
        # Json, trainval 5011
        with open(os.path.join(self.root, self.mode, self.image_label_path), 'r') as jr:
            self.image_label_list = json.load(jr)
        
        # logging
        logger.info('Selective Search Work List' + str(self.ssw_list.shape))
        #logger.info('Image label List' + str(self.image_label_list.shape))
        
        with open(os.path.join(self.root, self.mode, self.text_path), 'r') as text_f:
            for idx, file_name in tqdm.tqdm(enumerate(text_f.readlines())):
                file_name = file_name.rstrip()
                
                # image_label parsing
                image_label_current = [0 for i in range(20)]
                image_label_list = self.image_label_list[file_name]
                for i in range(0, len(image_label_list)):
                    image_label_current[i] = 1
                    
                ssw_info = self.ssw_list[idx]
                
                self.imgs.append([file_name, ssw_info, image_label_current])
    
    """
    def transform(self, image):
        # [480, 576, 688, 864, 1200]
        im_size_min = min(image)
        a = [480, 576, 688, 864, 1200]
        
        trans = transforms.Compose([  \
            transforms.Resize([480, 480]), \
            transforms.RandomHorizontalFlip(), \
            transforms.ToTensor(), \
            transforms.Normalize(mean = [ 0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])
        ])
    """
    def __getitem__(self, index):
        file_name = self.imgs[index][0]
        ssw_info = self.imgs[index][1]
        label_once = self.imgs[index][2]
        
        current_img = Image.open(os.path.join(self.root, self.mode, self.jpeg_path, file_name + '.jpg'))
        
        augment_image, augment_ssw_block = self.augmentation(current_img, ssw_info)
        #print('augment_image', augment_image.size)
        #print('augment_ssw_block', len(augment_ssw_block))
        #self.show_image(augment_image, ssw_block[6], augment_ssw_block[6])
        image_width, image_height = augment_image.size
        data_once = self.image_transform(augment_image)
        block_once = torch.Tensor(augment_ssw_block) 
        #reshaped_ssw_block = self.block_transform(augment_image, augment_ssw_block)
        #print('re', reshaped_ssw_block)
        
    
        
        
        return file_name, data_once, image_width, image_height, block_once, torch.Tensor(label_once)

    def __len__(self):
        return len(self.imgs)
    
    @staticmethod
    def show_image(image, ssw_block, augment_ssw_block):
        shape1 = [ssw_block[1], ssw_block[0], ssw_block[3], ssw_block[2]]
        shape2 = [augment_ssw_block[1], augment_ssw_block[0], augment_ssw_block[3], augment_ssw_block[2]]
        draw_image = ImageDraw.Draw(image)
        draw_image.rectangle(shape1, outline='red')
        draw_image.rectangle(shape2, outline='blue')
        image.show()
        
        
        




