from numpy import random
from PIL import Image, ImageOps

import math
import torch
import torchvision
            

class Augmentation(object):
    def __init__(self):
        self.compose = Compose([
            RandomResize(),
            RandomHorizontalFlip(),
        ])
    
    def __call__(self, img, box):
        return self.compose(img, box)


class BoxReshape(object):
    """
    ymin, xmin, ymax, xmax -> xmin, ymin, w, h
    """
    def __init__(self):
        self.final_size = 30
    
    def __call__(self, image, boxes):
        
        image_width, image_height = image.size
        feature_map_size = 30
        
        
        reshape_boxes = torch.Tensor(math.floor(len(boxes)), 4)
        
        for i, box in enumerate(boxes):
            
            box_minY = box[0]
            box_minX = box[1]
            box_maxY = box[2]
            box_maxX = box[3]
                                                                                                           
            N_box_minX = math.floor(feature_map_size * box_minX / image_width)
            N_box_minY = math.floor(feature_map_size * box_minY / image_height)
            N_box_maxX = math.floor(feature_map_size * box_maxX / image_width)
            N_box_maxY = math.floor(feature_map_size * box_maxY / image_height)
            
            N_box_width = math.ceil((box_minX + box_maxX) * feature_map_size / image_width) - 1 - (math.floor(box_minX * feature_map_size / image_width) + 1
            N_box_height = math.ceil((box_minY + box_maxY) * feature_map_size / image_height) - 1 - (math.floor(box_minY * feature_map_size / image_height) + 1)
            
            reshape_boxes[i, 0] = N_box_minX
            reshape_boxes[i, 1] = N_box_minY
            reshape_boxes[i, 2] = N_box_width
            reshape_boxes[i, 3] = N_box_height
            
        #print('re', len(reshape_boxes))
        return reshape_boxes
        

class Compose(object):
    """Composes several augmentations together
    Args:
        transforms (List[Transform]): list of trnasforms to compose.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, boxes):
        #print('trn', self.transforms)
        for transform in self.transforms:
            #print('t', transform)
            img, boxes = transform(img, boxes)
        
        return img, boxes

class Normalize(object):
    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __call__(self, img):
        image_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), \
            torchvision.transforms.Normalize(mean = self.mean,\
                                   std = self.std)
        ])
        
        return image_transform(img)
  
class RandomHorizontalFlip(object):
    def __init__(self):
        self.label = 2
        
    def __call__(self, img, boxes):
        # PIL Image
        width, _ = img.size
        #print('img_size', img.size)
        if random.randint(self.label):
            mirror_img = ImageOps.mirror(img)
            mirror_boxes = []
            
            for box in boxes:
                mirror_box = []
                minX = box[1]
                maxX = box[3]
            
                # Y is not changed
                minY = box[0]
                maxY = box[2]
            
                N_minX = width - maxX
                N_maxX = width - minX

                mirror_box.append(minY)
                mirror_box.append(N_minX)
                mirror_box.append(maxY)
                mirror_box.append(N_maxX)
            
                mirror_boxes.append(mirror_box)
            #print('aa', mirror_img.size)
            #print('mirror_box', len(mirror_boxes))
            #print(boxes[0])
            #print(mirror_boxes[0])
            return mirror_img, mirror_boxes

        else:
            return img, boxes

class RandomResize(object):
    def __init__(self):
        self.size_list = [480, 576, 688, 864, 1200]
        
    def __call__(self, img, boxes):
        
        max_resized_value = random.choice(self.size_list)
        
        ori_img_width, ori_img_height = img.size
        
        
        max_length = max(ori_img_width, ori_img_height)
        min_length = min(ori_img_width, ori_img_height)
            
        resize_ratio = float(max_resized_value / max_length)
        
        min_resized_value = int(min_length * resize_ratio)
        
        if ori_img_width >= ori_img_height:
            resize_img = img.resize((max_resized_value, min_resized_value))
        else:
            resize_img = img.resize((min_resized_value, max_resized_value))
        #print('resize_img', resize_img.size)
        
        resize_boxes = []
        for _, box in enumerate(boxes):
            resize_box = []
            # 0: ymin, 1: xmin, 2: ymax, 3: xmax
        
            for coord in (box):
                coord = (coord - 1) * resize_ratio
                resize_box.append(int(coord))

            resize_boxes.append(resize_box)
        #print('resize_boxes', len(resize_boxes))
        return resize_img, resize_boxes
        
        
        