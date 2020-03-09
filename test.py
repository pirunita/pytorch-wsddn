#encoding=utf-8
import argparse
import logging
import os

import cv2
import torch
import torch.nn as nn
import tqdm

from torch.autograd import Variable
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from net import WSDDN
from wsddn_dataset import WSDDNDataset

# Set logger
logger = logging.getLogger('test')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# Set Argument
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--datamode', type=str, default='test')
    
    # Data Directory
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--json_path', default='voc2007.json')
    parser.add_argument('--ssw_path', default='ssw.txt')
    
    # Directory
    parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='Load pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint info')
    parser.add_argument('--checkpoint', type=str, default='wsddn.pkl')
    parser.add_argument('--result_dir', type=str, default='result')
    
    args = parser.parse_args()
    
    return args

def draw_rect(args, file_name, rect):
    
    image = cv2.imread(os.path.join(args.dataroot, args.datamode, args.jpeg_path, file_name + '.jpg'))
    
    left = int(16 * rect[0])
    top = int(16 * rect[1])
    width = int(16 * rect[2])
    height = int(16 * rect[3])
    
    
    
    rect_image = cv2.rectangle(image, (left, top), (left+width, top+height), (0, 255, 0), 2)
    
    cv2.imwrite(os.path.join(args.result_dir, file_name + '.png'), rect_image)
    
    #image.show()
    #rect_image = ImageDraw.Draw(image)
    #rect_image.rectangle([(left, top), (left+width, top+height)], fill=None, outline='red', width=2)
    
    #image.show()
    #image.save(os.path.join(args.result_dir, str(idx) + '.png'))

def test(args, model):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    
    
    if args.datamode == 'test':
        test_data = WSDDNDataset(args)
        test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
        model.cuda()
        model.eval()
        for i, (file_name, images, ssw_block, labels) in tqdm.tqdm(enumerate(test_loader)):
            file_name = file_name[0]
            images = Variable(images).cuda()
            ssw_block = Variable(ssw_block).cuda()
            labels = Variable(labels).cuda()
            
            output, output_clf, output_det = model(images, ssw_block)
            """
            i: # of datasets
            j: # of classes
            k: # of proposal
            ssw_block: 16*left, 16*top, 16*width, 16*height
            """
            
            for j in range(output.shape[1]):
                if output[0, j] > 0.05:
                    for k in range(output_det.shape[0]):
                        if output_det[0, k, j] > 0.1:
                            draw_rect(args, file_name, ssw_block[0, k, :].data.cpu().numpy())
                            #draw_rect(args, file_name, images.data.cpu().clone(), ssw_block[0, k, :].data.cpu().numpy())
            #debugging
            #print(output_clf[0,0, :])
            
            #if i == 2:
            #    break
                
            
        

if __name__=='__main__':
    args = get_args()
    logger.info(args)

    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    model = WSDDN()
    
    if os.path.exists(os.path.join(args.checkpoint_dir, args.checkpoint)):
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint)))
    
    test(args, model)
        