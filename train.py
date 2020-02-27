#coding=utf-8
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from net import WSDDN

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001, \
        help='Base learning rate for Adam')
    parser.add_argument('--epoch', type=int, default=40)
    
    # Directory
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--datamode', default='train')
    parser.add_argument('--data_list', default='annotations.txt')
    parser.add_argument('--ssw_list', default='ssw.txt')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint info')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard info')
    
    args = parser.parse_args()
    
    return args


def train(args, model):
    loss = nn.BCELoss(weight=None, size_average=True)
    model.cuda()
    optimizer1 = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(model.parameters(), lr=0.1 * args.lr, momentum=0.9)
    board = SummaryWriter(log_dir=args.tensorboard_dir)
    
    
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    model = WSDDN()
    
    pretrained_dict = torch.load('vgg11_bn-6002323d.pth.1')
    modified_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    model.load_state_dict(modified_dict)
    
    train(args, model)
    