#encoding=utf-8
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from torch.autograd import Variable
from tensorboardX import SummaryWriter

from net import WSDDN
from wsddn_dataset import WSDDNDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001, \
        help='Base learning rate for Adam')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--datamode', type=str, default='train')
    
    # Directory
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--image_label_path', default='voc_2007_trainval_image_label.json')
    #parser.add_argument('--ssw_path', default='ssw.txt')
    parser.add_argument('--ssw_path', default='voc_2007_trainval.mat')
    
    parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='Load pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint info')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard info')
    
    # Backbone Network
    parser.add_argument('--backbone_network', type=str, default='vgg11')
    
    args = parser.parse_args()
    
    return args


def train(args, model):
    criterion = nn.BCELoss(weight=None, size_average=True)
    model.cuda()
    optimizer1 = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer2 = optim.SGD(model.parameters(), lr=0.1 * args.lr, momentum=0.9)
    
    # Visualization
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)
    board = SummaryWriter(log_dir=args.tensorboard_dir)
    
    # Checkpoint
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    
    if args.datamode == 'train':
        train_data = WSDDNDataset(args)
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model.train()
        for step in tqdm.tqdm(range(args.epoch)):
            running_loss = 0.0
            train_size = 0
            for i, (file_name, images, images_width, images_height, ssw_block, labels) in tqdm.tqdm(enumerate(train_loader)):
                
                images_width = Variable(images_width).cuda()
                images_height = Variable(images_height).cuda()
                images = Variable(images).cuda()
                ssw_block = Variable(ssw_block).cuda()
                labels = Variable(labels).cuda()
                if step < 10:
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                
                output, output_clf, output_det = model(images, ssw_block, images_width, images_height)
                output = torch.sigmoid(output)
                loss = criterion(output, labels)
                
                if step < 10:
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                loss.backward()
                
                if step < 10:
                    optimizer1.step()
                else:
                    optimizer2.step()
                
                running_loss += loss.item()
                train_size += 1
            
            print("Step: {step} Loss: {loss}".format( \
                step=step + 1, loss=running_loss/train_size
            ))
            

            board.add_scalar('Train/loss', running_loss/train_size, step+1)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'wsddn.pkl'))
        
        print('Finished Training')
        board.close()
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'wsddn.pkl'))
                    
                    
    """
    test_data = WSDDNDataset(args)
    
    
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    """
    
    
    
    
    
if __name__ == '__main__':
    args = get_args()
    print(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    model = WSDDN(args)
    
    if args.backbone_network == 'vgg11':
        pretrained_dict = torch.load(os.path.join(args.pretrained_dir, 'vgg11_bn-6002323d.pth'))
    elif args.backbone_network == 'alexnet':
        pretrained_dict = torch.load(os.path.join(args.pretrained_dir, 'alexnet-owt-4df8aa71.pth'))
    
    modified_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    model.load_state_dict(modified_dict)
    
    train(args, model)
    
