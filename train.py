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
    parser.add_argument('--json_path', default='voc2007.json')
    parser.add_argument('--ssw_path', default='ssw.txt')
    
    parser.add_argument('--pretrained_dir', type=str, default='pretrained', help='Load pretrained model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint info')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard info')
    
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
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False, num_workers=1)
        model.train()
        for step in tqdm.tqdm(range(args.epoch)):
            running_loss = 0.0
            train_size = 0
            for i, (images, ssw_block, labels) in tqdm.tqdm(enumerate(train_loader)):
                
                images = Variable(images).cuda()
                ssw_block = Variable(ssw_block).cuda()
                labels = Variable(labels).cuda()
                if step < 10:
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
                
                output, output_clf, output_dct = model(images, ssw_block)
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
    
    model = WSDDN()
    
    pretrained_dict = torch.load(os.path.join(args.pretrained_dir, 'vgg11_bn-6002323d.pth'))
    modified_dict = model.state_dict()
    
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modified_dict}
    modified_dict.update(pretrained_dict)
    model.load_state_dict(modified_dict)
    
    train(args, model)
    