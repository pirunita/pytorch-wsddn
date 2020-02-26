import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001, \
        help='Base learning rate for Adam')
    parser.add_argument('--epoch', type=int, default=40)
    
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    
    # Setting GPU number
    torch.cuda.set_device(args.gpu_id)
    
    model = WSDDN()