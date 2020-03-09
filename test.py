#encoding=utf-8
import argparse
import os



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--jpeg_path', default='JPEGImages')
    parser.add_argument('--text_path', default='annotations.txt')
    parser.add_argument('--json_path', default='voc2007.json')
    parser.add_argument('')