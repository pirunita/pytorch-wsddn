import argparse
import math
import os

import cv2
import numpy as np
import selectivesearch
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--data_list', default='annotations.txt')
    parser.add_argument('--data_path', default='JPEGImages')
    parser.add_argument('--ssw_list', default='ssw.txt')
    
    args = parser.parse_args()
    
    return args


def ssw(img, scale=500, sigma=0.7, min_size=20):
    img_label, regions = selectivesearch.selective_search(img, scale=scale, sigma=sigma, min_size=min_size)
    candidates = set()
    for proposal in regions:
        if proposal['rect'] in candidates:
            continue
        if proposal['size'] < 2000:
            continue
        
        candidates.add(proposal['rect'])
    return candidates


def feature_mapping(regions):
    mapping = []
    for ele in regions:
        mapping.append((math.floor(ele[0]/16) + 1, \
                        math.floor(ele[1]/16) + 1, \
                        math.ceil((ele[0]+ele[2])/16) - 1 - (math.floor(ele[0]/16) + 1), \
                        math.ceil((ele[1]+ele[3])/16) - 1 - (math.floor(ele[1]/16) + 1))) 
    mapping = list(set(mapping))
    return mapping
    
        
def main():
    args = get_args()
    
    
    fr = open(os.path.join(args.dataroot, args.data_list), 'r')
    with open(os.path.join(args.ssw_list), 'w') as fw:
    
        for word in tqdm.tqdm(fr.readlines()):
            word = word.rstrip()
            image = cv2.imread(os.path.join(args.dataroot, args.data_path, str(word)))
            image = cv2.resize(image, (480, 480))
            region_proposal = ssw(image)
            region_proposal = feature_mapping(region_proposal)
            region_proposal = list(np.array(region_proposal).flat)
            output_proposal = str(word) + " " + " ".join(str(i) for i in region_proposal) + '\n'
            fw.write(output_proposal)
    fr.close()
    fw.close()
    
        


if __name__ == '__main__':
    args = get_args()
    print(args)
    
    main()
