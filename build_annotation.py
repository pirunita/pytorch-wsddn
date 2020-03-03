import argparse
import os
import glob
import tqdm

import xml.etree.ElementTree as Et
"""
1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus,
7: car, 8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog,
13: horse, 14: motorbike, 15: person, 16: pottedplant, 17: sheep, 
18: sofa, 19: train, 20: tvmonitor
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--data_path', default='Annotations')
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    print(args)
    
    file_list = sorted(os.listdir(os.path.join(args.dataroot, args.data_path)))
    
    
    for file_name in tqdm.tqdm(file_list):
        xml_file = open(os.path.join(args.dataroot, args.data_path, file_name), 'r')
        xml_tree = Et.parse(xml_file)
        xml_root = xml_tree.getroot()
        
        xml_objects = xml_root.find('object')
        
        for _object in xml_objects:
            name = _object.find('name').text
        
        break
        
    """
    with open(os.path.join('data/annotations.txt'), 'w') as f:
        for ele in tqdm.tqdm(file_list):
            f.write(ele + "\n")
    """


if __name__=='__main__':
    main()