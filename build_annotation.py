import argparse
import os
import glob
import json

import tqdm

import xml.etree.ElementTree as Et
"""
0: aeroplane, 1: bicycle, 2: bird, 3: boat, 4: bottle, 5: bus,
6: car, 7: cat, 8: chair, 9: cow, 10: diningtable, 11: dog,
12: horse, 13: motorbike, 14: person, 15: pottedplant, 16: sheep, 
17: sofa, 18: train, 19: tvmonitor
"""

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES_TO_INT = dict(zip(CLASSES, range(len(CLASSES))))

print(CLASSES_TO_INT)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data')
    parser.add_argument('--data_path', default='Annotations')
    parser.add_argument('--json_path', default='voc2007.json')
    parser.add_argument('--text_path', default='annotations.txt')
    
    args = parser.parse_args()
    
    return args


def main():
    args = get_args()
    print(args)
    xml_file_list = sorted(os.listdir(os.path.join(args.dataroot, args.data_path)))
    
    ff = open(os.path.join(args.dataroot, args.text_path), 'w')
    with open(os.path.join(args.dataroot, args.json_path), 'w') as jf:
        label_dict = {}
        for xml_file_name in tqdm.tqdm(xml_file_list):
            file_name = os.path.splitext(xml_file_name)[0]
            ff.write(file_name + '\n')
            
            xml_file = open(os.path.join(args.dataroot, args.data_path, xml_file_name), 'r')
            xml_tree = Et.parse(xml_file)
            xml_root = xml_tree.getroot()
            xml_objects = xml_root.findall('object')
            
            label_list = []
            for _object in xml_objects:
                name = _object.find('name').text
                if not CLASSES_TO_INT[name] in label_list:
                    label_list.append(CLASSES_TO_INT[name])
            label_dict[file_name] = label_list
        
        json.dump(label_dict, jf)
        
    """
    with open(os.path.join('data/annotations.txt'), 'w') as f:
        for ele in tqdm.tqdm(file_list):
            f.write(ele + "\n")
    """


if __name__=='__main__':
    main()