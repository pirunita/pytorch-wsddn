import os
import glob
import tqdm

def main():
    file_list = sorted(os.listdir(os.path.join('data/JPEGImages/')))
    with open(os.path.join('annotations.txt'), 'w') as f:
        for ele in tqdm.tqdm(file_list):
            f.write(ele + "\n")
    


if __name__=='__main__':
    main()