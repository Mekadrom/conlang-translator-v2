from tqdm import tqdm

import argparse
import glob
import random
import shutil

def shuffle_file_pair(src_datafile, tgt_datafile, sort):
    with open(src_datafile, 'r') as src_file, open(tgt_datafile, 'r') as tgt_file:
        src_data = src_file.readlines()
        tgt_data = tgt_file.readlines()

    shutil.move(src_datafile, src_datafile + '.shuf.bak')
    shutil.move(tgt_datafile, tgt_datafile + '.shuf.bak')

    if sort:
        # sort by length of tgt_data
        src_data, tgt_data = zip(*sorted(zip(src_data, tgt_data), key=lambda x: len(x[1])))
        src_data = list(src_data)
        tgt_data = list(tgt_data)

    with open(src_datafile, 'a') as src_file, open(tgt_datafile, 'a') as tgt_file:
        if sort:
            for src_line, tgt_line in tqdm(zip(src_data, tgt_data), desc=f"Writing sorted {src_datafile} and {tgt_datafile}..."):
                src_file.write(src_line)
                tgt_file.write(tgt_line)
        else:
            with tqdm(total=len(src_data), desc=f"Shuffling {src_datafile} and {tgt_datafile}...") as pbar:
                while len(src_data) > 0:
                    data_idx = random.randint(0, len(src_data) - 1)

                    src_file.write(src_data.pop(data_idx))
                    tgt_file.write(tgt_data.pop(data_idx))

                    pbar.update(1)

def shuffle(sort):
    src_files = glob.glob("data/train_*.src")
    tgt_files = glob.glob("data/train_*.tgt")

    for src_file, tgt_file in tqdm(zip(sorted(src_files), sorted(tgt_files)), ):
        shuffle_file_pair(src_file, tgt_file, sort)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--sort', action='store_true')

    args = argparser.parse_args()

    shuffle(args.sort)
