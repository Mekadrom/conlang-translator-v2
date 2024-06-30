from tqdm import tqdm

import argparse
import shutil

def split_split(split, n_files):
    if n_files <= 1:
        shutil.move(f"data/{split}.src", f"data/{split}_0.src")
        shutil.move(f"data/{split}.tgt", f"data/{split}_0.tgt")
        return

    print(f"Splitting {split}...")

    src_datafile = f"data/{split}.src"
    tgt_datafile = f"data/{split}.tgt"

    with open(src_datafile, 'r') as src_file, open(tgt_datafile, 'r') as tgt_file:
        src_data = src_file.readlines()
        tgt_data = tgt_file.readlines()

    shutil.move(src_datafile + '.bak', src_datafile + '.bak.bak')
    shutil.move(tgt_datafile + '.bak', tgt_datafile + '.bak.bak')

    shutil.move(src_datafile, src_datafile + '.bak')
    shutil.move(tgt_datafile, tgt_datafile + '.bak')

    total_data_len = len(src_data)

    split_data_len = total_data_len // n_files

    for i in range(n_files):
        with open(f"data/{split}_{i}.src", 'a') as src_file, open(f"data/{split}_{i}.tgt", 'a') as tgt_file:
            for src_line, tgt_line in tqdm(zip(src_data[i * split_data_len:(i + 1) * split_data_len], tgt_data[i * split_data_len:(i + 1) * split_data_len]), total=split_data_len, desc=f"Splitting {split}..."):
                src_file.write(f"{src_line}")
                tgt_file.write(f"{tgt_line}")

    print(f"Split {split} into {n_files} files.")

def split(n_files):
    split_split('train', n_files)
    split_split('validation', 1)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--n_files', type=int, default=1)

    args = argparser.parse_args()

    split(args.n_files)