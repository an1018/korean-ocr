import os
import io
import random
from shutil import copyfile
import numpy as np


def split_val_test(rootdir, destdir, val_ratio, test_ratio):
    train_dir = os.path.join(destdir, "Split", "train")
    val_dir = os.path.join(destdir, "Split", "val")
    test_dir = os.path.join(destdir, "Split", "test")
    characters = os.listdir(rootdir)
    for character in characters:
        full_dir = os.path.join(rootdir, character)
        if os.path.isdir(full_dir):
            if not os.path.exists(os.path.join(train_dir, character)):
                os.makedirs(os.path.join(train_dir, character))
            if not os.path.exists(os.path.join(val_dir, character)):
                os.makedirs(os.path.join(val_dir, character))
            if not os.path.exists(os.path.join(test_dir, character)):
                os.makedirs(os.path.join(test_dir, character))
            filenames = os.listdir(full_dir)
            valid_files = [
                filename
                for filename in filenames
                if os.path.splitext(filename)[-1] == ".jpeg"
            ]
            random.shuffle(valid_files)
            train_end = int(len(valid_files) * (1 - val_ratio - test_ratio))
            val_end = int(len(valid_files) * val_ratio)
            for filename in valid_files[:train_end]:
                full_filename = os.path.join(rootdir, character, filename)
                copyfile(full_filename, os.path.join(train_dir, character, filename))
            for filename in valid_files[train_end : train_end + val_end]:
                full_filename = os.path.join(rootdir, character, filename)
                copyfile(full_filename, os.path.join(val_dir, character, filename))
            for filename in valid_files[train_end + val_end :]:
                full_filename = os.path.join(rootdir, character, filename)
                copyfile(full_filename, os.path.join(test_dir, character, filename))


def split_val_test_npy(rootdir, destdir, val_ratio, test_ratio):
    train_dir = os.path.join(destdir, "Split", "train")
    val_dir = os.path.join(destdir, "Split", "val")
    test_dir = os.path.join(destdir, "Split", "test")
    if not os.path.exists(os.path.join(train_dir)):
        os.makedirs(os.path.join(train_dir))
    if not os.path.exists(os.path.join(val_dir)):
        os.makedirs(os.path.join(val_dir))
    if not os.path.exists(os.path.join(test_dir)):
        os.makedirs(os.path.join(test_dir))
    characters = os.listdir(rootdir)
    for character in characters:
        ext = os.path.splitext(character)[-1]
        if ext == ".npy":
            npy_path = os.path.join(rootdir, character)
            npy = np.load(npy_path)

            idx = list(range(len(npy)))
            random.shuffle(idx)
            train_end = int(len(idx) * (1 - val_ratio - test_ratio))
            val_end = int(len(idx) * val_ratio)

            train_npy = npy[idx[:train_end]]
            np.save(os.path.join(train_dir, character), train_npy)

            val_npy = npy[idx[train_end : train_end + val_end]]
            np.save(os.path.join(val_dir, character), val_npy)

            test_npy = npy[idx[train_end + val_end :]]
            np.save(os.path.join(test_dir, character), test_npy)


def split_subset_npy(rootdir, destdir, label_file):
    with io.open(label_file, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    subset_dir = os.path.join(destdir, "Subset")
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
    for character in labels:
        character = character + ".npy"
        npy_path = os.path.join(rootdir, character)
        copyfile(npy_path, os.path.join(subset_dir, character))


if __name__ == "__main__":
    random.seed(1234)
    # split_val_test_npy("/home/heonsong/Disk2/Dataset/phd08","/home/heonsong/Disk2/Dataset",0.2, 0.2)
    # split_subset_npy("/home/heonsong/Disk2/Dataset/PHD08/phd08_2350_split/train", "/home/heonsong/Disk2/Dataset/PHD08/phd08_256_split", "256-common-hangul.txt")
    # split_val_test(
    #     "/home/heonsong/Disk2/Dataset/Handmade-OCR/512",
    #     0.2,
    #     0.2,
    # )

    # split_small("/home/heonsong/Disk2/Dataset/Handmade-OCR/512", 0.3)

    # split_diff(
    #     "/home/heonsong/Disk2/Dataset/Handmade-OCR/2350",
    #     "/home/heonsong/Disk2/Dataset/Handmade-OCR/512",
    # )
