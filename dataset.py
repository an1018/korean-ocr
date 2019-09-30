import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import os
from utils import CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST


def korean2onehot(w):
    if "가" <= w <= "힣":
        ch1 = (ord(w) - ord("가")) // 588
        ch2 = ((ord(w) - ord("가")) - (588 * ch1)) // 28
        ch3 = (ord(w) - ord("가")) - (588 * ch1) - 28 * ch2
        first = torch.eye(len(CHOSUNG_LIST))[ch1]
        middle = torch.eye(len(JUNGSUNG_LIST))[ch2]
        last = torch.eye(len(JONGSUNG_LIST))[ch3]
        return torch.cat((first, middle, last))
    else:
        raise ValueError("Not valid korean")


class OCRDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.classes = set()
        self.data = self._get_data()
        self.transforms = transforms

    def _get_data(self):
        data = []
        characters = os.listdir(self.root)
        for character in characters:
            full_dir = os.path.join(self.root, character)
            if os.path.isdir(full_dir):
                onehot_char = korean2onehot(character)
                filenames = os.listdir(full_dir)
                for filename in filenames:
                    full_filename = os.path.join(self.root, character, filename)
                    ext = os.path.splitext(full_filename)[-1]
                    if ext == ".jpeg":
                        data.append((onehot_char, full_filename))
                        self.classes.add(onehot_char)
        return data

    def __getitem__(self, index):
        character, filename = self.data[index]
        img = Image.open(filename)
        if self.transforms is not None:
            img = self.transforms(img)

        return (character, img)

    def __len__(self):
        return len(self.data)


class PHD08(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.classes = set()
        self.data = self._get_data()
        self.transforms = transforms

    def _get_data(self):
        data = []
        characters = os.listdir(self.root)
        for character in characters:
            ext = os.path.splitext(character)[-1]
            if ext == ".npy":
                npy_path = os.path.join(self.root, character)
                npy = np.load(npy_path)

                onehot_char = korean2onehot(character.split(".")[0])
                self.classes.add(onehot_char)

                for one_data in npy:
                    norm = np.uint8(255 - one_data)
                    img = Image.fromarray(norm)
                    data.append((onehot_char, img))
        return data

    def __getitem__(self, index):
        character, img = self.data[index]
        if self.transforms is not None:
            img = self.transforms(img)

        return (character, img)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # dataset = OCRDataset(
    #     "/home/heonsong/Desktop/HWR/datasets/tensorflow-hangul-recognition/512/hangul-images"
    # )
    # print(len(dataset.classes))

    # dataset = PHD08("/home/heonsong/Disk2/Dataset/phd08_split/train")

    print(korean2onehot("종"))
