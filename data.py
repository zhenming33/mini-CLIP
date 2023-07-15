# -*- coding: utf-8 -*-
import os
import cv2
import random
import torchvision
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path, img_size, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len
        self.img_size = img_size
        self.label_path = os.path.join(data_path, 'Flickr8k.token.txt')
        self.img_data_path = os.path.join(data_path, 'Flicker8k_Dataset')
        self.labels = OrderedDict()
        with open(self.label_path, 'r', encoding='utf-8') as f:
            for line in f:
                img_name, caption = line.strip().split('\t')
                img_name = img_name.split('#')[0]
                if not os.path.exists(os.path.join(self.img_data_path, img_name)):
                       continue
                if img_name not in self.labels:
                    self.labels[img_name] = []
                self.labels[img_name].append(caption)
        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.labels.keys())

    def __getitem__(self, idx):
        img_name = sorted(list(self.labels.keys()))[idx]
        img = cv2.imread(os.path.join(self.img_data_path, img_name))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)
        caption = random.choice(self.labels[img_name])
        caption = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_len,
                                 return_tensors="pt")

        return {'img': img, 'caption': caption}


def build_dataloader(data_config):
    dataset = MyDataset(data_config['data_path'], data_config['img_size'], data_config['max_cap_len'])
    dataloader = DataLoader(dataset, num_workers=0, batch_size=data_config['batch_size'], shuffle=True)
    return dataloader


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    dataset = build_dataloader(config['data'])
    x = dataset.__iter__().__next__()

