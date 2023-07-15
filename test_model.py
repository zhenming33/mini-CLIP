# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

import cv2
import torch
import numpy as np
import torchvision
from glob import glob
from tqdm import tqdm
from model import CLIP
from omegaconf import OmegaConf
from collections import OrderedDict
from transformers import AutoTokenizer
from model import PLModel
from data import build_dataloader
from torchvision import transforms


img_path = '/data/dataset/flickr8k/Flicker8k_Dataset/1001773457_577c3a7d70.jpg'
text = 'A black dog and a white dog with brown spots are staring at each other in the street .'

config = OmegaConf.load("config.yaml")
clip = PLModel(config['model'])
checkpoint = torch.load('lightning_logs/version_0/checkpoints/last.ckpt', map_location=torch.device('cpu'))
clip.load_state_dict(checkpoint['state_dict'])
clip.eval()


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
with torch.no_grad():
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = transform(img)
    img = img.unsqueeze(0)
    img_feat = clip.clip.img_encoder(img)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    text = tokenizer(text, padding="max_length", truncation=True, max_length=config['data']['max_cap_len'],
                     return_tensors="pt")
    text_feat = clip.clip.text_encoder(text)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    print(img_feat @ text_feat.t())





