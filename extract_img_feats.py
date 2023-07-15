# -*- coding: utf-8 -*-
import os
import cv2
import torch
import pickle
import numpy as np
import torchvision
from glob import glob
from tqdm import tqdm
from model import PLModel
from omegaconf import OmegaConf
from collections import OrderedDict
from torchvision import transforms



img_path = '/data/dataset/flickr8k/Flicker8k_Dataset'

config = OmegaConf.load("config.yaml")
clip = PLModel(config['model'])
checkpoint = torch.load('lightning_logs/version_0/checkpoints/last.ckpt', map_location=torch.device('cpu'))
clip.load_state_dict(checkpoint['state_dict'])
clip.eval()
clip.cuda()
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
all_imgs = glob(os.path.join(img_path, '*.jpg'))
feats = {'img_path': all_imgs, 'img_feats': []}
img_size = config['data']['img_size']
with torch.no_grad():
    for _, img_path in enumerate(tqdm(all_imgs)):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = transform(img)
        img = img.unsqueeze(0).cuda()
        img_feat = clip.clip.img_encoder(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        img_feat = img_feat.squeeze().cpu().numpy()
        feats['img_feats'].append(img_feat)

print(len(feats['img_path']))
print(len(feats['img_feats']))
with open('img_feats.pkl', 'wb') as f:
    pickle.dump(feats, f)

