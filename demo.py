# -*- coding: utf-8 -*-
import cv2
import torch
import pickle
import numpy as np
import gradio as gr
from model import PLModel
from omegaconf import OmegaConf
from collections import OrderedDict
from transformers import AutoTokenizer

config = OmegaConf.load("config.yaml")
clip = PLModel(config['model'])
checkpoint = torch.load('lightning_logs/version_0/checkpoints/last.ckpt', map_location=torch.device('cpu'))
clip.load_state_dict(checkpoint['state_dict'])
clip.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

with open('img_feats.pkl', 'rb') as f:
    img_feats = pickle.load(f)
img_feats['img_feats'] = np.array(img_feats['img_feats'])

def T2I(input_text):
    with torch.no_grad():
        text = tokenizer(input_text, padding="max_length", truncation=True, max_length=config['data']['max_cap_len'],
                            return_tensors="pt")
        text_feat = clip.clip.text_encoder(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.squeeze().cpu().numpy()
        res = text_feat @ img_feats['img_feats'].T
        idx = np.argmax(res)
        img = cv2.imread(img_feats['img_path'][idx])[:, :, ::-1]
        return img


if __name__ == '__main__':
    with gr.Blocks() as demo:
        caption = gr.Textbox(label="text")
        output = gr.Image(label="image")
        greet_btn = gr.Button("text to image")
        greet_btn.click(fn=T2I, inputs=caption, outputs=output)

    demo.launch(server_name='0.0.0.0', server_port=7861)

