# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50
from transformers import AutoModel
from data import build_dataloader
import pytorch_lightning as pl


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Sequential()
        self.img_proj = ProjectionHead(2048, config['embed_dim'], config['dropout'])
        for p in self.encoder.parameters():
            p.requires_grad = config['trainable']
        for p in self.img_proj.parameters():
            p.requires_grad = config['trainable']
        self.lr = config['lr']

    def forward(self, x):
        return self.img_proj(self.encoder(x))


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_proj = ProjectionHead(768, config['embed_dim'], config['dropout'])
        for p in self.encoder.parameters():
            p.requires_grad = config['trainable']
        for p in self.text_proj.parameters():
            p.requires_grad = config['trainable']
        self.lr = config['lr']

    def forward(self, x):
        input_ids = x['input_ids'].squeeze(1)
        attention_mask = x['attention_mask'].squeeze(1)
        pred = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.text_proj(pred['last_hidden_state'][:, 0])


class ProjectionHead(nn.Module):
    def __init__(self, feat_dim, embed_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(feat_dim, embed_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_encoder = ImageEncoder(config['img_encoder'])
        self.text_encoder = TextEncoder(config['text_encoder'])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        img = x['img'].float()
        img_feat = self.img_encoder(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text = x['caption']
        text_feat = self.text_encoder(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        return img_feat, text_feat


class PLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.clip = CLIP(config)
        self.logit_scale = self.clip.logit_scale
        self.ce_loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        self.clip.train()
        img_feat, text_feat = self.clip(batch)
        logits_per_img = self.logit_scale.exp() * img_feat @ text_feat.t()
        logits_per_text = logits_per_img.t()
        ground_truth = torch.arange(len(logits_per_img)).long().to(logits_per_img.device)
        loss = (self.ce_loss(logits_per_img, ground_truth) + self.ce_loss(logits_per_text, ground_truth)) / 2
        i2t_acc = (logits_per_img.argmax(-1) == ground_truth).sum() / len(logits_per_img)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        self.log_dict({'loss': loss, 'i2t': i2t_acc, 't2i': t2i_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.clip.eval()
        with torch.no_grad():
            img_feat, text_feat = self.clip(batch)
            logits_per_img = self.logit_scale.exp() * img_feat @ text_feat.t()
            logits_per_text = logits_per_img.t()
            ground_truth = torch.arange(len(logits_per_img)).long().to(logits_per_img.device)
            loss = (self.ce_loss(logits_per_img, ground_truth) + self.ce_loss(logits_per_text, ground_truth)) / 2
            i2t_acc = (logits_per_img.argmax(-1) == ground_truth).sum() / len(logits_per_img)
            t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
            self.log_dict({'loss': loss, 'i2t': i2t_acc, 't2i': t2i_acc},  on_epoch=True)
            return loss

    def configure_optimizers(self):
        params = []

        img_encoder_params = []
        for p in filter(lambda p: p.requires_grad, self.clip.img_encoder.parameters()):
            img_encoder_params.append(p)
        params.append({'params': img_encoder_params, 'lr': self.clip.img_encoder.lr})

        text_encoder_params = []
        for p in filter(lambda p: p.requires_grad, self.clip.text_encoder.parameters()):
            text_encoder_params.append(p)
        params.append({'params': text_encoder_params, 'lr': self.clip.text_encoder.lr})

        optimizer = torch.optim.Adam(params)
        return optimizer


if __name__ == '__main__':
    # model = ImageEncoder()
    # x = torch.randn(1,3,224,224)
    # y = model(x)
    # print(y.shape)

    from omegaconf import OmegaConf

    config = OmegaConf.load("config.yaml")
    dataset = build_dataloader(config['data'])
    x = dataset.__iter__().__next__()
    model = CLIP(config['model'])
    img_feat, text_feat = model(x)

