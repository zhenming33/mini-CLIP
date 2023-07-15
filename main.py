# -*- coding: utf-8 -*-
import os
import torch
from model import PLModel
import pytorch_lightning as pl
from omegaconf import OmegaConf
from data import build_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint

config = OmegaConf.load("config.yaml")
train_loader = build_dataloader(config['data'])
valid_loader = build_dataloader(config['data'])

model = PLModel(config['model'])
#checkpoint = torch.load('lightning_logs/version_3/checkpoints/last.ckpt', map_location=torch.device('cpu'))
#model.load_state_dict(checkpoint['state_dict'])


checkpoint_callback = ModelCheckpoint(save_last=True, every_n_epochs=config['train']['save_n_epoch'])

trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config['train']['max_epochs'], callbacks=[checkpoint_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

