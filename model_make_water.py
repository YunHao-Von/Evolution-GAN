import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import warp, make_grid
from submodels import Generative_Encoder, Generative_Decoder, Evolution_Network,Noise_Projector
from torchvision.models import resnet50




class NowcastNet(nn.Module):
    def __init__(self, configs):
        super(NowcastNet, self).__init__()
        self.configs = configs
        self.pred_length = configs['pred_length']
        self.evo_net = Evolution_Network(self.configs['input_length'], self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.configs['total_length'], base_c=self.configs['ngf'])
        self.gen_dec = Generative_Decoder(self.configs)
        self.proj = Noise_Projector(self.configs['ngf'], configs)
        self.conv_merge = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=1)
        sample_tensor = torch.zeros(1, 1, self.configs['image_height'], self.configs['image_width'])
        self.grid = make_grid(sample_tensor)


    def forward(self, all_frames):
        # all_frames = all_frames[:, :, :, :, :1]
        all_frames = all_frames.permute(0,4,2,3,1)
        all_frames = self.conv_merge(all_frames)
        all_frames = all_frames.permute(0,4,2,3,1)
        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.configs['input_length']]
        input_frames = input_frames.reshape(batch, self.configs['input_length'], height, width)

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.configs['input_length'] - 1):self.configs['input_length'], :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid.cuda(), mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result
        
        # Generative Network
        evo_feature = self.gen_enc(input_frames+evo_result)

        noise = torch.randn(batch, self.configs['ngf'], height // 32, width // 32).cuda()
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_feature)

        return gen_result.unsqueeze(-1)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Load a pre-trained ResNet50 model
        self.resnet = resnet50(pretrained=False)

        # Modify the first layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the last layer to output a single probability value
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze().unsqueeze(1) # Reshape the input tensor
        return self.resnet(x)