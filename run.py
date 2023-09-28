import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import warp, make_grid
from evaluator import save_plots
from submodels import Generative_Encoder, Generative_Decoder, Evolution_Network,Noise_Projector
from torch.autograd import Variable
from model_make import *
from dataset import *
from torch.optim.lr_scheduler import StepLR

args = {
    'batch_size': 1,
    'worker': 1,
    'device':'cuda:0',
    'cpu_worker': 1,
    'dataset': 'dBZ',
    'data_path':'Data',
    'input_data_type': 'float32',
    'image_width': 256,
    'image_height': 256,
    'case_type':'normal',
    'total_length':10,
    'input_length':10,
    'pred_length':10,
    'ngf':32,
    
}  # 配置信息
args['ic_feature'] = args['ngf'] * 10
args['evo_ic'] = 20
args['gen_oc'] = 10
configs = args
data_loader = make_dataloader(args)
pred_length = 10
n_epochs = 10000
lr = 0.0001

generator = NowcastNet(configs).to(configs['device'])
discriminator = Discriminator().to(configs['device'])
criterion = torch.nn.BCELoss().to(configs['device'])
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)



scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.8)
scheduler_D = StepLR(optimizer_D, step_size=5, gamma=0.8)


class MaxPoolLoss(nn.Module):
    def __init__(self):
        super(MaxPoolLoss,self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, true_image, fake_image):
        true_image = true_image[...,0]
        fake_image = fake_image[...,0]
        output1 = self.pool1(true_image)
        output2 = self.pool2(fake_image)
        loss = torch.mean(torch.square(output1 - output2))
        return loss

if os.path.exists("Result"):
    shutil.rmtree("Result")
if os.path.exists("X"):
    shutil.rmtree("X")

if os.path.exists("ModelSave"):
    shutil.rmtree("ModelSave")

if os.path.exists("output.output"):
    os.remove("output.output")

if os.path.exists("Y"):
    shutil.rmtree("Y")

os.mkdir("Y")    
os.mkdir("Result")
os.mkdir("X")
os.mkdir("ModelSave")
loss_G_list = []
loss_p_list = []
loss_real_D_list = []
loss_fake_D_list = []
maxpool_loss = MaxPoolLoss().to(configs['device'])
for epoch in range(n_epochs):
    for batch_id, (x_data,y_data) in enumerate(data_loader):
        x_data = x_data.numpy()
        y_data = y_data.numpy()
        x_data = torch.FloatTensor(x_data).to(configs['device'])
        y_data = torch.FloatTensor(y_data).to(configs['device'])
        real_label = torch.ones(10*args['batch_size'],1).cuda()
        real_out = discriminator(y_data[...,:1])
        loss_real_D = criterion(real_out,real_label)
        optimizer_D.zero_grad()
        loss_real_D.backward()
        optimizer_D.step()
        loss_real_D_list.append(loss_real_D)
        
        

        
        fake_label = torch.zeros(10*args['batch_size'],1).cuda()
        frames_tensor = x_data.cpu()
        frames_tensor = torch.FloatTensor(frames_tensor).to(configs['device'])
        fake_img = generator(frames_tensor).detach()
        fake_out = discriminator(fake_img)
        loss_fake_D = criterion(fake_out,fake_label)
        optimizer_D.zero_grad()
        loss_fake_D.backward()
        optimizer_D.step()
        loss_fake_D_list.append(loss_fake_D)
        
        
        genera_x = x_data.cpu()
        genera_x = Variable(torch.FloatTensor(genera_x)).to(configs['device'])
        fake_img = generator(genera_x)
        output = discriminator(fake_img)
        loss_G = criterion(output, torch.ones_like(output))
        loss_p = maxpool_loss(y_data[...,:1],fake_img)
        
        loss_G = loss_G*6 + loss_p*20
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        if batch_id % 10 == 0:
            with open('output.output','a') as file:
                print("Epoch:{},Batch:{}".format(epoch,batch_id),file=file)
                print("loss_real_D:{},loss_fake_D:{},loss_P:{},loss_G:{}".format(loss_real_D.detach().cpu(),loss_fake_D.detach().cpu(),loss_p.detach().cpu(),loss_G.detach().cpu()),file=file)
                print("----------------------------------------",file=file)
        if batch_id % 80 == 0:
            genera_y = y_data.cpu()
            temp_plot = fake_img[0,0,:,:,0]
            save_plots(genera_x[0,0,:,:,0],epoch,batch_id,"X")
            save_plots(genera_y[0,0,:,:,0],epoch,batch_id,"Y")
            save_plots(temp_plot,epoch,batch_id,"Result")
    torch.save(generator, 'ModelSave/epoch_'+str(epoch)+'generator.pth')  
    torch.save(discriminator, 'ModelSave/epoch_'+str(epoch)+'discriminator.pth')
    scheduler_G.step()
    scheduler_D.step()