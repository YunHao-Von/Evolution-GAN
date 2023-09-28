import numpy as np
import os
import shutil
from torch.utils.data import Dataset,DataLoader
import torch
import cv2
from scipy.signal import medfilt2d

class SinaDataset(Dataset):
    def __init__(self, input_param,time_window=1):
        self.data_path = input_param['data_path']
        self.image_height = input_param['image_height']
        self.image_width = input_param['image_width']
        self.total_length = input_param['total_length']
        self.input_data_type = input_param['input_data_type']
        start_dict = {}
        for dataset in ['dBZ']:
            start_npy = []
            water_npy = []
            zdr_npy = []
            water_path = 'Rain'
            zdr_path = 'Data/'+'ZDR'+'/3.0km'
            data_path = 'Data/'+dataset+'/3.0km'
            # name_list = os.listdir(data_path)
            name_list = [
            'data_dir_027', 'data_dir_083', 'data_dir_022', 'data_dir_047', 'data_dir_099', 'data_dir_006', 'data_dir_057',
            'data_dir_028', 'data_dir_065', 'data_dir_098', 'data_dir_054', 'data_dir_029', 'data_dir_176', 'data_dir_013',
            'data_dir_001', 'data_dir_000', 'data_dir_055', 'data_dir_178', 'data_dir_018', 'data_dir_019','data_dir_014']
            name_list = sorted(name_list, key=lambda text:int(text[-3:]))
            for data_dir in name_list:
                rain_path = data_path + str('/') + data_dir
                num_frame = len(os.listdir(rain_path))
                if num_frame < 20:
                    continue
                start_npy.append(data_path + str('/') + data_dir+'/frame_'+str(0).zfill(3) + '.npy')
                water_npy.append(water_path + str('/') +data_dir+'/frame_'+str(0).zfill(3) + '.npy')
                zdr_npy.append(zdr_path + str('/') +data_dir+'/frame_'+str(0).zfill(3) + '.npy')
                for i in range(0,num_frame-20,time_window):
                    start_npy.append(data_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
                    water_npy.append(water_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
                    zdr_npy.append(zdr_path + str('/') + data_dir+'/frame_'+str(i+time_window).zfill(3) + '.npy')
            start_dict[dataset] = start_npy
            start_dict['WATER'] = water_npy
            start_dict['ZDR'] = zdr_npy

        self.start_dict = start_dict
    
    def load(self,file_name,norm_param):
        data_frame = np.load(file_name)
        data_frame = medfilt2d(data_frame,kernel_size=3)
        norm_dict = {'dBZ': [0, 65],'ZDR': [-1, 5],'WATER': [0, 1]}
        mmin, mmax = norm_dict[norm_param]
        data_frame = (data_frame-mmin) / (mmax - mmin)
        return data_frame.astype(self.input_data_type)
    
    
    def mask(self,data_array):
        data_mask = np.ones_like(data_array)
        data_mask[data_array < 0] = 0
        # data_array[data_array < 0] = 0
        # data_array = np.clip(data_array, 0, 65)  # 对数组中的元素进行截断
        xy_data = np.zeros((self.total_length, self.image_height, self.image_width, 1))
        xy_data[..., 0] = data_array
        # xy_data[..., 1] = data_mask
        return xy_data

    def get_dataset_data(self,index,data_name):
        self.start_npy = self.start_dict[data_name]
        initial_filename = self.start_npy[index]
        directory, filename = initial_filename.rsplit('/', maxsplit=1)
        prefix, extension = filename.rsplit('.', maxsplit=1)
        initial_frame_number = int(prefix[-3:])
        x_file_name = [initial_filename]
        y_file_name = []
        # 生成后续文件名并打印
        for i in range(1, 10):  # 生成帧号从1到20的文件名
            new_frame_number = initial_frame_number + i
            new_filename = f'{directory}/frame_{str(new_frame_number).zfill(3)}.{extension}'
            x_file_name.append(new_filename)
        for i in range(10,20):
            new_frame_number = initial_frame_number + i
            new_filename = f'{directory}/frame_{str(new_frame_number).zfill(3)}.{extension}'
            y_file_name.append(new_filename)
        
        array_shape = (self.total_length,self.image_width,self.image_height)
        x_array = np.zeros(array_shape)
        y_array = np.zeros(array_shape)
        for i in range(self.total_length):
            x_array[i,:] = self.load(x_file_name[i],data_name)
            y_array[i,:] = self.load(y_file_name[i],data_name)
        x_data = self.mask(x_array)
        y_data = self.mask(y_array)
        return x_data,y_data
    
    def __getitem__(self,index):
        """读取npy中存储的图像"""
        dbz_x,dbz_y = self.get_dataset_data(index,'dBZ')
        water_x,water_y = self.get_dataset_data(index,'WATER')
        zdr_x,zdr_y = self.get_dataset_data(index,'ZDR')
        x = np.concatenate((water_x,dbz_x,zdr_x),axis=-1)
        y = np.concatenate((water_y,dbz_y,zdr_y),axis=-1)
        return x,y
    
    

    def __len__(self):
        return len(self.start_dict['dBZ'])

def make_dataloader(configs):
    sina_dataset = SinaDataset(configs)
    data_loader = DataLoader(sina_dataset,batch_size=configs['batch_size'],
                            shuffle=False,num_workers=configs['cpu_worker'],drop_last=True)
    return data_loader
        