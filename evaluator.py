import os
import datetime
import numpy as np
import torch

import matplotlib.pyplot as plt



def test(model,dataloader,configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , 'test...')
    res_path = os.path.join(configs['gen_frm_dir'], str(itr))
    os.mkdir(res_path)
    
    
def save_plots(data,epoch_id,batch_id, res_path, figsize=None, cmap="viridis", npy=False, **imshow_args):
    data = data.detach().cpu()
    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    ax.set_axis_off()
    # alpha = data
    # alpha[alpha < 1] = 0
    # alpha[alpha > 1] = 1
    img = ax.imshow(data,  cmap=cmap, **imshow_args)
    plt.savefig('{}/{}.png'.format(res_path, 'epoch'+str(epoch_id) + '_batch_id' + str(batch_id)))
    plt.close()  