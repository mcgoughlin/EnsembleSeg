# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 09:35:14 2022

@author: mcgoug01
"""

import torch
import torch.nn as nn
from os import *
from os.path import *
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import sys

class EnsembleClassifier(nn.Module):
    def __init__(self,filepath:list = [],weights:list=[],device_list:list = []):
        super().__init__()
            

        if len(weights)>0: assert(len(weights)==len(filepath))
        else: weights = [1]*len(filepath)
        
        self.weights = torch.Tensor(weights)*(len(weights)/sum(weights))
        
        if torch.cuda.is_available():
            dev = 'cuda'
        else:
            dev = 'cpu'
            
        self.model_list = [torch.load(path,map_location=dev) for path in filepath]
        
        if len(device_list)>0:
            for i in range(len(self.model_list)):
                self.model_list[i].to(device_list[i])
                
        self.soft = nn.Softmax(dim=1)
        self.eval()
        
    def forward(self,x,boundary=0.75):
        #0.75 is roughly optimum DSC for non-kidney differentiation
        ans = None
        for model,weight in zip(self.model_list,self.weights):
            p = model(x)
            sp = self.soft(p)*(weight/len(self.weights))
            if ans is None: ans = sp
            else:ans+=sp
            
            del(model,p,sp)
            torch.cuda.empty_cache()
            gc.collect()
        ret = []
        for i in range(len(ans)):
            ret.append((ans[i,1]>boundary).cpu())

        return ret
        

if __name__ == "__main__":
    home = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\classification\\NEW 10-fold_results\\final_kidney_differentiation"
    folders = listdir(home)
    mod = "highest_DSC_model"
    instance_norm = nn.InstanceNorm2d(5)
    # results_highest-02-noncontrast-CB.csv
    
    m_list = []
    w_list = []
    for fold in folders:
        m_list.append(join(home,fold,mod))
        
        res_file = [file for file in listdir(join(home,fold)) if "results" in file][0]
        df = pd.read_csv(join(home,fold,res_file))
        sens = len(df[(df.correct == 1) & (df.label==1)])/len(df.label==1)
        spec =  len(df[(df.correct == 1) & (df.label==0)])/len(df.label==0)
        w_list.append(sens*spec/(spec + sens))
    
    model = EnsembleClassifier(m_list,w_list,['cuda']*10)
    
    
    im_path = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\classification\\test3Dcv_nc\\10-fold_data.csv"
    meta = pd.read_csv(im_path)
    num_samples_perclass = 500
    
    data = []
    # for psuedo_boundary in np.linspace(0.05,0.95,19):
    #     sens = 0
    #     spec = 0
    #     print("\nPseudoboundary: {:.2f}".format(psuedo_boundary))
    
    rand_n = meta[meta.label==0].sample(num_samples_perclass).filename.values
    # rand_k = np.concatenate([meta[(meta.label==1)].sample(num_samples_perclass//2).filename.values, meta[(meta.label==2)].sample(num_samples_perclass//2).filename.values])
    rand_k = np.concatenate([meta[(meta.label==2)].sample(num_samples_perclass).filename.values])
    i  =  np.random.randint(0,len(rand_n))
    # for i in range(len(rand_n)):
    name_n = rand_n[i]
    name_k = rand_k[i]
    
    rand_non = torch.Tensor(
         np.expand_dims(
                 np.load(name_n + '.npy'),axis=0))
     
    rand_kid = torch.Tensor(
         np.expand_dims(
                 np.load(name_k + '.npy'),axis=0))
    
    rand = torch.vstack([rand_non,rand_kid]).to('cuda')
    rand = instance_norm(torch.clip(rand,-100,300))
     
    ex = model(rand)
    print(ex)
    #     if ex[0,1]<psuedo_boundary: spec+=1
    #     if ex[1,1]>=psuedo_boundary: sens+=1
         
    #     del(ex,rand,rand_kid,rand_non)
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     i +=1
    #     print_str = "Number of Samples: {}\tSensitivity: {:.2f}%\tSpecificity: {:.2f}%".format(i*2,100*sens/i,100*spec/i)
        
    #     sys.stdout.write("\r{0}".format(print_str))
    #     sys.stdout.flush()
    #     sleep(0.001)
    
    # data.append({'boundary':psuedo_boundary,
    #              'sensitivity':100*sens/i,
    #              'sepcificity':100*spec/i})
    # print("")
        
    
    #for displaying the CT scans accurately
    for i in range(len(rand)): 
        rand[i] -= rand[i].min()
        rand[i] /= rand[i].max()
        
    plt.subplot(121)
    plt.imshow(np.rot90(rand[0][1:4].T.cpu(),3))
    plt.subplot(122)
    plt.imshow(np.rot90(rand[1][1:4].T.cpu(),3))
    del(model,ex,rand)
    torch.cuda.empty_cache()
    gc.collect()