# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:23:34 2022

@author: mcgoug01
"""
import os 
from os.path import *
import pandas as pd
import numpy as np
from codebase.dataset_generator import create_from_case
import csv
import shutil
path = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\classification"
cases = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\manifest-1592488683281\\manifest-1592488683281\\file.csv"
def setup_cv_folders(path = path, folds=5,arterial=True):
    if arterial: home = join(path,'test3Dcv_art')
    else: home = join(path,'test3Dcv_nc')
    if not exists(home):
        os.mkdir(home)
    if not exists(join(home,"data")):
        os.mkdir(join(home,"data"))
    # os.chdir(home)
    # for i in range(folds):
    #     fold = "fold_{:02d}".format(i)
    #     if not exists(fold):
    #         os.mkdir(fold)
    #         setup_fold(join(home,fold))
            

def setup_fold(path = None):
    train = join(path,"train")
    valid = join(path,"valid")
    
    folders = [train,valid]
    
    for folder in folders:
        print(folder)
        if not exists(folder):
            os.mkdir(folder)

            
def split(list_, folds, tr_split = 0.9):
    np.random.shuffle(list_)
    k, m = divmod(len(list_), folds)
    fold_cases = [list_[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(folds)]
    split_cases = []
    
    for fold in fold_cases:
        split_cases.append(fold)
        # train = int(len(fold)*tr_split)
        # valid = len(fold)-train
        # split = []
        # # split.append(np.sort(fold[0:train]))
        
        # split.append(np.sort(fold[train:]))
        # split_cases.append(split)
        
    return split_cases
    

def data_generation(fold_cases,cv_path = path, seg_path = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\ovseg_kits19data\\",
                    folds = 5,arterial=True,rand_pc=50):
    
    if arterial: home = join(cv_path,'test3Dcv_art')
    else: home = join(cv_path,'test3Dcv_nc')
    full_path = join(home,"data")
    
    if exists(join(home,'metadata.csv'.format(folds))): 
        flag = True
    else:
        flag = False
    


    with open(join(home,'metadata.csv'.format(folds)), 'a', newline='') as file:
        writer = csv.writer(file)
        if not flag:
            writer.writerow(["fold","case", "filename", "label", "number of kidney vox", "number of mass vox","image spacing (by index)","z spacing between slices"])
        for fold in range(folds):
            
            split_list = fold_cases[fold]
            for case in split_list:

                print("Generating Tiles for case {}...".format(case))
                # dataset = split_cases[0]
                # full_path = join(home,dataset)
                # for case in split_cases[1]:
                for label,kvox,tvox,sv_im,ni,z_spacing in create_from_case(int(case),full_path,seg_path,arterial=arterial,rand_pc=rand_pc):
                    writer.writerow([fold, case, sv_im,label,kvox,tvox,ni,z_spacing])
                print("")
                            
if __name__ == "__main__":
    folds = [10]
    case_path = "C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\ovseg_kits19data\\raw_data\\kits19_nc\\imagesTr"
    c = [int(file.split('-')[1][:5])for file in os.listdir(case_path)]
    
    arterials = [False]
    for fold in folds:
        print(fold)
        fold_cases = split(c,fold,tr_split=1)
        for arterial in arterials:
            print(arterial)
            setup_cv_folders(folds=fold,arterial=arterial)
            data_generation(fold_cases,folds=fold,arterial=arterial)
