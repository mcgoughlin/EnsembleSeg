# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:23:52 2022

@author: mcgoug01
"""
import nibabel as nib
from os.path import *
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import random
import gc
import csv
import math


def get(cases_folder="C:\\Users\\mcgoug01\\OneDrive - CRUK Cambridge Institute\\Python Scripts\\kits19\\ovseg_kits19data",
        case = None, arterial = True):
    
    case = "KiTS-{:05d}".format(case)
    
    case_path = join(cases_folder,"raw_data","kits19_nc","images",case+".nii.gz")
    seg_path = join(cases_folder,"predictions","kits19_nc","binary_kits19","SimStudy",case+".nii.gz")
    fp_path = join(cases_folder,"preprocessed\\kits19_nc\\binary_kits19\\fingerprints",case+".npy")

    im = nib.load(case_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    fingerprint = dict(np.load(fp_path,allow_pickle=True).tolist())

    return im, seg,fingerprint

def refine_kidney(im,seg,fp_spacing,desired_z_spacing=5):
    num_indices = max(int(desired_z_spacing/fp_spacing),1)
    
    kidneys = np.any(np.any(seg>0,axis=1),axis=1)
    true_slices = np.array([i for i, x in enumerate(kidneys) if x])
    
    if true_slices.sum() == 0:
        bottom = 0
        top = len(im)
    else:
        top = max(true_slices)+ num_indices
        bottom = min(true_slices)- num_indices

    im,seg = im[bottom:top],seg[bottom:top]
    return im,seg,num_indices

def centre_to_window(centroid, window_size=224):
    c= [int(centroid[0]),int(centroid[1])]
    if c[0] < 112: c[0] = 112
    if c[0] > 400: c[0] = 400
    if c[1] < 112: c[1] = 112
    if c[1] > 400: c[1] = 400
    return [c[0]-112,c[0]+112,c[1]-112,c[1]+112]
    
def get_windows(im):
     bone = np.where((im>400) & (im<1000),1,0).astype(np.int8)
     centroid = regionprops(bone)[0].centroid[1:]
     left = (centroid[0],centroid[1]-128)
     right = (centroid[0],centroid[1]+128)
     coords = [centre_to_window(left),centre_to_window(right)]
    
     return coords

def apply_window(im,window):
    return im[window[0]:window[1],window[2]:window[3]]

def create_noise(std_dev=50,shape = (224,224)):
    return np.random.normal(scale=std_dev,size=shape)

def check_validity(seg):
    kidneys = sum(sum(sum(np.where(seg>0,1,0))))
    mass = sum(sum(sum(np.where(seg>1,1,0))))
    
    if kidneys<1024: return 0,kidneys,mass
    #https://jamanetwork.com/journals/jama/fullarticle/2747673#:~:text=A%20small%20kidney%20tumor%20is,the%20size%20of%20a%20fist.
    #small is less than 4cm
    #lets have threshold of 1.5cm
    # circle has area of 0.00070685834 - this is radius 1.5cm!!! oh no...
    # no this is fine.. represent this as an easy task.
    
    #kits 19 median voxel size 1.62 × 1.62mm
    #0.0000027556 area per pixel in slice
    if mass > 1024: return 2,kidneys,mass
    else: return 1,kidneys,mass
    
    return k,m,kidneys,mass

def get_random(im,seg,idx_list,num_indices):
    z,x,y = im.shape
    z_rand = np.random.randint(2*num_indices,z-2*num_indices)
    im_slice,seg_slice = im[z_rand], seg[z_rand]
    window = centre_to_window((np.random.randint(0,y),np.random.randint(0,z)))
    img = np.expand_dims(apply_window(im_slice,window),axis=0)
    segm = np.expand_dims(apply_window(seg_slice,window),axis=0)
    for idx in idx_list:
        img = np.vstack([img,np.expand_dims(apply_window(im[z_rand+idx],window),axis=0)])
        segm = np.vstack([segm,np.expand_dims(apply_window(seg[z_rand+idx],window),axis=0)])
            
    return img,segm

def create_from_case(case_num, path,seg_path, arterial=False,rand_pc = 50):
    im,seg,fp = get(cases_folder = seg_path, case=case_num,arterial=arterial)
    
    z_spacing = fp['orig_spacing'][0]
    im_ref, seg_ref, num_indices = refine_kidney(im,seg,z_spacing)
    
    if im_ref.shape[0]*z_spacing<25: ##assumption:  if kidneys are less than 25mm in z plane then we have a badly wrong kidney segmentation
        im_ref, seg_ref = im,seg
    windows = get_windows(im_ref)
    idx_list = [ -num_indices,num_indices,-2*num_indices, 2*num_indices]
    sorting_index = [3,1,0,2,4]
    
    for i in range(rand_pc):
        sv_im = join(path,"KiTS-{:05d}-{}-{:02d}".format(case_num,'random',i))
        img,segm = get_random(im,seg,idx_list,num_indices)
        output,kvox,tvox = check_validity(segm)
        np.save(sv_im,img)
        yield output,kvox,tvox,sv_im,num_indices,z_spacing
        
    for slice,(axial_im,axial_seg) in enumerate(zip(im_ref[2*num_indices:-2*num_indices],seg_ref[num_indices:-num_indices])):
        for i,window in enumerate(windows):
            i*=2
            sv_im = join(path,"KiTS-{:05d}-{:03d}-{:01d}".format(case_num,slice,i))
            img = np.expand_dims(apply_window(im_ref[slice],window),axis=0)
            segm = np.expand_dims(apply_window(seg_ref[slice],window),axis=0)
            for idx in idx_list:
                img = np.vstack([img,np.expand_dims(apply_window(im_ref[slice+idx],window),axis=0)])
                segm = np.vstack([segm,np.expand_dims(apply_window(seg_ref[slice+idx],window),axis=0)])
            
            
            malignant,kvox,tvox = check_validity(segm[:3])
            np.save(sv_im,img)

            yield malignant,kvox,tvox,sv_im,num_indices,z_spacing
                      

            
            
    

    
