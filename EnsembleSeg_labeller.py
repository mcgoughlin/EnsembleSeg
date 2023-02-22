# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:12:09 2022

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
import gc
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    print('No tqdm found, using no pretty progressing bars')
    tqdm = lambda x: x


from codebase.Ensemble import EnsembleClassifier as QC
from codebase.Ensemble_malig import EnsembleClassifier as Detect
from codebase.running_inferseg import get_3d_UNet as SegLoader
from codebase.running_inferseg import SlidingWindowPrediction as Segment
from codebase.running_inferseg import SegmentationPostprocessing as SegProcess
from codebase.setup_crossfold_validation import setup_cv_folders as TileFolderSetup
from codebase.setup_crossfold_validation import data_generation as TileGenerator

from ovseg.preprocessing.SegmentationPreprocessing import SegmentationPreprocessing
from ovseg.data.SegmentationData import SegmentationData
from ovseg.utils.io import save_nii_from_data_tpl, save_npy_from_data_tpl, load_pkl, read_nii, save_dcmrt_from_data_tpl, is_dcm_path
from ovseg.utils.torch_np_utils import maybe_add_channel_dim

class Early_Detection(nn.Module):
    def __init__(self,case_path:str=None, ##seg preprocess args
                 low_seg_mp:str=None,high_seg_mp:str=None,spacing = np.array([3,3,3]),
                 segpred_path:str=None,seg_dev:str=None, ##seg args
                 tile_path:str=None, ##tile args
                 QC_model_path:str = None, QC_dev:str=None, ##QC args
                 Detect_model_path:str=None, Detect_dev:str=None  ##Detect Args
                 ):
        super().__init__()
        
        print("")
        print("Initialising Early Detection System...")
        print("")
        torch.cuda.empty_cache()
        gc.collect()
        self.cases = [[int(file.split('-')[1][:5])for file in listdir(case_path) if 'KiTS' in file]]
        self.segpredpath =segpred_path
        
        
        ### SEG PREPROCESS ###
        self.seg_dev = seg_dev
        self.spacing =spacing
        self.prep_name = str(spacing[0])+','+str(spacing[1])+','+str(spacing[2])+"mm"
        self.SegModel = None
        self.Segmentation_Preparation(self.spacing)
        
        ### SEG ####
        print("Conducting Segmentation.")
        self.seg_mp_low = low_seg_mp
        self.Segment_CT()
        print("Segmentation complete!")
        print("")
        torch.cuda.empty_cache()
        gc.collect()
        
        
    def Segmentation_Preparation(self,seg_spacing,
                                 data_name = 'Addenbrookes',force_overwrite=False):
        
        preprocessed_name = str(seg_spacing[0])+','+str(seg_spacing[1])+','+str(seg_spacing[2])+"mm"
        environ['OV_DATA_BASE'] = self.segpredpath
        environ['OV_DATA_BASE'] = self.segpredpath
        
        self.prepname = preprocessed_name
        self.data_name = data_name
        pp_save_path = join(self.segpredpath,"preprocessed",self.data_name,self.prepname,'images')
        
        rawdata_path = join(self.segpredpath,"raw_data",self.data_name,'images')

        if exists(pp_save_path):
            if ([file[:len('Rcc_002')] for file in listdir(pp_save_path)] == [file[:len('Rcc_002')] for file in listdir(rawdata_path)]) and not force_overwrite: 
                print("Preprocessing appears to be complete already. Found the following preprocessed scans:",
                      [splitext(file)[0] for file in listdir(pp_save_path)])
                print("")
                return
        
        print("##SEG PREPROCESS##\nPreprocessing CT Volumes to {}\n Stored in location {}.".format(seg_spacing,pp_save_path))
        print("")
        preprocessing = SegmentationPreprocessing(apply_resizing=True,
                                                  apply_pooling=False,
                                                  apply_windowing=True,
                                                  target_spacing=seg_spacing,
                                                  pooling_stride=None,
                                                  window=np.array([-116., 130.]),
                                                  scaling=np.array([41.301857, 12.257426]),
                                                  lb_classes=None,
                                                  reduce_lb_to_single_class=True,
                                                  lb_min_vol=None,
                                                  prev_stages=[],
                                                  save_only_fg_scans=False,
                                                  n_im_channels = 1)

        preprocessing.preprocess_raw_data(raw_data=data_name,
                                          preprocessed_name=preprocessed_name,
                                          data_name=None,
                                          save_as_fp16=True,
                                          force_rewrite = force_overwrite)
        print("")
        
        
    def _load_UNet(self, path = None,
                   dev=None):
        model_files = [file for file in listdir(path) if "fold_" in file]
        
        for foldpath in model_files:
            self.SegModel = SegLoader(1, 2, 6, 2, filters=32)
            b = 64
            sm = torch.load(join(path,foldpath,"network_weights"))
            self.SegModel.load_state_dict(sm)
            self.SegModel.to(self.seg_dev)
            self.SegModel.eval()
                
            self.Segment.append(Segment(self.SegModel,[b,b,b],batch_size=1,overlap=0.5))

        
    def seg_pred(self, data_tpl, res='low',do_postprocessing=True):

        import matplotlib.pyplot as plt
        im = data_tpl['image']
        im = maybe_add_channel_dim(im)
        bin_pred = None
        # now the importat part: the sliding window evaluation (or derivatives of it)
        pred_holder = None
        models = self.Segment[:5]
        
        im_interp = None
        lo_boundary = 20
        scaled_boundary = (lo_boundary - 12.257426) / 41.301857

        for model in models:
            pred = model(im)
            if im_interp == None:
                im_interp = nn.functional.interpolate(torch.unsqueeze(torch.Tensor(im),axis=0),size = pred.shape).numpy()[0,0]
            pred = np.where(im_interp>scaled_boundary,pred,0)
            data_tpl['pred'] = pred

            # inside the postprocessing the result will be attached to the data_tpl
            if do_postprocessing:
                self.SegProcess.postprocess_data_tpl(data_tpl, 'pred', bin_pred)
            
            if type(pred_holder) == type(None):
                pred_holder = data_tpl['pred_orig_shape']
            else:
                pred_holder += data_tpl['pred_orig_shape']

        pred_holder =  np.round(pred_holder/5)
        # print(pred_holder.shape)

        return pred_holder
    
    
    def save_prediction(self, data_tpl, folder_name, filename=None):

        # find name of the file
        if filename is None:
            filename = data_tpl['scan'] + '.nii.gz'
        else:
            # remove fileextension e.g. .nii.gz
            filename = filename.split('.')[0] + '.nii.gz'

        # all predictions are stored in the designated 'predictions' folder in the OV_DATA_BASE
        if not exists(self.seg_save_loc):
            makedirs(self.seg_save_loc)

        key = 'pred'
        if 'pred_orig_shape' in data_tpl:
            key += '_orig_shape'
        save_nii_from_data_tpl(data_tpl, join(self.seg_save_loc, filename), key)
        
        
        
        
    def Segment_CT(self, im_path:str =  None, save_path:str= None,
                   volume_thresholds = [250],force_rewrite=True):
        ##to convert the save process from .npy to .nii.gz we need to do the following:
        ## use from ovseg.utils.io import save_nii_from_data_tpl, which requires data to be in data_tpl form not volume form. This requires:
        ## we use Dataset dataloader to feed data to unet, as this generates the data_tpl for each scan. Effectively, we will be setting up Simstudy data as a validation
        #dataset. This requires that we provide: preprocessed data loc, scans, and 'keys' - I do not know what the keys are (in Dataset)
        self.Segment = []
        self._load_UNet(self.seg_mp_low,self.seg_dev)
        # self._load_UNet(self.seg_mp_high,self.seg_dev,res='high')
        
        self.preprocess_path = join(self.segpredpath,"preprocessed",self.data_name,self.prepname,"images")
        self.seg_save_loc = join(self.segpredpath,"predictions")
        
        self.SegProcess = SegProcess(apply_small_component_removing= True,lb_classes=[1],
                                                  volume_thresholds = volume_thresholds,
                                                  remove_comps_by_volume=True,
                                                  use_fill_holes_3d = False)
        self.segmodel_parameters_low = np.load(join(self.seg_mp_low,"model_parameters.pkl"),allow_pickle=True)
        
        params_low = self.segmodel_parameters_low['data'].copy()
        params_low['folders'] = ['images']
        params_low['keys'] = ['image']

        self.segpp_data = SegmentationData(preprocessed_path=split(self.preprocess_path)[0],
                                     augmentation= None,
                                     **params_low)

        
        ppscans = [self.segpp_data.val_ds[i]['scan'] for i in range(len(self.segpp_data.val_ds))]
        if exists(self.seg_save_loc):
            existing_segmentations = [scan[:len('KiTS-00000')] for scan in listdir(self.seg_save_loc)]
        else:
            existing_segmentations = []
        
        
        if (ppscans == existing_segmentations) and not force_rewrite: 
            print("Segmentation appears to be complete already. Found the following segmented scans:",
                  [file for file in listdir(self.seg_save_loc)])
            print("")
            return
        
        for i in range(len(self.segpp_data.val_ds)):
            
            # get the data
            data_tpl = self.segpp_data.val_ds[i]
            filename = data_tpl['scan'] + '.nii.gz'
            print("Segmenting {}...".format(data_tpl['scan']))
            
            if not force_rewrite:
                if exists(join(self.seg_save_loc, filename)):
                    print("Already Segmented!\n")
                    continue
            
            # first let's try to find the name
            if 'scan' in data_tpl.keys():
                scan = data_tpl['scan']
            else:
                d = str(int(np.ceil(np.log10(len(self.segpp_data)))))
                scan = 'case_%0'+d+'d'
                scan = scan % i

            # predict from this datapoint
            pred = self.seg_pred(data_tpl,res='low')
            # if pred.max() <=1:
            #     print("low res failed, trying high res")
            #     data_tpl = self.segpp_data_high.val_ds[i]
            #     pred = self.seg_pred(data_tpl,res='high')
                
            if torch.is_tensor(pred):
                pred = pred.cpu().numpy()
            
            data_tpl['pred_orig_shape'] = pred
            self.save_prediction(data_tpl, folder_name='SimStudy', filename=scan)
            print("")

        print("Segmentations complete!\n")
        
   
if __name__ == "__main__":
    case_path = "D:\\Data\\AddenbrookesRCC\\CECT\\images"
    # trained_model_path = "D:\\Data\\AddenbrookesRCC\\CECT\\3mm\\6,3,32"
    # low_seg_mp = "D:\\Data\\AddenbrookesRCC\\CECT\\3mm\\6,3,32"
    trained_model_path = "D:\\Data\\AddenbrookesRCC\\CECT\\2mm\\6,3,32"
    low_seg_mp = "D:\\Data\\AddenbrookesRCC\\CECT\\2mm\\6,3,32"
    high_seg_mp = None
    
    QC_mp = None
    
    Detect_mp = None
    
    test = Early_Detection(case_path,
                          low_seg_mp = low_seg_mp,high_seg_mp=high_seg_mp,
                          segpred_path="D:\\Data\\AddenbrookesRCC\\CECT",seg_dev='cuda',
                          tile_path="D:\\SimStudy_v3_testset\\tile_path",
                          QC_model_path = QC_mp, QC_dev='cuda',
                          Detect_model_path = Detect_mp, Detect_dev='cuda')
        
        