# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:59:57 2019

@author: Mengjiao Wang
"""
import json
import glob
import pdb
import os
import scipy.io as sio 
import numpy as np
import LAVD_SLA              # Algorithm 1
import RCLV_start            # Algorithm 2


if __name__ == '__main__':

    in_dir='./LAVD_parameters.json'
    with open(in_dir,'r') as f:
        origin_data=json.load(f)['input_parameters']['2D']      
    for year in range(1993,2019,1):
        for quater in range(1,5,1):
            print(year,quater)
            path = origin_data['data']['uv_path']+str(year)+'\\'+str(quater)+'\\'
            h_indirs = glob.glob(origin_data['data']['uv_path']+str(year)+'\\'+str(quater)+'\\'+'*.tif')  
            t0=int(origin_data['t0'])
            tp=int(origin_data['tp'])
 
            startfile = h_indirs[0]
            timeindex = len(origin_data['data']['uv_path'])+31
            timeID = startfile[timeindex:timeindex+8]
            filedir = origin_data['save_path']+'/'+str(timeID)+'/lavd/'
            filename = filedir+'lavd_glb_'+'%s'%timeID+'_%dday.mat'%tp
            if not os.path.exists(filedir):
                os.makedirs(filedir)
#☆=☆=☆=☆=☆=☆=☆=☆=☆=☆=☆ Algorithm 1 GPU-based parallel LAVD calculation
            LAVD, is_invalid = LAVD_SLA.LAVD_GPU(origin_data,tp,h_indirs)
            LAVD = np.array(LAVD)
            print('savefile:',filename)
            sio.savemat(filename, {'VMatrix': LAVD,'Invalid': is_invalid})
#☆=☆=☆=☆=☆=☆=☆=☆=☆=☆=☆ Algorithm 2 CPU-based parallel RCLV extraction 
            RCLV_in_dir='./RCLV_parameters.json'    
            with open(RCLV_in_dir,'r') as f:
                RCLV_origin_data=json.load(f)['out_info']['2D']
            RCLV_origin_data['time_ID']=timeID
            RCLV_origin_data['input_dir']['h']=filedir
            RCLV_start.RCLV_CPU(RCLV_origin_data,filename)

                
             