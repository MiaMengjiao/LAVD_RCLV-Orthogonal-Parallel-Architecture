# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:58:34 2018

@author: Mengjiao Wang
"""

import json
import gdal
import glob
import pdb
import numpy as np
import math
from scipy import ndimage
import os
from multiprocessing import Pool,Process
import i4_eddy_detect_core
import scipy.io as sio  # read .mat
import time
import gc

def load_mat(load_fn,key_name):
    load_data = sio.loadmat(load_fn)
    load_matrix = load_data[key_name] 
    return load_matrix

def time_ID_loc(h_indirs,time_ID,origin_data):
    '''
    Determine the position of timeID
    
    '''
    h_indirs = glob.glob(origin_data['input_dir']['h']+'*'+time_ID+'*.mat')
    
    if len(h_indirs)>1:
        print('time_ID_loc error')
    else:
        file = h_indirs[0]
        a = file.find(time_ID)
        name_len = len(time_ID)
    
        return a,a+name_len

def filter_right_file(uv_indirs,time_ID):
    '''
    Matching lavd files
    '''
    if len(uv_indirs) == 1:
        return uv_indirs[0]
    else:

        d=[]
        for file in uv_indirs:
            d.append(file.count(time_ID)) 
        d_max = max(d) 
        if d.count(d_max) > 1: 
            print('Matching error')
        else:
            return uv_indirs[d.index(d_max)]
            
def caculate_pixel(indir,left_top_right_bottom):
    
    h_file = glob.glob(indir+'*.mat')[0]
    sladata = load_mat(h_file,'VMatrix') 
    x = max(np.shape(sladata))
    y = min(np.shape(sladata))
    pixelwidth = abs(left_top_right_bottom[2]-left_top_right_bottom[0])/x
    pixelheight = abs(left_top_right_bottom[3]-left_top_right_bottom[1])/y

    return pixelwidth,pixelheight

def caculate_block(origin_data):
    '''
    Divide the global LAVD field into 5×9 regions, 
    in which the width of the inner block is 30°; the width of the outer block is 10°
    '''
    overlapping_area_width = 0
    overlapping_area_height = 0
    begin_lon = origin_data['study_area'][0]
    end_lon = origin_data['study_area'][1]    
    begin_lat = origin_data['study_area'][2]
    end_lat = origin_data['study_area'][3]
    lon_div = abs(end_lon-begin_lon)
    lat_div = abs(end_lat-begin_lat)
    
    inner_lon = lon_div
    inner_lat = lat_div

    if origin_data['if_block_auto']:
        overlapping_area_height=10
        overlapping_area_width=10
                
        inner_lat=30            
        lat_block_num=int((lat_div+overlapping_area_height)/(inner_lat+overlapping_area_height) )
        if (lat_div-lat_block_num*(inner_lat+overlapping_area_height))>10:
            lat_block_num+=1
        if lat_block_num==1:
           overlapping_area_height=0
           
        inner_lon=30        
        if lon_div>=359.875:
            lon_div+=overlapping_area_width
            end_lon+=overlapping_area_width            
        lon_block_num=int(((lon_div+overlapping_area_width)/(inner_lon+overlapping_area_width) ))
        if (lon_div-lon_block_num*(inner_lon+overlapping_area_width))>10:
            lon_block_num+=1
        if lon_block_num==1:
           overlapping_area_width=0
       
      
    elif origin_data['if_block']:

        overlapping_area_width=origin_data['overlapping_area'][0]
        overlapping_area_height=origin_data['overlapping_area'][1]
        
        if abs(begin_lon-end_lon)>=359.75:                                    # global
            end_lon+=overlapping_area_width
            lon_div=abs(end_lon-begin_lon)+overlapping_area_width
        lat_div+=overlapping_area_height
        
        inner_lon=origin_data['inner_lon_lat'][0]
        inner_lat=origin_data['inner_lon_lat'][1]       
        lon_block_num=int(np.ceil(lon_div/(inner_lon+overlapping_area_width)))-1
        lat_block_num=int(np.ceil(lat_div/(inner_lat+overlapping_area_height)))
        
        
    else:
        lon_block_num=1
        lat_block_num=1
        
    block={}
    lon_block_list=[]
    lon1=begin_lon 
    lon2=begin_lon
    
    for i in range(lon_block_num):
        lon=[]
        
        if i==(lon_block_num-1):
            lon1=lon2
            lon.append(lon1)
            lon.append(end_lon)
        elif i==0:
            lon2=lon1+overlapping_area_width+inner_lon
            lon.append(lon1)
            lon.append(lon2+overlapping_area_width)            
        else:
            lon1=lon2
            lon2=lon2+overlapping_area_width+inner_lon
            lon.append(lon1)
            lon.append(lon2+overlapping_area_width)
            
        lon_block_list.append(lon)
        lat_block_list=[]
        
        lat1=begin_lat
        lat2=begin_lat
            
        for j in range(lat_block_num):
            
            lat=[]
            
            if j==(lat_block_num-1):
                lat1=lat2
                lat.append(lat1)
                lat.append(end_lat)
            elif j==0:
                lat2=lat1+inner_lat
                lat.append(lat1)
                lat.append(lat2+overlapping_area_height)    
            else:
                lat1=lat2
                lat2=lat2+overlapping_area_height+inner_lat
                lat.append(lat1)
                lat.append(lat2+overlapping_area_height)
            
            keyname=str(i)+'_'+str(j)
            block[keyname]=lon+lat
            lat_block_list.append(lat)

    return [overlapping_area_width,overlapping_area_height],block,lon_block_list,lat_block_list
 
    
def data_filter(data_type,sladata,pixelwidth,if_global,filter_message,fillvalue,overlapping_area_width):

    h_data = load_mat(sladata,'VMatrix')
    maxx = np.max(h_data)
    minn = np.min(h_data)
    print('data max & min:',maxx,minn)
    
    h_invalid = load_mat(sladata,'Invalid')
    sladata_copy = np.ma.copy(h_data)
    invdata_copy = np.ma.copy(h_invalid)

    if if_global: 
        exstra_range=int(overlapping_area_width/pixelwidth)
        '''
        Extension of 10° [340,370]
        '''    
        h_data = np.hstack((h_data,h_data[:,0:exstra_range]))
        h_invalid = np.hstack((h_invalid,h_invalid[:,0:exstra_range]))
        end_exstra_sladata = sladata_copy[:,len(sladata_copy[0])-1-exstra_range:len(sladata_copy[0])-1]
        end_exstra_invdata = invdata_copy[:,len(invdata_copy[0])-1-exstra_range:len(invdata_copy[0])-1]
        h_data = np.hstack((end_exstra_sladata,h_data))
        h_invalid = np.hstack((end_exstra_invdata,h_invalid))
    
    if data_type == 'h':
        sla_mask = np.logical_not(h_invalid==0.0) 
    else:
        sla_mask = np.logical_or(h_data<-500,h_data>500)

    sladatamask=np.ma.array(h_data,mask=sla_mask)                             # mask
    if data_type == 'h':
        np.place(h_data, sladatamask.mask == True, 0.)
        h_data = np.ma.masked_where(sladatamask == False, h_data)    
    if if_global:
        h_data = h_data[:,exstra_range:len(h_data[0]
        )]
        
    return h_data,maxx,minn
    
    
def set_basic_message(origin_data):

    time_ID_begin,time_id_end = time_ID_loc(origin_data['input_dir']['h'],origin_data['time_ID'],origin_data)
    origin_data['time_index'] = [time_ID_begin,time_id_end]    
    origin_data['pixelwidth'],origin_data['pixelheight'] = caculate_pixel(origin_data['input_dir']['h'],origin_data['left_top_right_bottom'])
    pixelwidth = origin_data['pixelwidth']
    pixelheight = origin_data['pixelheight']

    if abs(origin_data['study_area'][1]-origin_data['study_area'][0]) >= 359.75:
        origin_data['if_global'] = True
    else:
        origin_data['if_global'] = False

    origin_data['overlapping_area_defined'],origin_data['block'],origin_data['lon_block_list'],origin_data['lat_block_list'] = caculate_block(origin_data)
    overlapping_area_width = origin_data['overlapping_area_defined'][0]

    origin_data['lat_grid'] = np.arange(origin_data['left_top_right_bottom'][1]+pixelwidth/2,origin_data['left_top_right_bottom'][3],pixelheight)
    if origin_data['if_global']:
        origin_data['lon_grid'] = np.arange(origin_data['left_top_right_bottom'][0]+pixelwidth/2,origin_data['left_top_right_bottom'][2]+overlapping_area_width,pixelwidth)
    else:
        origin_data['lon_grid'] = np.arange(origin_data['left_top_right_bottom'][0]+pixelwidth/2,origin_data['left_top_right_bottom'][2],pixelwidth)
        
    return origin_data

def read_h_uv(h_file,origin_data):

    sla_data, maxx, minn = data_filter( 'h',h_file,origin_data['pixelwidth'],origin_data['if_global'],origin_data['filter'],origin_data['fillvalue']['h'],origin_data['overlapping_area_defined'][0])  
    time_ID = h_file[origin_data['time_index'][0]:origin_data['time_index'][1]]
    print(time_ID)
      
    if origin_data['only_lavd']:
        udata = 0
        vdata = 0
    else:
        print("Matching error")
    return sla_data,udata,vdata,time_ID

def detect_eddy_main(h_file1,origin_data1):
    
    h_data,u_data,v_data,time_ID=read_h_uv(h_file1,origin_data1)    
    print('start load LAVD!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('start origin_data!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    seedjson_data = i4_eddy_detect_core.seedjson(origin_data1,h_data,time_ID)    

    block_list = origin_data1['block']

    lon_block_list = origin_data1['lon_block_list']
    lat_block_list = origin_data1['lat_block_list']
    pdb.set_trace()
    block_list={8_0:block_list['8_0']}
    
    print('start pool!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    if_parallel = origin_data1['if_parallel']
    
    if if_parallel:                                                           # parallel
        parallel_num = origin_data1['parallel_num']
        pool=Pool(processes = parallel_num)
        for block_i in list(block_list.keys()):
            print(block_i)
            pool.apply_async(i4_eddy_detect_core.I4_eddy_detect_core,(origin_data1,h_data,u_data,v_data,time_ID,block_i))
            gc.collect()
        pool.close()
        pool.join()
    else:                   
        for block_i in list(block_list.keys()):
            # pdb.set_trace()
            print(block_i)
            i4_eddy_detect_core.I4_eddy_detect_core(origin_data1,h_data,u_data,v_data,time_ID,block_i)    
    print('start merge!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    mergejson = i4_eddy_detect_core.eddyjson_merge(origin_data1,lon_block_list,lat_block_list,time_ID)
    i4_eddy_detect_core.reject(origin_data1,seedjson_data,mergejson)    
    print('finish!',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def RCLV_CPU(origin_data,h_file):
    print('################### start RCLV_CPU ###################')
    origin_data = set_basic_message(origin_data)
    detect_eddy_main(h_file, origin_data)