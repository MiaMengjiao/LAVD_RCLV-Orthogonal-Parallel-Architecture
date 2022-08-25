# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 15:00:09 2021

Algorithm 1 GPU-based parallel LAVD calculation
 
@author: Mengjiao Wang
"""

import os
import scipy.io as sio  
import operator
from numpy import *
from tkinter import _flatten
from scipy.stats import zscore
from scipy.integrate import odeint
from scipy.integrate import ode
from scipy import interpolate
import numpy as np 
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull # calculate convexhull
from scipy import signal
from osgeo import gdal
from scipy.interpolate import griddata
import os  
import time
import timeit
import pdb
import numba
from numba import cuda, jit, float64
@cuda.jit


def RK4_kernal_inner(Position_x,Position_y,C,jihe_xyz_u,jihe_xyz_v,mBlend,t_c,invalid,GPU_para): 
    R = 6370997.0 
    minlon = GPU_para[0]
    minlat = GPU_para[1]
    deltaT = GPU_para[2]
    tNum = int(GPU_para[3])
    latNum = int(GPU_para[4])
    lonNum = int(GPU_para[5])
    deltalat = GPU_para[6]
    deltalon = GPU_para[7]
    h = GPU_para[8]
    rho = GPU_para[9]
    
    # particles are assigned in parallel to each thread in the GPU based on their index 
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y  
    if i >= C.shape[0] or j >= C.shape[1]:
        return

    r = R*(math.cos(Position_y[i][j]*math.pi/180.0))                          # Latitude circle radius
    rati = 24*3600*180/(r*math.pi)

    pt_0 = Position_x[i][j]
    pt_1 = Position_y[i][j]
    tc = t_c[i][j]
        
    U = cuda.local.array(shape=4,dtype=numba.float64)
    V = cuda.local.array(shape=4,dtype=numba.float64)
    W = cuda.local.array(shape=4,dtype=numba.float64)
    P = cuda.local.array(shape=4,dtype=numba.float64)
    Q = cuda.local.array(shape=4,dtype=numba.float64)
    R1 = cuda.local.array(shape=4,dtype=numba.float64)
    Ui = cuda.local.array(shape=4,dtype=numba.float64)
    Vi = cuda.local.array(shape=4,dtype=numba.float64)
        
    MP1_0 = pt_0
    MP1_1 = pt_1  
            
    pt0 = MP1_0
    pt1 = MP1_1
    tq = tc
    iz= int(math.floor(tq/deltaT))
    tloc = (tq-iz*deltaT)/deltaT
    ix = int(math.floor((pt0 - minlon) / deltalon))
    xloc = (pt0 - minlon - ix * deltalon) / deltalon
    iy = int(math.floor((pt1 - minlat) / deltalat))
    yloc = (pt1 - minlat - (iy*deltalat)) / deltalat
    
    U[0] = 1.0
    U[1] = xloc
    U[2] = xloc*xloc
    U[3] = xloc*xloc*xloc
    V[0] = 1.0
    V[1] = yloc
    V[2] = yloc*yloc
    V[3] = yloc*yloc*yloc
    W[0] = 1.0
    W[1] = tloc
    W[2] = tloc*tloc
    W[3] = tloc*tloc*tloc
    for row in range(0,4,1):
        P[row] = 0.0
        Q[row] = 0.0
        R1[row] = 0.0
        for col in range(0,4,1):
            P[row] = P[row] + mBlend[row][col]*U[col]
            Q[row] = Q[row] + mBlend[row][col]*V[col]
            R1[row] = R1[row] + mBlend[row][col]*W[col]
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1
    result1 = 0.0
    result2 = 0.0
    for sli in range(0,4,1):
        zClamp = iz + sli
        if (zClamp < 0):
            zClamp = 0
        if (zClamp > tNum - 1):
            zClamp = tNum - 1
        for row in range(0,4,1):
            yClamp = iy + row
            if (yClamp < 0):
                yClamp = 0
            if (yClamp > latNum - 1):
                yClamp = latNum - 1
            for col in range (0,4,1):
                xClamp = ix + col
                if (xClamp < 0):
                    xClamp = 0
                if (xClamp > lonNum - 1):
                    xClamp = lonNum - 1
                result1 = result1+P[col]*Q[row]*R1[sli]*jihe_xyz_u[zClamp][xClamp][yClamp] 
                result2 = result2+P[col]*Q[row]*R1[sli]*jihe_xyz_v[zClamp][xClamp][yClamp]
    k1_0,k1_1 = result1,result2
#      
    MP2_0 = pt_0 + 0.5 * k1_0 * h *rati  #x
    MP2_1 = pt_1 + 0.5 * k1_1 * h *rati
    
    pt0 = MP2_0
    pt1 = MP2_1
    tq = tc + h/2
    iz=int(math.floor(tq/deltaT))
    tloc = (tq-iz*deltaT)/deltaT
    ix = int(math.floor((pt0 - minlon) / deltalon))
    xloc = (pt0 - minlon - ix * deltalon) / deltalon
    iy = int(math.floor((pt1 - minlat) / deltalat))
    yloc = (pt1 - minlat - iy * deltalat) / deltalat
    U[0] = 1.0
    U[1] = xloc
    U[2] = xloc*xloc
    U[3] = xloc*xloc*xloc
    V[0] = 1.0
    V[1] = yloc
    V[2] = yloc*yloc
    V[3] = yloc*yloc*yloc
    W[0] = 1.0
    W[1] = tloc
    W[2] = tloc*tloc
    W[3] = tloc*tloc*tloc
    for row in range(0,4,1):
        P[row] = 0.0
        Q[row] = 0.0
        R1[row] = 0.0
        for col in range(0,4,1):
            P[row] = P[row] + mBlend[row][col]*U[col]
            Q[row] = Q[row] + mBlend[row][col]*V[col]
            R1[row] = R1[row] + mBlend[row][col]*W[col]
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1
    result1 = 0.0
    result2 = 0.0
    for sli in range(0,4,1):
        zClamp = iz + sli
        if (zClamp < 0):
            zClamp = 0
        if (zClamp > tNum - 1):
            zClamp = tNum - 1
        for row in range(0,4,1):
            yClamp = iy + row
            if (yClamp < 0):
                yClamp = 0
            if (yClamp > latNum - 1):
                yClamp = latNum - 1
            for col in range (0,4,1):
                xClamp = ix + col
                if (xClamp < 0):
                    xClamp = 0
                if (xClamp > lonNum - 1):
                    xClamp = lonNum - 1
                result1 = result1+P[col]*Q[row]*R1[sli]*jihe_xyz_u[zClamp][xClamp][yClamp] 
                result2 = result2+P[col]*Q[row]*R1[sli]*jihe_xyz_v[zClamp][xClamp][yClamp]
    k2_0,k2_1 = result1,result2

    MP3_0 = pt_0 + 0.5 * k2_0 * h *rati  #x
    MP3_1 = pt_1 + 0.5 * k2_1 * h *rati
         
    pt0 = MP3_0
    pt1 = MP3_1
    tq = tc + h/2
    iz=int(math.floor(tq/deltaT))
    tloc = (tq-iz*deltaT)/deltaT
    ix = int(math.floor((pt0 - minlon) / deltalon))
    xloc = (pt0 - minlon - ix * deltalon) / deltalon
    iy = int(math.floor((pt1 - minlat) / deltalat))
    yloc = (pt1 - minlat - iy * deltalat) / deltalat
    U[0] = 1.0
    U[1] = xloc
    U[2] = xloc*xloc
    U[3] = xloc*xloc*xloc
    V[0] = 1.0
    V[1] = yloc
    V[2] = yloc*yloc
    V[3] = yloc*yloc*yloc
    W[0] = 1.0
    W[1] = tloc
    W[2] = tloc*tloc
    W[3] = tloc*tloc*tloc
    for row in range(0,4,1):
        P[row] = 0.0
        Q[row] = 0.0
        R1[row] = 0.0
        for col in range(0,4,1):
            P[row] = P[row] + mBlend[row][col]*U[col]
            Q[row] = Q[row] + mBlend[row][col]*V[col]
            R1[row] = R1[row] + mBlend[row][col]*W[col]
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1
    result1 = 0.0
    result2 = 0.0
    for sli in range(0,4,1):
        zClamp = iz + sli
        if (zClamp < 0):
            zClamp = 0
        if (zClamp > tNum - 1):
            zClamp = tNum - 1
        for row in range(0,4,1):
            yClamp = iy + row
            if (yClamp < 0):
                yClamp = 0
            if (yClamp > latNum - 1):
                yClamp = latNum - 1
            for col in range (0,4,1):
                xClamp = ix + col
                if (xClamp < 0):
                    xClamp = 0
                if (xClamp > lonNum - 1):
                    xClamp = lonNum - 1
                result1 = result1+P[col]*Q[row]*R1[sli]*jihe_xyz_u[zClamp][xClamp][yClamp]
                result2 = result2+P[col]*Q[row]*R1[sli]*jihe_xyz_v[zClamp][xClamp][yClamp]
    k3_0,k3_1 = result1,result2
     
    MP4_0 = pt_0 + k3_0 * h *rati  #x
    MP4_1 = pt_1 + k3_1 * h *rati
    
    pt0 = MP4_0
    pt1 = MP4_1
    tq = tc + h
    iz= int(math.floor(tq/deltaT))
    tloc = (tq-iz*deltaT)/deltaT
    ix = int(math.floor((pt0 - minlon) / deltalon))
    xloc = (pt0 - minlon - ix * deltalon) / deltalon
    iy = int(math.floor((pt1 - minlat) / deltalat))
    yloc = (pt1 - minlat - iy * deltalat) / deltalat
    U[0] = 1.0
    U[1] = xloc
    U[2] = xloc*xloc
    U[3] = xloc*xloc*xloc
    V[0] = 1.0
    V[1] = yloc
    V[2] = yloc*yloc
    V[3] = yloc*yloc*yloc
    W[0] = 1.0
    W[1] = tloc
    W[2] = tloc*tloc
    W[3] = tloc*tloc*tloc
    for row in range(0,4,1):
        P[row] = 0.0
        Q[row] = 0.0
        R1[row] = 0.0
        for col in range(0,4,1):
            P[row] = P[row]+mBlend[row][col]*U[col]
            Q[row] = Q[row]+mBlend[row][col]*V[col]
            R1[row] = R1[row]+mBlend[row][col]*W[col]
    ix = ix - 1
    iy = iy - 1
    iz = iz - 1
    result1 = 0.0
    result2 = 0.0
    for sli in range(0,4,1):
        zClamp = iz + sli
        if (zClamp < 0):
            zClamp = 0
        if (zClamp > tNum - 1):
            zClamp = tNum - 1
        for row in range(0,4,1):
            yClamp = iy + row
            if (yClamp < 0):
                yClamp = 0
            if (yClamp > latNum - 1):
                yClamp = latNum - 1
            for col in range (0,4,1):
                xClamp = ix + col
                if (xClamp < 0):
                    xClamp = 0
                if (xClamp > lonNum - 1):
                    xClamp = lonNum - 1
                result1 = result1+P[col]*Q[row]*R1[sli]*jihe_xyz_u[zClamp][xClamp][yClamp]
                result2 = result2+P[col]*Q[row]*R1[sli]*jihe_xyz_v[zClamp][xClamp][yClamp]

    k4_0,k4_1 = result1,result2

    pt_0 = pt_0 + h * (k1_0 + 2 * k2_0 + 2 * k3_0 + k4_0) / 6 *rati  #x
    pt_1 = pt_1 + h * (k1_1 + 2 * k2_1 + 2 * k3_1 + k4_1) / 6 *rati
    
    if (pt_0>360):
        pt_0=pt_0-360
    if (pt_0<0):
        pt_0=pt_0+360
    
    is_invalid=False
    for k in range(0,4,1):
        pt0 = pt_0 + rho*(math.cos(k*math.pi/2))
        pt1 = pt_1 + rho*(math.sin(k*math.pi/2))
        tq = tc+h
        iz=int(math.floor(tq/deltaT))
        tloc = (tq-iz*deltaT)/deltaT
        ix = int(math.floor((pt0 - minlon) / deltalon))
        xloc = (pt0 - minlon - ix * deltalon) / deltalon
        iy = int(math.floor((pt1 - minlat) / deltalat))
        yloc = (pt1 - minlat - iy * deltalat) / deltalat
        U[0] = 1.0
        U[1] = xloc
        U[2] = xloc*xloc
        U[3] = xloc*xloc*xloc
        V[0] = 1.0
        V[1] = yloc
        V[2] = yloc*yloc
        V[3] = yloc*yloc*yloc
        W[0] = 1.0
        W[1] = tloc
        W[2] = tloc*tloc
        W[3] = tloc*tloc*tloc
        for row in range(0,4,1):
            P[row] = 0.0
            Q[row] = 0.0
            R1[row] = 0.0
            for col in range(0,4,1):
                P[row] = P[row] + mBlend[row][col]*U[col]
                Q[row] = Q[row] + mBlend[row][col]*V[col]
                R1[row] = R1[row] + mBlend[row][col]*W[col]
        ix = ix - 1
        iy = iy - 1
        iz = iz - 1
        result1 = 0.0
        result2 = 0.0
        for sli in range(0,4,1):
            zClamp = iz + sli
            if (zClamp < 0):
                zClamp = 0
            if (zClamp > tNum - 1):
                zClamp = tNum - 1
            for row in range(0,4,1):
                yClamp = iy + row
                if (yClamp < 0):
                    yClamp = 0
                if (yClamp > latNum - 1):
                    yClamp = latNum - 1
                for col in range (0,4,1):
                    xClamp = ix + col
                    if (xClamp < 0):
                        xClamp = 0
                    if (xClamp > lonNum - 1):
                        xClamp = lonNum - 1
                    if (jihe_xyz_u[zClamp][xClamp][yClamp]==0 or jihe_xyz_v[zClamp][xClamp][yClamp]==0):
                        is_invalid=True
                    result1 = result1+P[col]*Q[row]*R1[sli]*jihe_xyz_u[zClamp][xClamp][yClamp]
                    result2 = result2+P[col]*Q[row]*R1[sli]*jihe_xyz_v[zClamp][xClamp][yClamp]
        Ui[k] = result1*rati 
        Vi[k] = result2*rati
    
    
    grady_u = (Ui[1]-Ui[3])/(2*rho)
    gradx_v = (Vi[0]-Vi[2])/(2*rho)
    C[i][j] = gradx_v-grady_u                                                 # vorticity
    Position_x[i][j] = pt_0
    Position_y[i][j] = pt_1                                                   # Particle coordinates
    tc = tc+h
    t_c[i][j] = tc
    invalid[i][j] = is_invalid
            
def host_Integrator(xi,yi,Ci,GPU_para,TPB,tspan,m,n,jihe_xyz_u,jihe_xyz_v,mBlend,Nt):
    stream = cuda.stream()
    d_A = cuda.to_device(xi,stream)
    d_B = cuda.to_device(yi,stream)
    d_u = cuda.to_device(jihe_xyz_u,stream)
    d_v = cuda.to_device(jihe_xyz_v,stream)
    d_blend = cuda.to_device(mBlend,stream)
    d_GPUpara = cuda.to_device(GPU_para,stream)
    
    threadsperblock = (TPB,TPB)                                               # (16,16) based on the computing power of the GPU
    blockspergrid_x = math.ceil(xi.shape[0]/threadsperblock[0])               # math.ceil >
    blockspergrid_y = math.ceil(xi.shape[1]/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)                         # threads in each block
    
    tspan1 = tspan[0]
    tc = np.full((m,n), tspan1, dtype=np.float64)                             # time array for each thread
    tc = np.array(tc)
    h = GPU_para[8]
    tspan2 = tspan[-1]
    invalid = np.full((m,n), False, dtype=np.float64)                         # mask array, ocean-false, land-true

    d_invalid = cuda.to_device(invalid,stream)
    d_C2 = cuda.to_device(Ci,stream)
    d_t = cuda.to_device(tc,stream)
    
    t = 0

    LAVD_sum = np.full((m,n), 0.0, dtype=np.float64)
    LAVD_dlt_2 = np.full((m,n), 0.0, dtype=np.float64)
    while((tc[0][0] - tspan2)*(tspan2 - tspan1) < 0.0 ):                      #If trying to integrate past tspan2, reset step size to integrate until tspan2
        if((tc[0][0] + h - tspan2)*(tc[0][0] + h - tspan1) > 0.0):
            h = tspan2 - tc[0][0]

        RK4_kernal_inner[blockspergrid, threadsperblock,stream](d_A, d_B, d_C2, d_u, d_v, d_blend,d_t,d_invalid,d_GPUpara)

        d_A.to_host(stream)
        d_B.to_host(stream)
        d_C2.to_host(stream)
        d_t.to_host(stream)
        d_invalid.to_host(stream)
#        sio.savemat('E:\\lavd\\xpt\\saveddata_%d.mat'%t, {'Ci':Ci})
        LAVD_dlt_1 = np.full((m,n), 0.0, dtype=np.float64)                      # for exchange
        
        Curlz_avg_t = np.mean(Ci)                                               # temp LAVD, for exchange

        LAVD_dlt_1 = LAVD_dlt_2
        
        LAVD_dlt_2 = abs(Ci-Curlz_avg_t)                                        # current LAVD
        
        LAVD_dlt_2 = np.array(LAVD_dlt_2)
        
        t1 = tspan[t]
        t2 = t1+h                                                               # tspan[t+1] for the same

        LAVD_temp = 0.5*(t2-t1)*(LAVD_dlt_1+LAVD_dlt_2)                         # trapz integrate
        LAVD_sum = LAVD_sum+LAVD_temp                                           # sum the LAVD
        print("LAVD：",LAVD_sum[0][0])
#        # save the IVD
#        sio.savemat('G:\\LAVD\\data\\sub\\saveddata_%d.mat'%t, {'LAVDperT':LAVD_temp})
#        stream.synchronize()
        
        t = t+1
        if t == int(Nt/5):
            print('20%',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        if t == int(Nt/2):
            print('50%',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        if t == int(Nt/1.4):
            print('70%',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
#    sio.savemat('G:\\LAVD\\data\\saveddata_LAVD.mat'%t, {'LAVD':LAVD_sum})
#    pdb.set_trace()
            
    print('finish integrate',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    return LAVD_sum,invalid

    
def load_value(h_indirs,latNum,tstart,tp,jihe_xyz_u,jihe_xyz_v):              # load tiff
    for i in range(0,tp,1):
        
        dataset1 = gdal.Open(h_indirs[tstart+i])
        dataset1.RasterCount
        band1 = dataset1.GetRasterBand(1)
        band2 = dataset1.GetRasterBand(2)
        data1 = band1.ReadAsArray()[::]
        data2 = band2.ReadAsArray()[::]
        x1 = dataset1.RasterXSize
        y1 = dataset1.RasterYSize
        for j in range(0,y1,1):
            for k in range(0,x1,1):
                if data1[j][k]!=0:

                    jihe_xyz_u[i][k][latNum-1-j] = data1[j][k]
        for j in range(0,y1,1):
            for k in range(0,x1,1):
                if data2[j][k]!=0:
                    jihe_xyz_v[i][k][latNum-1-j] = data2[j][k]

def load_mat(load_fn,key_name):
    load_data = sio.loadmat(load_fn)
    load_matrix = load_data[key_name] 
    return load_matrix
    
def LAVD_GPU(origin_data,tp,h_indirs):
    print('################### start LAVD_GPU ###################')
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
          
    '''data parameters'''
    print('load_parameter',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))   
    
    dataset1 = gdal.Open(h_indirs[0])
    lonNum = dataset1.RasterXSize
    latNum = dataset1.RasterYSize
    tNum = len(h_indirs)                                                      # include t0
    # pdb.set_trace()
    lonmin = origin_data['data']['data_area'][0]
    lonmax = origin_data['data']['data_area'][1]
    latmin = origin_data['data']['data_area'][2]
    latmax = origin_data['data']['data_area'][3]
#    dt0=origin_data['data']['data_time'][0]                                  # if do not need all the data
#    dtf=origin_data['data']['data_time'][1]
    deltaT = origin_data['data']['deltaT']
    deltalat = origin_data['data']['deltalat']
    deltalon = origin_data['data']['deltalon']
    invalid_value = origin_data['data']['invalid']


    mBlend = [[1.0/6.0, -3.0/6.0,  3.0/6.0, -1.0/6.0],
              [4.0/6.0,  0.0/6.0, -6.0/6.0,  3.0/6.0],
              [1.0/6.0,  3.0/6.0,  3.0/6.0, -3.0/6.0],
              [0.0/6.0,  0.0/6.0,  0.0/6.0,  1.0/6.0]]
    
    '''calculate parameters'''
    TPB = origin_data['TPB']
    t0 = origin_data['t0']
    tf = t0+origin_data['tp']
    Nt = origin_data['Nt']
    tspan = np.linspace(t0,tf,Nt)
    h = tspan[2]-tspan[1]
    
    auxi = origin_data['auxi']
    n = int(auxi*lonNum)
    m = int(auxi*latNum)
    x = np.linspace(lonmin,lonmax,n) 
    dx = abs(x[1]-x[0])
    rho = 0.5*dx                                                              # auxi distance for vorticity along trajectories
    y = np.linspace(latmin,latmax,m)
    xi,yi = np.meshgrid(x,y) 
    
    GPU_para = [lonmin,latmin,deltaT,tNum,latNum,lonNum,deltalon,deltalat,h,rho]
    GPU_para = np.array(GPU_para,dtype='float64')
    
    print('finish init parameter & start load data',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    # jihe_xyz_u = [[[0 for x in range(latNum)] for x in range(lonNum)]for x in range(tNum)]
    # jihe_xyz_v = [[[0 for x in range(latNum)] for x in range(lonNum)]for x in range(tNum)]
    # tstart = 0
    # load_value(h_indirs,latNum,tstart,tNum,jihe_xyz_u,jihe_xyz_v)

#     jihe_xyz_u=np.array(jihe_xyz_u,dtype='float64')
#     jihe_xyz_v=np.array(jihe_xyz_v,dtype='float64')
#     invalid_value=float(-2147483648.0)
#     jihe_xyz_u[jihe_xyz_u==invalid_value]=0                                 # invalid value -> 0
#     jihe_xyz_v[jihe_xyz_v==invalid_value]=0
    
#     np.savez("xyz_u.npz",jihe_xyz_u)
#     np.savez("xyz_v.npz",jihe_xyz_v)
    jihe_xyz_u = np.load("F:\\spyder_code\\LAVD\\xyz_u.npz")
    jihe_xyz_v = np.load("F:\\spyder_code\\LAVD\\xyz_v.npz")
    jihe_xyz_u = np.array(jihe_xyz_u['arr_0'],dtype='float64')
    jihe_xyz_v = np.array(jihe_xyz_v['arr_0'],dtype='float64')

    print('finish load data & start integrate',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 

    mBlend = np.array(mBlend,dtype='float64')
    C = np.full((m, n), 0.0, dtype=np.float64)
    xi = np.array(xi,dtype='float64')
    yi = np.array(yi,dtype='float64')
    
    start = time.time()
    LAVD,is_invalid = host_Integrator(xi, yi, C, GPU_para,TPB,tspan,m,n,jihe_xyz_u,jihe_xyz_v,mBlend,Nt)
    print('gpu mat mul global:', time.time()-start)
    return LAVD,is_invalid

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    