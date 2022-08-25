# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 15:56:21 2021

define the LAVD_parameters

@author: Mengjiao Wang
"""

import json

result={'input_parameters':
{'2D':                                           #2D/3D LAVD
{'data':
{'uv_path': 'E:\\SLA_nc&tif\\tif\\',             # SLA uv path
'data_area': [0.125,359.875,-89.875,89.875],     # lonmin,lonmax,latmin,latmax of data
'data_time': [0,90],                             # t0,tf of data,not used
'out_dir': './',                                 # outdir
'deltaT': 1,                                     # time resolution of data
'invalid': -2147483648.0,#10000000,              # Invalid values
'deltalat': 0.25,                                # spatial resolution of data (latitude)
'deltalon': 0.25},                               # spatial resolution of data (longitude)
'TPB': 16,                                       # grid*grid of GPU
'study_area': [0.125,359.875,-89.875,89.875],    # lonmin,lonmax,latmin,latmax of study area
't0': 0,                                         # t0 of study time span
'tp': 90,                                        # time period of study time span
'Nt': 901,                                       # number of auxi time in current timespan
'auxi':7,                                        # times of auxi spatial resolution 
'save_path':'./'                                 # save path
}}}

out_dir='./LAVD_parameters.json'
json.dump(result,open(out_dir,'w'))