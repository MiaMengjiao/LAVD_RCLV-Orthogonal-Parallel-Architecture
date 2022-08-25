# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 19:24:31 2021

define the RCLV_parameters
1. Contains no more than one center;
2. Vortex size ≥ 4 pixels;
3. Convexity deficiency = 0.01.

@author: Mengjiao Wang
"""

import json

result={'out_info':
{'2D':                                                                  # surface
{'only_lavd': True,                                                       # True: lavd, False: h and lavd
'input_dir':
{'h': './',
'uv': './'},            
'out_dir': './',                                                        # Path of results
'left_top_right_bottom':[0.125,-89.875,359.875,89.875],                 # [originX=origin_lon, originY=origin_lat，endX= end_lon, endY =end_lat]
'study_area':[0.125,359.875,-89.875,89.875],                            # [lonmin, lonmax, latmin, latmax]
'interval':1.0,                                                         # Contour traversal interval
'fillvalue':
 {'h':-9999,
  'uv':-2147483648.0}, #10000000                                        # Invalid values
'if_block_auto': False,                                                 # The default [30,30]
'if_block': True,                                                       # True/False ‘block_num’:[lon_block_num,lat_block_num]
'inner_lon_lat':[30,30],                                                # [lon_inner,lat_inner]
'overlapping_area':[10,10],                                             # [lon_overlapping,lat_overlapping]
'if_parallel': False,                                                   # True/False
'DeficiencyThresh': 0.01,                                               # Convexity deficiency
'LengthThresh':1.57,                                                    # Minimum Perimeter of RCLV boundary
'parallel_num':15                                                       # numbers of processes
}}}


out_dir='./RCLV_parameters.json'
json.dump(result,open(out_dir,'w'))