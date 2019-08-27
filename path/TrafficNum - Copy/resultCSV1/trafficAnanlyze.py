# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:05:11 2019

@author: os19107
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import permutations



Contrast=pd.read_csv('Contras.csv')
Contrast_nobypass=pd.read_csv('contrast_nobypass.csv')

Random_nobypass=pd.read_csv('RandomwavelengthnoBypas.csv')
Random=pd.read_csv('Randomwavelength.csv')

Pack=pd.read_csv('packedwavelength.csv')
Pack_nobypass=pd.read_csv('packedwavelengthnoBypas.csv')

Spread=pd.read_csv('spreadwavelength.csv')
Spread_nobypass=pd.read_csv('spreadwavelengthnoBypas.csv')

Success=[]
Traffic_length=[]
traffic_band=[]

Total_trafficNum=[]
traffic=range(len(Contrast))
Total_trafficNum=[(x+8)*50 for x in traffic]
TotalBandNum=[int((x*24)/4) for x in Total_trafficNum ]

contrast_blockRatio=[]
contrast_nobypass_block=[]
contrast_BandblockRatio=[]
contrast_nobypass_Bandblock=[]
traffic=[]
contrast_average_link=[]
contrast_average_linkBypass=[]
for i in range(len(Contrast)):
    if((i%8)==0):
        traffic.append(Total_trafficNum[i])
        contrast_blockRatio.append((1-Contrast['success'][i]/Total_trafficNum[i])*0.5)
        contrast_nobypass_block.append((1-Contrast_nobypass['success'][i]/Total_trafficNum[i])*0.5)
        contrast_BandblockRatio.append((1-Contrast['band'][i]/TotalBandNum[i])*0.5)
        contrast_nobypass_Bandblock.append((1-Contrast_nobypass['band'][i]/TotalBandNum[i])*0.5)
        contrast_average_link.append(Contrast['length'][i])
        contrast_average_linkBypass.append(Contrast_nobypass['length'][i])
    
plt.figure()
plt.plot(traffic, contrast_blockRatio,marker = 'o',label='baseline algorithm in bypass network')
plt.plot(traffic, contrast_nobypass_block,marker = '+',label='baseline algorithm in no-bypass network')
plt.xlabel('Traffic Load (Erlang)')
plt.ylabel('Blocking Probability')
plt.legend()
plt.show()
    

plt.figure()
plt.plot(traffic, contrast_BandblockRatio,marker = 'o',label='baseline algorithm in bypass network')
plt.plot(traffic, contrast_nobypass_Bandblock,marker = '+',label='baseline algorithm in no-bypass network')
plt.xlabel('Traffic Load (Erlang)')
plt.ylabel('Bandwidth Blocking Probability')
plt.legend()
plt.show()

plt.figure()
plt.plot(traffic, contrast_average_link,marker = 'o',label='baseline algorithm in bypass network')
plt.plot(traffic, contrast_average_linkBypass,marker = '+',label='baseline algorithm in no-bypass network')
plt.xlabel('Traffic Load (Erlang)')
plt.ylabel('Average Path Length (hops)')
plt.legend()
plt.show()