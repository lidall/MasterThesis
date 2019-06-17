#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:07:40 2019

@author: Lida

## this code is used to achieve SDM/WDM node algorithm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;
#import sys
from pandas import DataFrame

'''-----------------------------------------read the file and start to compare the result------------------------------------'''
traffic_result=pd.read_csv("trafficResult.csv", sep=' ')
time_result=pd.read_csv("timeResult.csv", sep=' ')

traffic_contrast=pd.read_csv("trafficResultTestContrast.csv", sep=' ')
time_contrast=pd.read_csv("timeResultTestContrast.csv", sep=' ')

maxTrafficLoadTime=time_result['trafficSuccessNum'].argmax()
maxTrafficLoad=time_result['totalTraffic'][maxTrafficLoadTime]

traffic_time=np.loadtxt('time.txt',dtype=np.int64)
print traffic_time[0][1]

blocking_P=np.zeros(6000)
blocking_P_Contrast=np.zeros(6000)
TrafficNum=0
TrafficSuccess=0
C_TrafficSuccess=0
for i in range(6000):
    if(traffic_result['SuccessFlag'][i]==0):
        TrafficNum+=1.0;
        TrafficSuccess+=0.0
        C_TrafficSuccess+=traffic_contrast['SuccessFlag'][i]
        
        
    elif(traffic_result['SuccessFlag'][i]==1):
        TrafficNum+=1.0;
        TrafficSuccess+=1.0
        C_TrafficSuccess+=traffic_contrast['SuccessFlag'][i]
        
        
        
    blocking_P[i]=1-(TrafficSuccess/TrafficNum)
    blocking_P_Contrast[i]=1-(C_TrafficSuccess/TrafficNum)
'''
ROADMcost=np.zeros(96000)
Contrastcost=np.zeros(96000)
time_cost=np.zeros([6000,96000])
C_time_cost=np.zeros([6000,96000])

for j in range(6000):
    for k in range(traffic_time[j][0],traffic_time[j][1]):
        time_cost[j][k]=traffic_result['cost'][j]
        C_time_cost[j][k]=traffic_contrast['cost'][j]
    ROADMcost+=time_cost[j]
    Contrastcost+=C_time_cost[j]
    print j

'''    
    
    


plt.figure()
plt.plot(range(6000),blocking_P, label="WDM/SDM ROADM")
plt.plot(range(6000),blocking_P_Contrast, label="Contrast ROADM")
plt.xlabel("Traffic Load")
plt.ylabel("Blocking Possibility")
plt.ylim(0,)
plt.legend()
plt.show()
plt.close()

plt.figure()
a=plt.subplot(211)
a.plot(time_result['time'],time_result['trafficSuccessNum'], label="WDM/SDM")
a.plot(time_result['time'],time_contrast['trafficSuccessNum'], label="Contrast")
a.plot(time_result['time'],time_result['totalTraffic'], label="Total Traffic Number")
a.set_ylabel("Traffic Number")
a.set_xlabel("Time (s)")
a.set_ylim(0,)
a.legend(fontsize=8)
a.set_title("( a )")


device=time_result['WSS1']+time_result['WSS2']+time_result['WSSadd']+time_result['WSSdrop']+time_result['split']
a=plt.subplot(212)
a.plot(time_result['time'],device, label="WDM/SDM")
a.plot(time_result['time'],[84]*96000, label="Contrast")
a.plot(time_result['time'],[90]*96000, label="Total Hardware Resource")
a.set_xlabel("Time (s)")
a.set_ylabel("Device Number")
a.set_ylim(0,100)
a.set_title("( b )")

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
a.legend(fontsize=8)
plt.show()
plt.close
#print device
'''
plt.figure()
plt.plot(time_result['time'],ROADMcost, label="ROADM")
plt.plot(time_result['time'],Contrastcost, label="CONTRAST")
plt.legend()
plt.show()
plt.close()    
'''

