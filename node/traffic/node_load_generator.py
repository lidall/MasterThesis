#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:09:21 2019

@author: Lida
"""

import numpy as np
from pandas.core.frame import DataFrame


load_number=range(1,201)
load=[]
for i in range(200):
    load.append(load_number[i]*50)

ranPort1=[1,4,5,6,7]
ranPort2=[1,2,3,6,7]
ranPort3=[1,2,3,4,5]

Bandwidth=[3,4,8,32]
"""" start """""
for i in range(200):
    node1=[]
    node2=[]
    port=[]
    operation=[]
    startwave=[]
    bandwidth=[]    
    
   # print (load[i])
    while(len(node1)<=(load[i]-1)):
        Node1=np.random.randint(0,4)
        Node2=np.random.randint(0,4)
        S_wave=np.random.randint(0,320)
        random_seed=np.random.randint(48)
        if(random_seed<12):
            S_band=Bandwidth[0]
        if(random_seed<24 and random_seed>11):
            S_band=Bandwidth[1]
        if(random_seed<36 and random_seed>23):
            S_band=Bandwidth[2]
        if(random_seed<48 and random_seed>35):
            S_band=Bandwidth[3]        
        
        if(Node1!=Node2 and S_wave+S_band<=320 ):

            if(Node1==0):
                Port=0
                OP=1

            elif(Node2==0):
                if(Node1==1):
                    Port=ranPort1[np.random.randint(0,5)]
                elif(Node1==2):
                    Port=ranPort2[np.random.randint(0,5)]
                elif(Node1==3):
                    Port=ranPort3[np.random.randint(0,5)]
                OP=3    
              
            else:
                Port= np.random.randint(1,8)
                if(Node1==1 and Node2==2 and Port==3):
                    continue
                if(Node1==1 and Node2==3 and Port==2):
                    continue
                if(Node1==2 and Node2==3 and Port==4):
                    continue
                if(Node1==2 and Node2==1 and Port==5):
                    continue
                if(Node1==3 and Node2==1 and Port==7):
                    continue
                if(Node1==3 and Node2==2 and Port==6):
                    continue
                    
               
                OP=2
                
            node1.append(Node1)
            node2.append(Node2)
            startwave.append(S_wave)
            bandwidth.append(S_band)
            port.append(Port)
            operation.append(OP)
                
                
     
    Dic={"1node1":node1,
         "2node2":node2,
         "3port":port,
         "4operation":operation,
         "5startwave": startwave,
         "6bandwidth":bandwidth
             
            }           
    DF=DataFrame(Dic)
    txtname='Load-'+str(load[i])+'.txt'
    DF.to_csv(txtname, sep='\t', index=False,header=None)                
            
        
        