#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:50:50 2019

Contrast node algorithms

@author: Lida
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt;
#import sys
from pandas import DataFrame


totalWave=321;
coreNum=7;
nodeDegree=3;
trafficNum=6000;
BypassNum=2;
runTime=96000;

updatFlag=0;
'''
class Device():
    WSS_1st=np.zeros([nodeDegree,coreNum])
    Spliter=np.zeros([nodeDegree,coreNum, nodeDegree])
    WSS_2nd=np.zeros([nodeDegree,coreNum])
    WSS_drop=np.zeros([nodeDegree,coreNum])
    WSS_add=np.zeros([nodeDegree,coreNum])

class Connection():
    WSS1_ports=np.zeros([nodeDegree,coreNum,nodeDegree])
    WSS2_ports=np.zeros([nodeDegree*coreNum,(nodeDegree-1)*coreNum+1])
    SplitterPorts=np.zeros([nodeDegree*coreNum,(nodeDegree-1)*coreNum])
    WSS_addPorts=np.zeros([nodeDegree,coreNum])
    WSS_dropPorts=np.zeros([nodeDegree,coreNum])
''' 
 #n for index of traffic
inputResource=np.zeros([nodeDegree,coreNum,totalWave]);
#n for index of traffic
outputResource=np.zeros([nodeDegree,coreNum,totalWave]);
#1,2,3 for destination, 0 for switch and pass, -1 initial
portDes=[[-1 for col in range(coreNum)] for row in range(nodeDegree)]#only used for inport

#1 for bypass,2 for switch,3 for switch and drop
inportConnection=np.zeros([nodeDegree,coreNum]);# only needs 1,3 
outportConnection=np.zeros([nodeDegree,coreNum]); #only needs 1,2
#matrix of connection of ports,1 for connected, 0 for no
portToPort=np.zeros([coreNum*nodeDegree,coreNum*nodeDegree]); 
#1 for bypass,2 for switch
nextOpOfPort=np.zeros([nodeDegree,coreNum]); 
#traffic cost
trafficCost=np.zeros(trafficNum+1)
trafficOutputPort=np.zeros(trafficNum+1);# track the output port: 0 means drop 7-27 means to other ports successfully -1 means failure
trafficSuccess=np.zeros(trafficNum+1);# 0 means fail and 1 means success
trafficExist=np.zeros(trafficNum+1)

#device=Device();
#connection=Connection();


#list that how traffic and hardware resources change with time
numofTraffic =np.zeros(runTime);
'''
numofWSS_1st = np.zeros(runTime);
numofSplitter = np.zeros(runTime);
numofWSS_2nd = np.zeros(runTime);
numofWSS_drop =np.zeros(runTime);
numofWSS_add = np.zeros(runTime);
numofWSS_1stCon = np.zeros(runTime);
numofSplitterCon = np.zeros(runTime);
numofWSS_2ndCon = np.zeros(runTime);
numofWSS_dropCon =np.zeros(runTime);
numofWSS_addCon = np.zeros(runTime);
'''
numofTotalTraffic=np.zeros(runTime)
throughput=np.zeros(runTime);
bypassNumInPort=np.zeros(nodeDegree)    
bypassNumOutPort=np.zeros(nodeDegree)    

#-------------------functions--------------------------
def wavehCheck(Resource, fiber, port, wave,bandwidth):
    resourceFlag=0;
    for i in range (bandwidth):
        if(Resource[fiber-1][port][wave+i]!=0):
            resourceFlag=1;#once the bandwidth is occupied, then stop the session
            #print wave+i
            #print fiber
            #print port 
            #print Resource[fiber-1][port][wave+i]
            break
    return resourceFlag 

def ADresourceOccupy(fiber, port, wave,trafficIndex, bandwidth, operation, firstfit):
    #only consider the add and drop operation, pass operation will be defined later
    # firstfit is used to indicate the operation. If the input and output fiber has been connected, the it's firstfit. 
  
    if(operation==1):#add
        for b in range(bandwidth):
            #inputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            outputResource[fiber-1][port][wave+b]=trafficIndex;
            '''
        device.WSS_2nd[fiber-1][port]=1
        device.WSS_add[fiber-1][port]=1;
        
        connection.WSS2_ports[(fiber-1)*coreNum+port][(nodeDegree-1)*coreNum]=trafficIndex;
        connection.WSS_addPorts[fiber-1][port]=trafficIndex
        '''
        if(firstfit==True):
            trafficCost[trafficIndex]=2;
        else:
             trafficCost[trafficIndex]=3;
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=trafficIndex;
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            '''
        inportConnection[fiber-1][port]=3;
        device.WSS_1st[fiber-1][port]=1
        device.WSS_drop[fiber-1][port]=1
        
        connection.WSS1_ports[fiber-1][port][nodeDegree-1]=trafficIndex;
        connection.WSS_dropPorts[fiber-1][port]=trafficIndex
        '''
        if(firstfit==True):
            trafficCost[trafficIndex]=2;
        else:
             trafficCost[trafficIndex]=3;
    return

def PassResourceOccupy(infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass):
    #firstfit means that the infiber is connected to outfiber in someway
    #the cost is influenced by the number of hardware used within the system.
    #bypass is used with cost 1 and general pass use 3 hardware to pass the data
    
    for b in range(bandwidth):
        inputResource[infiber-1][inport][wave+b]=trafficIndex;
        #outputResource[infiber-1][inport][wave+bandwidth]=trafficIndex;
        #inputResource[outfiber-1][outport][wave+bandwidth]=trafficIndex;
        outputResource[outfiber-1][outport][wave+b]=trafficIndex;
    if (bypass==True):
        trafficCost[trafficIndex]=1;#bypass without any WSS involved
    else:
         trafficCost[trafficIndex]=3;
         '''
         device.WSS_1st[infiber-1][inport]=1
         device.WSS_2nd[outfiber-1][outport]=1
         if(infiber<outfiber):
             device.Spliter[infiber-1][inport][outfiber-2]=1
             connection.WSS1_ports[infiber-1][inport][outfiber-2]=trafficIndex
             connection.SplitterPorts[(infiber-1)*coreNum+inport][(outfiber-2)*coreNum+outport]=trafficIndex
             connection.WSS2_ports[(outfiber-1)*coreNum+outport][(infiber-1)*coreNum+inport]=trafficIndex
         elif(infiber>outfiber):
             device.Spliter[infiber-1][inport][outfiber-1]=1
             connection.WSS1_ports[infiber-1][inport][outfiber-1]=trafficIndex
             connection.SplitterPorts[(infiber-1)*coreNum+inport][(outfiber-1)*coreNum+outport]=trafficIndex
             connection.WSS2_ports[(outfiber-1)*coreNum+outport][(infiber-2)*coreNum+inport]=trafficIndex
    '''
    
    return

def ADresourceRelease(fiber, port, wave, bandwidth, operation):
    #only consider the add and drop operation, pass operation will be defined later
    # firstfit is used to indicate the operation. If the input and output fiber has been connected, the it's firstfit. 
  
    if(operation==1):#add
        for b in range(bandwidth):
            #inputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            outputResource[fiber-1][port][wave+b]=0;
        #device.WSS_2nd[fiber-1][port]=0
        #device.WSS_add[fiber-1][port]=0;
        '''
        connection.WSS2_ports[(fiber-1)*coreNum+port][(nodeDegree-1)*coreNum]=0;
        connection.WSS_addPorts[fiber-1][port]=0
        outportConnection[fiber-1][port]=0
        for k in range(coreNum*nodeDegree):
            portToPort[k][(fiber-1)*coreNum+port]=0
        '''
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=0
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
        '''    
        inportConnection[fiber-1][port]=0;

        #device.WSS_1st[fiber-1][port]=0
        #device.WSS_drop[fiber-1][port]=0
        portDes[fiber-1][port]=-1
        connection.WSS1_ports[fiber-1][port][nodeDegree-1]=0
        connection.WSS_dropPorts[fiber-1][port]=0
        for k in range(coreNum*nodeDegree):
            portToPort[(fiber-1)*coreNum+port][k]=0
            '''
    return            
            
def PassResourceRelease(infiber, inport,outfiber,outport, wave, bandwidth, bypass):
    #firstfit means that the infiber is connected to outfiber in someway
    #the cost is influenced by the number of hardware used within the system.
    #bypass is used with cost 1 and general pass use 3 hardware to pass the data
    
    for b in range(bandwidth):
        inputResource[infiber-1][inport][wave+b]=0;
        #outputResource[infiber-1][inport][wave+bandwidth]=trafficIndex;
        #inputResource[outfiber-1][outport][wave+bandwidth]=trafficIndex;
        outputResource[outfiber-1][outport][wave+b]=0;
        '''
    portDes[infiber-1][inport]=-1
    inportConnection[infiber-1][inport]=0
    outportConnection[outfiber-1][outport]=0

    if (bypass==False):
         #device.WSS_1st[infiber-1][inport]=1
         #device.WSS_2nd[outfiber-1][outport]=1
        if(infiber<outfiber):
             #device.Spliter[infiber-1][inport][outfiber-2]=1
            connection.WSS1_ports[infiber-1][inport][outfiber-2]=0
            connection.SplitterPorts[(infiber-1)*coreNum+inport][(outfiber-2)*coreNum+outport]=0
            connection.WSS2_ports[(outfiber-1)*coreNum+outport][(infiber-1)*coreNum+inport]=0
        elif(infiber>outfiber):
             #device.Spliter[infiber-1][inport][outfiber-1]=1
            connection.WSS1_ports[infiber-1][inport][outfiber-1]=0
            connection.SplitterPorts[(infiber-1)*coreNum+inport][(outfiber-1)*coreNum+outport]=0
            connection.WSS2_ports[(outfiber-1)*coreNum+outport][(infiber-2)*coreNum+inport]=0
        for k in range(coreNum*nodeDegree):
            portToPort[(infiber-1)*coreNum+inport][k]=0
            portToPort[k][(outfiber-1)*coreNum+outport]=0
    '''
    return        

"""  *********     Read file from the input files  *******    """
trraffic = np.loadtxt('traffic.txt')
trrafficc=np.delete(trraffic,[0],axis=1)
trafficindex=np.arange(trafficNum).reshape(trafficNum,1)
traffic_time=np.loadtxt('time.txt')
Traffic=np.hstack((trafficindex,trrafficc,traffic_time))
tra_columns=['index','preNode','nextNode','inPort','operation','nextOperation', 'startWave','bandwidth','createTime','releaseTime']
traffic =pd.DataFrame(Traffic, columns=tra_columns,dtype=np.int64)
#print Traffic.shape
#print traffic


"""------------------------algorithms---------------------"""
infiber =0;
outfiber=0
inport =0

for t in range(1,runTime):#start running
    print t
    numofTraffic[t] = numofTraffic[t-1];
    numofTotalTraffic[t] = numofTotalTraffic[t-1];
    '''
    numofWSS_1st[t] = numofWSS_1st[t-1];
    numofSplitter[t] = numofSplitter[t-1];
    numofWSS_2nd[t] = numofWSS_2nd[t-1];
    numofWSS_drop[t] = numofWSS_drop[t-1];
    numofWSS_add[t] = numofWSS_add[t-1];
    '''
    throughput[t] = throughput[t-1];
    updatFlag=0
    
    for i in range(trafficNum):#traffic start time check
        infiber=traffic['preNode'][i]
        
        outfiber=int(traffic['nextNode'][i])
        inport=int(traffic['inPort'][i]-1) # consist with the index of array
        #print range(trafficNum)
        if(traffic['createTime'][i]==t):
            numofTotalTraffic[t]+=1
            print traffic['index'][i]         
            trafficCost[i]=0
           
            trafficExist[i]=1
            firstFit=False
            bypass=False

            if(traffic['operation'][i]==1):
        
                c =np.random.randint(0,coreNum-1)
                resourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])

                if(resourceFlag==0):
                     firstFit=True
                     ADresourceOccupy(outfiber,c,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],1,firstFit)
                     trafficOutputPort[i]=outfiber*coreNum+c+1;
                     trafficSuccess[i]=1;
                     updatFlag=1
                     throughput[t]+=traffic['bandwidth'][i]
                     numofTraffic[t]+=1 # add the success traffic number and success traffic bandwidth
                else:
                    trafficOutputPort[i]=-1
                    trafficSuccess[i]=0;#failure situation

            elif(traffic['operation'][i]==2):
                inputresourceFlag=wavehCheck(inputResource, infiber, inport, traffic['startWave'][i],traffic['bandwidth'][i])
                if(inputresourceFlag==1):
                    trafficOutputPort[i]=-1;
                    trafficSuccess[i]=0;
                else:
                    outputport=inport;
                    outputresourceFlag=wavehCheck(outputResource, outfiber, outputport, traffic['startWave'][i],traffic['bandwidth'][i])
                    if(outputresourceFlag==1):
                        trafficOutputPort[i]=-1;
                        trafficSuccess[i]=0;
                    else:
                        PassResourceOccupy(infiber, inport,outfiber,outputport, traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                        #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                        trafficOutputPort[i]=outfiber*coreNum+outputport+1;
                        trafficSuccess[i]=1;
                        updatFlag=1
                        throughput[t]+=traffic['bandwidth'][i]
                        numofTraffic[t]+=1
                                   
            elif(traffic['operation'][i]==3):#drop
               inputresourceFlag=wavehCheck(inputResource, infiber, inport, traffic['startWave'][i],traffic['bandwidth'][i])
               if(inputresourceFlag==0):
                   firstFit=True
                   ADresourceOccupy(infiber,inport,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],3,firstFit)
                   trafficOutputPort[i]=0;
                   trafficSuccess[i]=1;
                   updatFlag=1                           
                   throughput[t]+=traffic['bandwidth'][i]
                   numofTraffic[t]+=1
               else:                   
                   trafficOutputPort[i]=-1;
                   trafficSuccess[i]=0;


#---------------------------traffic ends-------------------
        if(traffic['releaseTime'][i]==t):
            numofTotalTraffic[t]-=1
            trafficExist[i]=0
            if(trafficSuccess[i]==1):
                updatFlag=1
              #only successful traffic needs release resource operation
                numofTraffic[t]-=1;
                throughput[t]-=traffic['bandwidth'][i];
                splitterResourceFlag=0
                inputresourceFlag=0
                outputresourceFlag=0
                infiber=traffic['preNode'][i]
                outfiber=traffic['nextNode'][i]
                inport=traffic['inPort'][i]-1

                if(traffic['operation'][i]==1):
                    
                  #this traffic ought to be added into the system
                    ADresourceRelease(outfiber,int(trafficOutputPort[i]%coreNum-1),traffic['startWave'][i],traffic['bandwidth'][i],1)
                    #trafficSuccess[i]=0
                elif(traffic['operation'][i]==3):
                  
                    ADresourceRelease(infiber,inport,traffic['startWave'][i],traffic['bandwidth'][i],3)
                    #trafficSuccess[i]=0
                elif(traffic['operation'][i]==2):
                    #trafficSuccess[i]=0
                    PassResourceRelease(infiber, inport,outfiber,int(trafficOutputPort[i]%coreNum-1), traffic['startWave'][i],traffic['bandwidth'][i], bypass)


"""-------------------traffic result output---------------"""
Time=range(runTime)
timeResult=pd.DataFrame({'time':Time,'totalTraffic':numofTotalTraffic,'trafficSuccessNum':numofTraffic,'throughput':throughput})
trafficResult=pd.DataFrame({'index':range(trafficNum+1),'outport':trafficOutputPort,'cost':trafficCost,'SuccessFlag':trafficSuccess})                                                        
                        
timeResult.to_csv("timeResultTestContrast.csv",index=False,sep=' ')                        
trafficResult.to_csv("trafficResultTestContrast.csv",index=False,sep=' ')                       
                       
  

