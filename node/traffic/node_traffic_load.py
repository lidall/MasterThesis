#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:17:02 2019

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
BypassNum=2;
runTime=2;


""" ------------------------------------------------------------------------ """    
#----------------functions---------------------#    
def bypassInitial():# need adjustment
    portDes[0][1]=2#bypass connection node
    portDes[0][2]=3
    portDes[1][3]=1
    portDes[1][4]=3
    portDes[2][5]=1
    portDes[2][6]=2
    
    portToPort[coreNum*0+1][coreNum*1+3]=1#bypass node connection
    portToPort[coreNum*1+3][coreNum*0+1]=1
    
    portToPort[coreNum*0+2][coreNum*2+5]=1
    portToPort[coreNum*2+5][coreNum*0+2]=1
    
    portToPort[coreNum*1+4][coreNum*2+6]=1
    portToPort[coreNum*2+6][coreNum*1+4]=1
    
    inportConnection[0][1]=1;
    inportConnection[0][2]=1;
    inportConnection[1][3]=1;
    inportConnection[1][4]=1;
    inportConnection[2][5]=1;
    inportConnection[2][6]=1;
    outportConnection[0][1]=1;
    outportConnection[0][2]=1;
    outportConnection[1][3]=1;
    outportConnection[1][4]=1;
    outportConnection[2][5]=1;
    outportConnection[2][6]=1;
    
    bypassNumInPort[1-1]=2
    bypassNumInPort[2-1]=2
    bypassNumInPort[3-1]=2
    bypassNumOutPort[1-1]=2
    bypassNumOutPort[2-1]=2
    bypassNumOutPort[3-1]=2
    
    return;
    
def freePortCheck(fiber,direction):#direction means the direction of data, out is 0 and in is 1
    freePort=-1
    if(direction==0):
        for i in range(coreNum):
            if (outportConnection[fiber-1][i]==0):
                freePort=i;
                break
    elif(direction==1):
         for i in range(coreNum):
            if (inportConnection[fiber-1][i]==0):
                freePort=i;
                break
    
    return freePort
            

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

def setConnection(toBefiber,toBeport,operation):#operation 1: add, 2: pass, 3: drop
    for i in range(coreNum*nodeDegree):
        fiber=int(i/coreNum)
        port=int(i%coreNum)
        if(operation==1):
            if(inportConnection[fiber][port]==3 and fiber!=toBefiber-1 and portDes[fiber][port]==0):
                portToPort[i][(toBefiber-1)*coreNum+toBeport]=1
        elif(operation==3):
            if(outportConnection[fiber][port]==2 and fiber!=toBefiber-1 and portDes[fiber][port]==0):
                portToPort[(toBefiber-1)*coreNum+toBeport][i]=1
    return

def setPassConnection(toBeInfiber,toBeInport,toBeOutfiber,toBeOutport):#operation 1: add, 2: pass, 3: drop
    for i in range(coreNum*nodeDegree):
        fiber=int(i/coreNum)
        port=int(i%coreNum)
        if(inportConnection[fiber][port]==3 and fiber!=toBeOutfiber-1 and portDes[fiber][port]==0):
            portToPort[i][(toBeOutfiber-1)*coreNum+toBeOutport]=1
        if(outportConnection[fiber][port]==2 and fiber!=toBeInfiber-1 and portDes[fiber][port]==0):
            portToPort[(toBeInfiber-1)*coreNum+toBeInport][i]=1
    return




def ADresourceOccupy(fiber, port, wave,trafficIndex, bandwidth, operation, firstfit):
    #only consider the add and drop operation, pass operation will be defined later
    # firstfit is used to indicate the operation. If the input and output fiber has been connected, the it's firstfit. 
  
    if(operation==1):#add
        for b in range(bandwidth):
            #inputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            outputResource[fiber-1][port][wave+b]=trafficIndex+1;
        WSS_2nd[fiber-1][port]=1
        WSS_add[fiber-1][port]=1;
        
        WSS2_ports[(fiber-1)][port][(nodeDegree)*coreNum]+=trafficIndex+1;
        WSS_addPorts[fiber-1][port]+=trafficIndex+1
        
        if(firstfit==True):
            trafficCost[trafficIndex]=2;
        else:
             trafficCost[trafficIndex]=3;
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=trafficIndex;
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
        inportConnection[fiber-1][port]=3;
        WSS_1st[fiber-1][port]=1
        WSS_drop[fiber-1][port]=1
        
        WSS1_ports[fiber-1][port][nodeDegree]+=trafficIndex+1;
        WSS_dropPorts[fiber-1][port]+=trafficIndex+1
        
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
         trafficCost[trafficIndex]=4;
         WSS_1st[infiber-1][inport]=1
         WSS_2nd[outfiber-1][outport]=1
         Spliter[infiber-1][inport][outfiber-1]=1
         WSS1_ports[infiber-1][inport][outfiber-1]+=trafficIndex+1
         SplitterPorts[(infiber-1)][inport][(outfiber-1)][outport]+=trafficIndex+1
         WSS2_ports[(outfiber-1)][outport][(infiber-1)*coreNum+inport]+=trafficIndex+1
    
    return
        
def ADresourceRelease(fiber, port, wave, bandwidth, trafficIndex,operation):
    #only consider the add and drop operation, pass operation will be defined later
    # firstfit is used to indicate the operation. If the input and output fiber has been connected, the it's firstfit. 
  
    if(operation==1):#add
        for b in range(bandwidth):
            #inputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            outputResource[fiber-1][port][wave+b]=0;
        #device.WSS_2nd[fiber-1][port]=0
        #device.WSS_add[fiber-1][port]=0;
        
        WSS2_ports[(fiber-1)][port][(nodeDegree)*coreNum]-=trafficIndex+1;
        WSS_addPorts[fiber-1][port]-=trafficIndex+1
        if(np.count_nonzero(outputResource[fiber-1][port])==0):
            outportConnection[fiber-1][port]=0
            for k in range(coreNum*nodeDegree):
                portToPort[k][(fiber-1)*coreNum+port]=0
        
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=0
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
        if(np.count_nonzero(inputResource[fiber-1][port])==0):
            inportConnection[fiber-1][port]=0;

        #device.WSS_1st[fiber-1][port]=0
        #device.WSS_drop[fiber-1][port]=0
            portDes[fiber-1][port]=-1
            for k in range(coreNum*nodeDegree):
                portToPort[(fiber-1)*coreNum+port][k]=0
        WSS1_ports[fiber-1][port][nodeDegree]-=trafficIndex+1
        WSS_dropPorts[fiber-1][port]-=trafficIndex+1
    return            
            
def PassResourceRelease(infiber, inport,outfiber,outport, wave, bandwidth,trafficIndex, bypass):
    #firstfit means that the infiber is connected to outfiber in someway
    #the cost is influenced by the number of hardware used within the system.
    #bypass is used with cost 1 and general pass use 3 hardware to pass the data
    
    for b in range(bandwidth):
        inputResource[infiber-1][inport][wave+b]=0;
        #outputResource[infiber-1][inport][wave+bandwidth]=trafficIndex;
        #inputResource[outfiber-1][outport][wave+bandwidth]=trafficIndex;
        outputResource[outfiber-1][outport][wave+b]=0;
    if(np.count_nonzero(inputResource[infiber-1][inport])==0):
        portDes[infiber-1][inport]=-1
        inportConnection[infiber-1][inport]=0
    if(np.count_nonzero(outputResource[outfiber-1][outport])==0):
        outportConnection[outfiber-1][outport]=0

    if (bypass==False):
         #device.WSS_1st[infiber-1][inport]=1
         #device.WSS_2nd[outfiber-1][outport]=1
             #device.Spliter[infiber-1][inport][outfiber-2]=1
        
        WSS1_ports[infiber-1][inport][outfiber-1]-= trafficIndex+1               
        WSS2_ports[outfiber-1][outport][(infiber-1)*coreNum+inport]-=trafficIndex+1
        SplitterPorts[(infiber-1)][inport][(outfiber-1)][outport]-=trafficIndex+1
        for k in range(coreNum*nodeDegree):
            
            if(np.count_nonzero(inputResource[infiber-1][inport])==0):
                portToPort[(infiber-1)*coreNum+inport][k]=0
            if(np.count_nonzero(outputResource[outfiber-1][outport])==0):
                portToPort[k][(outfiber-1)*coreNum+outport]=0
    
    return

    
"""""""----------------------------functions over

for e in range(120):
    trafficNum=50*(e+1);
    pdatFlag=0;

    WSS_1st=np.zeros([nodeDegree,coreNum])
    Spliter=np.zeros([nodeDegree,coreNum, nodeDegree])
    WSS_2nd=np.zeros([nodeDegree,coreNum])
    WSS_drop=np.zeros([nodeDegree,coreNum])
    WSS_add=np.zeros([nodeDegree,coreNum])
    
    
    WSS1_ports=np.zeros([nodeDegree,coreNum,nodeDegree+1])
    WSS2_ports=np.zeros([nodeDegree,coreNum, nodeDegree*coreNum+1])
    SplitterPorts=np.zeros([nodeDegree,coreNum,nodeDegree,coreNum])
    WSS_addPorts=np.zeros([nodeDegree,coreNum])
    WSS_dropPorts=np.zeros([nodeDegree,coreNum])
     
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
    
    '''
    device=Device();
    connection=Connection();
    '''
    
    #list that how traffic and hardware resources change with time
    numofTraffic =np.zeros(runTime);
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
    numofTotalTraffic=np.zeros(runTime)
    throughput=np.zeros(runTime);
    bypassNumInPort=np.zeros(nodeDegree)    
    bypassNumOutPort=np.zeros(nodeDegree)  
        """  *********     Read file from the input files  *******    """
    trraffic = np.loadtxt('traffic.txt')
    trrafficc=np.delete(trraffic,[0],axis=1)
    #print trrafficc
    trafficindex=np.arange(trafficNum).reshape(trafficNum,1)
    traffic_time=np.loadtxt('time.txt')
    Traffic=np.hstack((trafficindex,trrafficc,traffic_time))
    tra_columns=['index','preNode','nextNode','inPort','operation','nextOperation', 'startWave','bandwidth','createTime','releaseTime']
    traffic =pd.DataFrame(Traffic, columns=tra_columns,dtype=np.int64)
    #print Traffic.shape
    print traffic
    
    
    
    
    """
            ********************************************************
            ******************  algorithm  *************************
            ********************************************************
            
    """
    infiber =0;
    outfiber=0
    inport =0
    
    for t in range(1,runTime):#start running
        
        
       
        print "--------------------"
        print t
        print "--------------------"
        numofTraffic[t] = numofTraffic[t-1];
        numofTotalTraffic[t] = numofTotalTraffic[t-1];
        numofWSS_1st[t] = numofWSS_1st[t-1];
        numofSplitter[t] = numofSplitter[t-1];
        numofWSS_2nd[t] = numofWSS_2nd[t-1];
        numofWSS_drop[t] = numofWSS_drop[t-1];
        numofWSS_add[t] = numofWSS_add[t-1];
        throughput[t] = throughput[t-1];
        updatFlag=0
        
        for i in range(trafficNum):#traffic start time check
            #print range(trafficNum)
            infiber=traffic['preNode'][i]
            
            outfiber=int(traffic['nextNode'][i])
            inport=int(traffic['inPort'][i]-1) # consist with the index of array
        
             #----------------traffic satrt---------------------#
             
            if(traffic['createTime'][i]==t):
               numofTotalTraffic[t]+=1
               print traffic['index'][i]         
               trafficCost[i]=0
               
               trafficExist[i]=1
               firstFit=False
               bypass=False
               bypassInitial()
               #------------------add operation-----------------------#
               if(traffic['operation'][i]==1):
                   # wave add into the system,look for resource can be used 
                   #first check if there are connected core of outfiber
                   for c in range (coreNum):
                       
                       #--------------connection exist---------------#
                       
                       if (outportConnection[outfiber-1][c]==2):# or portConnection[outfiber-1][c]==3 ):
                           #there is a fiber core already connected with a switch
                           resourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                           if(resourceFlag==0):
                               firstFit=True
                               ADresourceOccupy(outfiber,c,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],1,firstFit)
                               #fiber, port, wave,trafficIndex, bandwidth, operation, firstfit
                               trafficOutputPort[i]=outfiber*coreNum+c+1;
                               trafficSuccess[i]=1;
                               updatFlag=1
                               throughput[t]+=traffic['bandwidth'][i]
                               numofTraffic[t]+=1 # add the success traffic number and success traffic bandwidth
                               break
                   #connection didn't exist, new connection needs to be setup#
                   findOutputPort=-1;
                   findOutputPort=freePortCheck(outfiber-1,0);
    
                   if(findOutputPort!=-1 and trafficCost[i]==0):   
                       
                       outportConnection[outfiber-1][findOutputPort]=2;
                       #portDes[outfiber-1][findOutputPort]=0;
                       setConnection(outfiber,findOutputPort,1);
                       ADresourceOccupy(outfiber,findOutputPort,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],1,firstFit)
                       
                       trafficOutputPort[i]=outfiber*coreNum+findOutputPort+1;
                       trafficSuccess[i]=1;
                       updatFlag=1        
                       throughput[t]+=traffic['bandwidth'][i]
                       numofTraffic[t]+=1 # add the success traffic number and success traffic bandwidth
                       
                   elif(findOutputPort==-1 and trafficCost[i]==0):
                       trafficOutputPort[i]=-1;
                       trafficSuccess[i]=0;#failure situation
                       
                       
               #------------------pass operation-----------------------#                   
               elif(traffic['operation'][i]==2):
                   inputresourceFlag=wavehCheck(inputResource, infiber, inport, traffic['startWave'][i],traffic['bandwidth'][i])
               
                   if(inputresourceFlag==1):
                       trafficOutputPort[i]=-1;
                       trafficSuccess[i]=-1;
                       numofTotalTraffic[t]-=1
                   else:
                       if(portDes[infiber-1][inport]==outfiber):
                           for c in range (coreNum):
                               if(portToPort[(infiber-1)*coreNum+inport][(outfiber-1)*coreNum+c]==1 and outportConnection[outfiber-1][c]==1):
                                   bypass=True
                                   outputresourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                                   if(outputresourceFlag==1):
                                       trafficOutputPort[i]=-1;
                                       trafficSuccess[i]=0;
                                       
                                   else:
                                       PassResourceOccupy(infiber, inport,outfiber,c, traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                       #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                                       trafficOutputPort[i]=outfiber*coreNum+c+1;
                                       trafficSuccess[i]=1;
                                       updatFlag=1
                                       throughput[t]+=traffic['bandwidth'][i]
                                       numofTraffic[t]+=1
                                       break
                                   
                       elif(portDes[infiber-1][inport]==0):
                           passsuccessfulFlag=0
                           #the input fiber is connected with switch
                           for c in range (coreNum):
                               if(portToPort[(infiber-1)*coreNum+inport][(outfiber-1)*+c]==1 and outportConnection[outfiber-1][c]==2):
                                   bypass=False
                                   outputresourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                                   if(outputresourceFlag==1):
                                       trafficOutputPort[i]=-1;
                                       trafficSuccess[i]=0;
                                       
                                   else:
                                       PassResourceOccupy(infiber, inport,outfiber,c, traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                       #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                                       trafficOutputPort[i]=outfiber*coreNum+c+1;
                                       trafficSuccess[i]=1;
                                       updatFlag=1
                                       throughput[t]+=traffic['bandwidth'][i]
                                       numofTraffic[t]+=1
                                       passsuccessfulFlag=1
                                       break 
                           if(passsuccessfulFlag==0):
                               for c in range (coreNum):
                                   if(outportConnection[outfiber-1][c]==1):
                                       continue# in case connect with bypass                                    
                                   outputresourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                                   if(outputresourceFlag==1):
                                       trafficOutputPort[i]=-1;
                                       trafficSuccess[i]=0;
                                   else:
                                       setPassConnection(infiber,inport,outfiber,c)
                                       PassResourceOccupy(infiber, inport,outfiber,c, traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                       trafficOutputPort[i]=outfiber*coreNum+c+1;
                                       trafficSuccess[i]=1;
                                       updatFlag=1
                                       throughput[t]+=traffic['bandwidth'][i]
                                       numofTraffic[t]+=1
                                       passsuccessfulFlag=1
                                       break 
                                                               
                                  
                       elif(portDes[infiber-1][inport]==-1):#initial input port without any connection
                           findOutputPortList=[];
                           successflag=0
                               
                           for c in range (coreNum):
                               bypass=False
                               if(outportConnection[outfiber-1][c]==1):
                                   continue
                               outputresourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                               if(outputresourceFlag==1):
                                   trafficOutputPort[i]=-1;
                                   trafficSuccess[i]=0;
                               else:
                                   
                                   if(outportConnection[outfiber-1][c]==2):
                                       inportConnection[infiber-1][inport]=3
                                       outportConnection[outfiber-1][c]=2
                                       portDes[infiber-1][inport]=0#connected with switch& pass
                                       nextOpOfPort[outfiber-1][c]=2
                                   
                                       setPassConnection(infiber,inport,outfiber,c)#setup connection
                                   
                                       PassResourceOccupy(infiber, inport,outfiber,c, traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                   #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                                       trafficOutputPort[i]=outfiber*coreNum+c+1;
                                       trafficSuccess[i]=1;
                                       updatFlag=1
                                       throughput[t]+=traffic['bandwidth'][i]
                                       numofTraffic[t]+=1
                                       successflag=1
                                       break  
                                   elif(outportConnection[outfiber-1][c]==0):
                                       findOutputPortList.append(c)
                           
                            
                            
     
                           if(outputresourceFlag==0 and len(findOutputPortList)!=0 and successflag==0 ):
                               #the new bypass set up 
                                   
                               if(bypassNumInPort[infiber-1]<2 and bypassNumOutPort[outfiber-1]<2):
                                   inportConnection[infiber-1][inport]=1
                                   outportConnection[outfiber-1][findOutputPortList[0]]=1
                                   portDes[infiber-1][inport] =outfiber;
                                   portToPort[(((infiber-1)*coreNum+inport))][findOutputPortList[0]] = 1
                                   bypassNumInPort[infiber-1]+=1
                                   bypassNumOutPort[outfiber-1]+=1
                                   bypass=True
                                   
                                   PassResourceOccupy(infiber, inport,outfiber,findOutputPortList[0], traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                   #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                                   trafficOutputPort[i]=outfiber*coreNum+findOutputPortList[0]+1;
                                   trafficSuccess[i]=1;
                                   updatFlag=1
                                   throughput[t]+=traffic['bandwidth'][i]
                                   numofTraffic[t]+=1
                                #new switch way set up   
                               else: 
                                   inportConnection[infiber-1][inport]=3
                                   outportConnection[outfiber-1][findOutputPortList[0]]=2
                                   portDes[infiber-1][inport]=0#connected with switch& pass
                                   nextOpOfPort[outfiber-1][findOutputPortList[0]]=2
                                   setPassConnection(infiber,inport,outfiber,findOutputPortList[0])#setup connection
                                   
                                   bypass=False
                                   PassResourceOccupy(infiber, inport,outfiber,findOutputPortList[0], traffic['startWave'][i],traffic['index'][i], traffic['bandwidth'][i], bypass)
                                   #infiber, inport,outfiber,outport, wave,trafficIndex, bandwidth, bypass
                                   trafficOutputPort[i]=outfiber*coreNum+findOutputPortList[0]+1;
                                   trafficSuccess[i]=1;
                                   updatFlag=1
                                   throughput[t]+=traffic['bandwidth'][i]
                                   numofTraffic[t]+=1
                                   
                           else:
                               if(successflag==0):
                                   trafficOutputPort[i]=-1;
                                   trafficSuccess[i]=0;
                               
               elif(traffic['operation'][i]==3):#drop
                   inputresourceFlag=wavehCheck(inputResource, infiber, inport, traffic['startWave'][i],traffic['bandwidth'][i])
                
                   if(inputresourceFlag==0):                                   
                       if(inportConnection[infiber-1][inport]==3):
                           
                           firstFit=True
                           ADresourceOccupy(infiber,inport,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],3,firstFit)
                           trafficOutputPort[i]=0;
                           trafficSuccess[i]=1;
                           updatFlag=1
                               
                           throughput[t]+=traffic['bandwidth'][i]
                           numofTraffic[t]+=1
                             
                       elif(inportConnection[infiber-1][inport]==0):
                           firstFit=False
                           inportConnection[infiber-1][inport]=3
                           portDes[infiber-1][inport]=0
                           setConnection(infiber,inport,3)
                           ADresourceOccupy(infiber,inport,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],3,firstFit)
                           trafficOutputPort[i]=0;
                           trafficSuccess[i]=1;
                           updatFlag=1
                           throughput[t]+=traffic['bandwidth'][i]
                           numofTraffic[t]+=1
                           
                       else:                 
                           trafficOutputPort[i]=-1;
                           trafficSuccess[i]=0;
                   else:
                       trafficOutputPort[i]=-1;
                       trafficSuccess[i]=-1;
                       numofTotalTraffic[t]-=1
                                                
               else:
                    trafficOutputPort[i]=-1;
                    trafficSuccess[i]=0;
                  
    
    
                           
    #--------------------------- traffic ends-------------------------#
            if(traffic['releaseTime'][i]==t):
                if(trafficSuccess[i]!=-1):
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
                        ADresourceRelease(outfiber,int((trafficOutputPort[i]-1)%coreNum),traffic['startWave'][i],traffic['bandwidth'][i],traffic['index'][i],1)
                        #trafficSuccess[i]=0
                    elif(traffic['operation'][i]==3):
                      
                        ADresourceRelease(infiber,inport,traffic['startWave'][i],traffic['bandwidth'][i],traffic['index'][i],3)
                        #trafficSuccess[i]=0
                    elif(traffic['operation'][i]==2):
                        #trafficSuccess[i]=0
                        if(trafficCost[i]==1):
                            bypass=True;
                        else:
                            bypass=False
                        PassResourceRelease(infiber, inport,outfiber,int((trafficOutputPort[i]-1)%coreNum), traffic['startWave'][i],traffic['bandwidth'][i],traffic['index'][i], bypass)
    
                                               
                            
                                   
    #---------------------------device update-------------------------#
        wss_1st_t=0;
        wss_2nd_t=0;
        wss_add_t=0;
        wss_drop_t=0;
        split_t=0;
    
        conn1stNum = 0;
        connSplitterNum = 0;
        conn2ndNum = 0;
        conndropNum = 0;
        connaddNum = 0;
        if(updatFlag==1):# device/connection change
            for node in range(nodeDegree):
                wssaddconFlag=0
                wssdropconFlag=0
                for c in range(coreNum):
                    wss1conFlag=0
                    wss2conFlag=0
                    splitterconFlag=0
                    if(WSS_addPorts[node][c]==0):
                        WSS_add[node][c]=0
                    if(WSS_dropPorts[node][c]==0):
                        WSS_drop[node][c]=0
                    for k in range(nodeDegree+1):
                        if(WSS1_ports[node][c][k]!=0):
                            wss1conFlag=1
                            break
                    for n in range(nodeDegree*coreNum+1):
                        if(WSS2_ports[node][c][n]!=0):
                            wss2conFlag=1
                            break
                    for m in range(nodeDegree):
                        for l in range(coreNum):
                            if(SplitterPorts[node][c][m][l]!=0):
                                splitterconFlag=1#Splitter 要重新设计才行
                                break
                        if(splitterconFlag==0):
                            Spliter[node][c][m]=0
    
                    if(wss1conFlag==0):
                        WSS_1st[node][c]=0
                    if(wss2conFlag==0):
                        WSS_2nd[node][c]=0
                        
    
        wss_1st_t=np.sum(WSS_1st==1)
        wss_2nd_t=np.sum(WSS_2nd==1)
        wss_drop_t=np.sum(WSS_drop==1)                  
        wss_add_t=np.sum(WSS_add==1)  
        split_t=np.sum(Spliter==1) 
    
        conn1stNum=np.sum(WSS1_ports!=0) 
        conn2ndNum=np.sum(WSS2_ports!=0)
        connSplitterNum=np.sum(SplitterPorts!=0)
        connaddNum=np.sum(WSS_addPorts!=0)
        conndropNum=np.sum(WSS_dropPorts!=0)
    
        numofWSS_1st[t] =wss_1st_t
        numofWSS_2nd[t] = wss_2nd_t
        numofSplitter[t] =split_t
        numofWSS_drop[t] =wss_drop_t
        numofWSS_add[t] = wss_add_t
        numofWSS_1stCon[t] = conn1stNum
        numofSplitterCon[t] = connSplitterNum
        numofWSS_2ndCon[t] = conn2ndNum
        numofWSS_dropCon[t] =conndropNum
        numofWSS_addCon[t] = connaddNum
        
        #numofTotalTraffic[t]=np.sum(trafficExist==1)
    
        #store the data and then change the splitter idea
    
    
    """-------------------traffic result output---------------"""
    Time=range(runTime)
    timeResult=pd.DataFrame({'time':Time,'totalTraffic':numofTotalTraffic,'trafficSuccessNum':numofTraffic,'throughput':throughput,'WSS1':numofWSS_1st,'WSS2': numofWSS_2nd,'split':numofSplitter,'WSSadd':numofWSS_add,'WSSdrop':numofWSS_drop})
    trafficResult=pd.DataFrame({'index':range(trafficNum+1),'outport':trafficOutputPort,'cost':trafficCost,'SuccessFlag':trafficSuccess})                                                        
                            
    timeResult.to_csv("timeResult.csv",index=False,sep=' ')                        
    trafficResult.to_csv("trafficResult.csv",index=False,sep=' ') 