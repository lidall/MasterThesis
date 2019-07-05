
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt;
#import sys
from pandas import DataFrame


# In[3]:


totalWave=321;
coreNum=7;
runTime=2
nodeDegree=3;
BypassNum=2;
updatFlag=0;








# In[6]:


#----------------functions---------------------#    
#-------------------functions--------------------------
def wavehCheck(Resource, fiber, port, wave,bandwidth):
    resourceFlag=0;
    for i in range (bandwidth):
        if(Resource[fiber-1][port][wave+i]!=0):
            resourceFlag=1;#once the bandwidth is occupied, then stop the session
            break
    return resourceFlag 

def ADresourceOccupy(fiber, port, wave,trafficIndex, bandwidth, operation, firstfit):
    #only consider the add and drop operation, pass operation will be defined later
    # firstfit is used to indicate the operation. If the input and output fiber has been connected, the it's firstfit. 
  
    if(operation==1):#add
        for b in range(bandwidth):
            #inputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
            outputResource[fiber-1][port][wave+b]=trafficIndex;
  
        if(firstfit==True):
            trafficCost[trafficIndex]=2;
        else:
             trafficCost[trafficIndex]=3;
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=trafficIndex;
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;
  
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
  
    elif(operation==3):
        #the fiber here should be infiber 
        for b in range(bandwidth):
            inputResource[fiber-1][port][wave+b]=0
            #outputResource[fiber-1][port][wave+bandwidth]=trafficIndex;

    return            
            
def PassResourceRelease(infiber, inport,outfiber,outport, wave, bandwidth, bypass):
    #firstfit means that the infiber is connected to outfiber in someway
    #the cost is influenced by the number of hardware used within the system.
    #bypass is used with cost 1 and general pass use 3 hardware to pass the data
    
    for b in range(bandwidth):
        inputResource[infiber-1][inport][wave+b]=0;
        #outputResource[infiber-1][inport][wave+bandwidth]=trafficIndex;
        #inputResource[outfiber-1][outport][wave+bandwidth]=trafficIndex;

   
    return        

    
    


# In[26]:


load=[]
load_number=range(1,201)
load_traffic=[]
load_success=[]
load_throughput=[]
for traffic_len in range(200):
    bypassNumInPort=np.zeros(nodeDegree)    
    bypassNumOutPort=np.zeros(nodeDegree)  
    load.append(load_number[traffic_len]*50)
    trafficNum=load[traffic_len]
    trafficCost=np.zeros(trafficNum+1)
    trafficOutputPort=np.zeros(trafficNum+1);# track the output port: 0 means drop 7-27 means to other ports successfully -1 means failure
    trafficSuccess=np.zeros(trafficNum+1);# 0 means fail and 1 means success
    trafficExist=np.zeros(trafficNum+1)
    filename='Load-'+str(load[traffic_len])+'.txt'
    trrafficc = np.loadtxt(filename)
    #print trrafficc
    trafficindex=np.arange(trafficNum).reshape(trafficNum,1)
    traffic_time=np.ones([trafficNum,2])
    Traffic=np.hstack((trafficindex,trrafficc,traffic_time))
    tra_columns=['index','preNode','nextNode','inPort','operation', 'startWave','bandwidth','createTime','releaseTime']
    traffic =pd.DataFrame(Traffic, columns=tra_columns,dtype=np.int64)
    
    
    
    
    
    numofTraffic =np.zeros(runTime);
    numofTotalTraffic=np.zeros(runTime)
    throughput=np.zeros(runTime);
    bypassNumInPort=np.zeros(nodeDegree)    
    bypassNumOutPort=np.zeros(nodeDegree)     
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

        #print Traffic.shape

    
    
    #algorithm
    infiber =0;
    outfiber=0
    inport =0

    for t in range(1,runTime):#start running



        print("--------------------")
        print (t)
        print ("--------------------")
        numofTraffic[t] = numofTraffic[t-1];
        numofTotalTraffic[t] = numofTotalTraffic[t-1];

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
               print (traffic['index'][i] )        
               trafficCost[i]=0

               trafficExist[i]=1
               firstFit=False
               bypass=False
                
               
                           #------------------add operation-----------------------#
               if(traffic['operation'][i]==1):
        
                    for c in range(coreNum):
                        resourceFlag=wavehCheck(outputResource, outfiber, c, traffic['startWave'][i],traffic['bandwidth'][i])
                        if(resourceFlag==0):
                            firstFit=True
                            ADresourceOccupy(outfiber,c,traffic['startWave'][i],traffic['index'][i],traffic['bandwidth'][i],1,firstFit)
                            trafficOutputPort[i]=outfiber*coreNum+c+1;
                            trafficSuccess[i]=1;
                            updatFlag=1
                            throughput[t]+=traffic['bandwidth'][i]
                            numofTraffic[t]+=1 # add the success traffic number and success traffic bandwidth
                            break
                    if(resourceFlag==1 and trafficCost[i]==0):  
                        trafficOutputPort[i]=-1;
                        trafficSuccess[i]=0;#failure situation


               #------------------pass operation-----------------------#                   
               elif(traffic['operation'][i]==2):
                    inputresourceFlag=wavehCheck(inputResource, infiber, inport, traffic['startWave'][i],traffic['bandwidth'][i])
                    if(inputresourceFlag==1):
                        trafficOutputPort[i]=-1;
                        trafficSuccess[i]=-1;
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
                       trafficSuccess[i]=-1;





    #--------------------------- traffic ends-------------------------#


       
    trafficResult=pd.DataFrame({'index':range(trafficNum+1),'outport':trafficOutputPort,'cost':trafficCost,'SuccessFlag':trafficSuccess})   
    print(str((traffic_len+1)*50))
    traffic_filename='Traffic-'+str((traffic_len+1)*50)+'.txt'
    trafficResult.to_csv(traffic_filename,index=False,sep='\t') 
    
    
    load_traffic.append(np.count_nonzero(trafficSuccess!=-1)-1)
    load_success.append(np.count_nonzero(trafficSuccess==1))
    load_throughput.append(throughput[1])
    
    print(load_traffic)


            
                
   


# In[27]:


print(load_success)


# In[17]:


Possible=[]
for i in range(len(load_success)):
    P=1-load_success[i]/load_traffic[i]
    Possible.append(P)

plt.figure()
plt.plot(load_traffic,Possible,label='T1')
plt.title("T10")
plt.ylabel('Acceptance Percentege')
plt.xlabel('Traffic Number')
plt.legend()
plt.savefig("T1_base.png")

    


# In[18]:


print(trafficSuccess)

