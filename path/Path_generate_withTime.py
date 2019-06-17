#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:01:48 2019

@author: Lida
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:43:49 2019

@author: Lida
"""

import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
from time import time



'''--------------functions----------------------'''
def edgeToLinks(path):
    pathlist=[]
    path_len=len(path)
    corePossib=7**path_len
    for i in range(corePossib):
        List=[]
        List.append((path[0]+(i%7,)))
        if(i%7 ==bypass_core[path[0][0]][0] or i%7==bypass_core[path[0][0]][1]):
            List=[]
            continue
        for j in range(1,path_len):
            List.append((path[j]+(i/(7**j),)))
            if((List[j-1][2]==bypass_core[List[j-1][1]][1] or(List[j-1][2]==bypass_core[List[j-1][1]][0] ) or (List[j][2]==bypass_core[List[j-1][1]][1] or(List[j][2]==bypass_core[List[j-1][1]][0] ))) and List[j-1][2]!=List[j][2] ):

                List=[]
                break
        if((i/(7**(path_len-1))==bypass_core[path[path_len-1][1]][1] or (i/(7**(path_len-1))==bypass_core[path[path_len-1][1]][0]))):            
            List=[]
            continue
        
        
        if(len(List)>0):
            pathlist.append(List)    
    return pathlist
    
def pathCost(path):
    # the length of single path
    path_len=len(path)
    IN=[0]*path_len
    OUT=[0]*path_len
    cost=0
    for i in range(path_len):
        IN[i]=InNode[path[i][0]][path[i][1]][path[i][2]]
        OUT[i]=OutNode[path[i][0]][path[i][1]][path[i][2]]
        if((i==0 and IN[i]==0)):
            cost+=1 # it is the WSS_ add cor WSS_drop cost
        if((i==path_len-1 and OUT[i]==0)):
            cost+=1
        if(IN[i]==0):
            cost+=1 # WSS_2 cost of the input node of the edge
        if(OUT[i]==0):
            cost+=1 # WSS_1 cost of the output node of the edge
        if(i>0 and linkMap[path[i-1][0]*21+path[i-1][1]*7+path[i-1][2]][path[i][0]*3+path[i][1]]==0):# splitter 改成映射形式
            cost+=1 # Splitter_WSS cost between two edges
    
    return cost


            
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges      

def pathBand(path):
    path_len=len(path)
    resource=Resource[path[0][0]][path[0][1]][path[0][2]]
    for i in range(1,path_len):
        resource=list(map(lambda x: x[0]+x[1], zip(resource, Resource[path[i][0]][path[i][1]][path[i][2]])))
    pathbandList=zero_runs(resource) 
    
    return pathbandList

def pathCost_Occupy(path,index):
    path_len=len(path)
    for i in range(path_len):
        InNode[path[i][0]][path[i][1]][path[i][2]]+=index+1
        OutNode[path[i][0]][path[i][1]][path[i][2]]+=index+1
        
        if(i>0 and path[i][2]!=bypass_core[path[i][0]][0] and path[i][2]!=bypass_core[path[i][0]][1]):
            linkMap[path[i-1][0]*21+path[i-1][1]*7+path[i-1][2]][path[i][0]*3+path[i][1]]+=index+1
        
                    
    return
def pathCost_Release(path,index):
    path_len=len(path)
    for i in range(path_len):
        InNode[path[i][0]][path[i][1]][path[i][2]]-=index+1
        OutNode[path[i][0]][path[i][1]][path[i][2]]-=index+1
        
        if(i>0 and path[i][2]!=bypass_core[path[i][0]][0] and path[i][2]!=bypass_core[path[i][0]][1]):
            linkMap[path[i-1][0]*21+path[i-1][1]*7+path[i-1][2]][path[i][0]*3+path[i][1]]-=index+1
        
                    
    return

def pathBand_Ocuupy(loc1,loc2,band,index):
    path=SR_pathLink[loc1][loc2]
    pathBand=ALL_SR_bandwidth[loc1][loc2]
    for i in pathBand:
        
        if ((i[1] - i[0])>band):
            startwave=i[0]
            break
    for i in range(len(path)):
        for j in range(band):
            Resource[path[i][0]][path[i][1]][path[i][2]][startwave+j]+=index+1
            
    return


def pathBand_Release(loc1,loc2,band,index):
    path=SR_pathLink[loc1][loc2]
    pathBand=ALL_SR_bandwidth[loc1][loc2]
    for i in pathBand:
        
        if ((i[1] - i[0])>band):
            startwave=i[0]
            break
    for i in range(len(path)):
        for j in range(band):
            Resource[path[i][0]][path[i][1]][path[i][2]][startwave+j]-=index+1
            
    return
        
def pathUpdate():
    SR_pathLink_cost=[]
    SR_bandwidth=[]
    MaxBand=[]
    Hops=[]
    for i in range(len(SR_pathLink)):
        cost=[]
        band=[]
        Max=[]
        hops=[]        
        for j in range(len(SR_pathLink[i])):
            cost.append(pathCost(SR_pathLink[i][j]))
            hops.append(len(SR_pathLink[i][j]))
            b=pathBand(SR_pathLink[i][j])
            Max.append(max(b[:,1] - b[:,0]))
            band.append(b)            
        SR_pathLink_cost.append(cost)
        SR_bandwidth.append(band)
        MaxBand.append(Max)
        Hops.append(hops)
    S_D_Information=[]
    global ALL_SR_bandwidth
    ALL_SR_bandwidth=SR_bandwidth
    for i in range(len(SR_pathLink)):
    
        SD_Infor={"place":range(len(SR_pathLink[i])),
                  "Cost":SR_pathLink_cost[i],
                  "Bandwidth":MaxBand[i],
                  "Hops":Hops[i]
                  
            }
        df = pd.DataFrame(SD_Infor, columns=['place', 'Cost', 'Bandwidth','Hops'])
        df=df.sort_values(by=["Cost","Bandwidth",'Hops'],ascending=[False,False,False])  
        df.reset_index(drop=True, inplace=True)#reset the index sequence
        S_D_Information.append(df)
        
    return S_D_Information
    
    
    
    
    

'''--------------------- state the nodes and the edges-----------'''
TrafficCost=0
node_list=[0,1,2]
SR_pair=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)]
edges_list=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0)]
bypass_core=[[0,1],[2,3],[4,5]]

# try to find all possible links between two nodes
link_list=[] 

for k in range(len(edges_list)):
    link_list0=[]
    for i in range(7):
        link_list0.append(edges_list[k]+(i,)) 
    link_list.append(link_list0)

'''------------generate the network Graph and find out all possible paths -----------'''

G=nx.DiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(edges_list)

S_R_path=[]
SR_pathLink=[]
for i in range(len(SR_pair)):
    path0=[]
    for path in nx.all_simple_paths(G, source=SR_pair[i][0], target=SR_pair[i][1]):
        path_edges=[]            
        for j in range(len(path)-1):
            path_edges.append((path[j],path[j+1]))
        path0.append(path_edges)
    S_R_path.append(path0)

for i in range(len(S_R_path)):
    link=[]
    for j in range(len(S_R_path[i])):
        t=edgeToLinks(S_R_path[i][j])        
        link=link+edgeToLinks(S_R_path[i][j])
    SR_pathLink.append(link)#link (a,b,c) means from a to b via core c
# now we get all the possible links that could convey data form S to D 

'''-----------find the edge indexs of all paths-------------'''
SR_link_index=[]
for i in range(len(SR_pathLink)):
    S_D=[]
    for j in range(len(SR_pathLink[i])):
        path_locator=[]
        for m in range(len(SR_pathLink[i][j])):
            location=edges_list.index((SR_pathLink[i][j][m][0],SR_pathLink[i][j][m][1]))
            path_locator.append((location,SR_pathLink[i][j][m][2]))        
        S_D.append(path_locator)
    SR_link_index.append(S_D) 
    
    
'''-------------construct the In and Out filter that indicates the cost--------'''
Node=np.zeros([2,3,3,7])
for m in range(2):
    for i in range(3):
        for j in range (3):
            for k in range(7):
                if(i==j):
                    Node[m][i][j][k]=-2
                if (k==bypass_core[i][0] or k==bypass_core[i][1]):                    
                    Node[0][i][j][k]=-1
                if(k==bypass_core[j][0] or k==bypass_core[j][1] ):
                    Node[1][i][j][k]=-1

InNode=Node[0].tolist()
OutNode=Node[1].tolist()

linkMap=np.zeros([3*3*7,3*3])
for i in range(3*3*7):
    for j in range(3*3):
        innode1=i/21
        outnode1=(i%21)/7
        core1=(i%21)%7
       
        innode2=j/3
        outnode2=j%3
        if(innode1==outnode1):
            linkMap[i][j]=-2
        if(innode2==outnode2):
            linkMap[i][j]=-2
        if(core1==bypass_core[outnode1][0] or core1==bypass_core[outnode1][1]):
            linkMap[i][j]=-1
        if(core1==bypass_core[innode2][0] or core1==bypass_core[innode2][1]):
            linkMap[i][j]=-1
 

       

SR_pathLink_cost=[]
for i in range(len(SR_pathLink)):
    cost=[]
    for j in range(len(SR_pathLink[i])):
        cost.append(pathCost(SR_pathLink[i][j]))
    SR_pathLink_cost.append(cost)      
                   
# now the code could calculate the cost for each path
'''-------------construct the path bandwidth that indicates the bandwidth avialable in the specific path--------'''
Resource=np.zeros([3,3,7,321]).tolist()
ALL_SR_bandwidth=[]
MaxBand=[]
Hops=[]
for i in range(len(SR_pathLink)):
    band=[]
    Max=[]
    hops=[]
    for j in range(len(SR_pathLink[i])):
        hops.append(len(SR_pathLink[i][j]))
        b=pathBand(SR_pathLink[i][j])
        Max.append(max(b[:,1] - b[:,0]))
        band.append(b)
    ALL_SR_bandwidth.append(band)
    MaxBand.append(Max)
    Hops.append(hops)
    
    
S_D_Information=[]
for i in range(len(SR_pathLink)):

    SD_Infor={"place":range(len(SR_pathLink[i])),
              "Cost":SR_pathLink_cost[i],
              "Bandwidth":MaxBand[i],
              "Hops":Hops[i]
              
        }
    df = pd.DataFrame(SD_Infor, columns=['place', 'Cost', 'Bandwidth','Hops'])
    df=df.sort_values(by=["Cost","Bandwidth",'Hops'],ascending=[False,False,False])  
    df.reset_index(drop=True, inplace=True)#reset the index sequence
    S_D_Information.append(df)

'''---------------------------traffic generation--------------'''
trafficNum=20
Traffic=[]
Band=[]
Creat_Time=[]
End_Time=[]
while(len(Traffic)<trafficNum):
    a=np.random.randint(0,3)
    b=np.random.randint(0,3)
    TC=np.random.randint(0,1000)#create time
    TE=np.random.randint(TC+1,1000+1)# end time
    if(a!=b):
        Traffic.append((a,b))
        Band.append(np.random.randint(1,20))
        End_Time.append(TE)
        Creat_Time.append(TC)
Trafficdf={"SD":Traffic,
           "Bandwidth":Band,
           "Create":Creat_Time,
           "End":End_Time,
           "Index":range(trafficNum)
                   }
TrafficDF=pd.DataFrame(Trafficdf, columns=['Index','SD', 'Bandwidth','Create','End'])   
TrafficDF=TrafficDF.sort_values(by=['Create','Bandwidth'],ascending=[True,False]) 
TrafficDF.reset_index(drop=True, inplace=True)

'''---------------timeTest-------------'''
TrafficSuccess=np.zeros(trafficNum)
Traffic_path=np.zeros(trafficNum).tolist()
UpdateFlag=0
SuccessFlag=0
for t in range(1000):
    traffic_create=TrafficDF.loc[TrafficDF['Create']==t]
    traffic_end=TrafficDF.loc[TrafficDF['End']==t]
    traffic_create.reset_index(drop=True, inplace=True)
    traffic_end.reset_index(drop=True, inplace=True)
    for i in range(len(traffic_create)):
        traffic_SR=traffic_create['SD'][i]
        traffic_Band=traffic_create['Bandwidth'][i]
        SR_path=edges_list.index(traffic_SR)
        if(UpdateFlag==1):
            S_D_Information=pathUpdate()
        for j in range(len(SR_pathLink[0])-1,-1,-1):
            if(S_D_Information[SR_path]['Bandwidth'][j]>traffic_Band):
                TrafficCost+=S_D_Information[SR_path]['Cost'][j]
                TrafficSuccess[traffic_create['Index'][i]]=1
                UpdateFlag=1
                SuccessFlag=1
                path=SR_pathLink[SR_path][S_D_Information[SR_path]['place'][j]]
                Traffic_path[traffic_create['Index'][i]]=path
                pathCost_Occupy(path,traffic_create['Index'][i])
                pathBand_Ocuupy(SR_path,S_D_Information[SR_path]['place'][j],traffic_Band,traffic_create['Index'][i])
                break
        if(SuccessFlag==0):
            TrafficSuccess.append(0)
    
    for k in range(len(traffic_end)):# 这个算法是有问题的，应该按照原来的link来消除，而不是在计算
        traffic_SR=traffic_end['SD'][k]
        traffic_Band=traffic_end['Bandwidth'][k]
        SR_path=edges_list.index(traffic_SR)
        if(UpdateFlag==1):
            S_D_Information=pathUpdate()
        if(TrafficSuccess[traffic_end['Index'][k]]==1):
            path=Traffic_path[traffic_end['Index'][k]]
            pathCost_Release(path,traffic_end['Index'][k])
            pathBand_Release(SR_path,S_D_Information[SR_path]['place'][j],traffic_Band,traffic_end['Index'][k])
            UpdateFlag=1
           
        
        
            

        

"""
'''---------------TEST---------------'''
TrafficSuccess=np.zeros(trafficNum)
Traffic_path=[]
UpdateFlag=0
for i in range(trafficNum):
    traffic_SR=TrafficDF['SD'][i]
    traffic_Band=TrafficDF['Bandwidth'][i]
    SR_path=edges_list.index(traffic_SR)
    if(UpdateFlag==1):
        S_D_Information=pathUpdate()

                
    for j in range(len(SR_pathLink[0])-1,-1,-1):
        if(S_D_Information[SR_path]['Bandwidth'][j]>traffic_Band):
            TrafficCost+=S_D_Information[SR_path]['Cost'][j]
            TrafficSuccess[i]=1
            UpdateFlag=1
            path=SR_pathLink[SR_path][S_D_Information[SR_path]['place'][j]]
            Traffic_path.append(path)
            pathCost_Occupy(path)
            pathBand_Ocuupy(SR_path,S_D_Information[SR_path]['place'][j],traffic_Band,i)
            break

            
        
"""

         