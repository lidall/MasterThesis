
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
import csv


# In[2]:


def Link_percentage():
   Edge=[]
   Link=[]
   for i in edges_list:
       Edge.append(str(i))
   for j in range(7):        
       occupy=[]
       for i in edges_list:
           #print(i)
           #print(list(np.nonzero(Resource[i[0]][i[1]][j])))
           k=np.array(Resource[i[0]][i[1]][j])
           per=np.count_nonzero(k)/160
           occupy.append(per)
       Link.append(occupy)
   df=pd.DataFrame(Link,columns=Edge)
   #print(df)
   #df.rename()
   return df
       


# In[3]:


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges      

def pathBand(path):
    path_len=len(path)
    #print(path)
    #print(path[0][0])
    resource=Resource[path[0][0]][path[0][1]][path[0][2]]
    for i in range(1,path_len):
        resource=list(map(lambda x: x[0]+x[1], zip(resource, Resource[path[i][0]][path[i][1]][path[i][2]])))
    pathbandList=zero_runs(resource) 
    
    return pathbandList


# In[4]:


# the functions used to select wavelength and path

def randomWave(wavelist):
    random_seed=np.random.randint(len(wavelist))
    return wavelist[random_seed]

def randomPath(PathList,hopsList,wave,bandwidth,pathBandList):
    Max=np.zeros(len(pathBandList))
    Sum=np.zeros(len(pathBandList))
    for i in range(len(pathBandList)):
        #print (pathBandList[i][:,1] - pathBandList[i][:,0])
        #print(i)
        #print(PathList[0])
        Max[i]=max(pathBandList[i][:,1] - pathBandList[i][:,0])
        Sum[i]=sum(pathBandList[i][:,1] - pathBandList[i][:,0])
 
    minHop=min(hopsList)
    for i in range(max(hopsList)-minHop):
        MinHop=minHop+i
        pathIndex=[index for index, value in enumerate(hopsList) if value==MinHop]
        Path=[]
        PathBand=[]
        availablePath=[]
        PathHops=[]
        for j in pathIndex:
            Path.append(PathList[j])
            PathBand.append(pathBandList[j])
        for k in range(len(PathBand)):
            successflag=0
            path_band=PathBand[k]
            if(Max[i]>bandwidth):
                if(wave+bandwidth>160):
                    break
                for j in path_band:
                    if(wave>=j[0] and wave+bandwidth<=j[1]):
                        successflag=1
                        break
            if(successflag==1):
                availablePath.append(Path[k])
        if(len(availablePath)):
            randomseed=np.random.randint(len(availablePath))
            return availablePath[randomseed]
            break
        
    if(len(availablePath)==0):
        return []
            
            
                
            
        
        
        
   


# In[4]:


def randomWave(wavelist):
    random_seed=np.random.randint(len(wavelist))
    return wavelist[random_seed]

def randomPath(PathList,hopsList,wave,bandwidth,pathBandList):
    Max=np.zeros(len(pathBandList))
    Sum=np.zeros(len(pathBandList))
    for i in range(len(pathBandList)):
        #print (pathBandList[i][:,1] - pathBandList[i][:,0])
        #print(i)
        #print(PathList[0])
        Max[i]=max(pathBandList[i][:,1] - pathBandList[i][:,0])
        Sum[i]=sum(pathBandList[i][:,1] - pathBandList[i][:,0])
        
    infor={"place":range(len(PathList)),
           "bandwidth": Max,
           "hops": hopsList,
           "sum":Sum
         }
    
    Path_infor=pd.DataFrame(infor)     
    #print(Path_infor)
    df=Path_infor.sort_values(by=['hops'],ascending=[True]) 
    df.reset_index(drop=True, inplace=True)#reset the index sequence
    #print(df)
    Path=[]
    for i in range(len(df)):
        successful_flag=0
        if(df["bandwidth"][i]>bandwidth):
            for j in pathBandList[df["place"][i]]:
                if(wave>=j[0] and wave+bandwidth<=j[1]):
                    Path=PathList[df["place"][i]]
                    #traffic_feedback.append(df["bypass"][i])
                    successful_flag=1
                    break 
        if(successful_flag==1):
            break
    #if(len(Path)==0):
        #traffic_feedback.append(-1)
        #print(df)
    return Path
            


# In[5]:


def pathBand_Ocuupy(loc1,loc2,wave1,band,index):
    path=SD_pathLink[loc1][loc2]
    pathBand=ALL_SD_bandwidth[loc1][loc2]
    for i in range(len(path)):
        for j in range(band):
            #print(path)
            #print(wave1)
            #print(Resource[path[i][0]][path[i][1]][path[i][2]][wave1+j])
            Resource[path[i][0]][path[i][1]][path[i][2]][wave1+j]+=index+1
            
    return

        
def pathUpdate():
    SR_bandwidth=[]
    #MaxBand=[]
    Hops=[]
    for i in range(len(SD_pathLink)):
        band=[]
        #Max=[]
        hops=[]        
        for j in range(len(SD_pathLink[i])):
            #cost.append(pathCost(SD_pathLink[i][j]))
            hops.append(len(SD_pathLink[i][j]))
            b=pathBand(SD_pathLink[i][j])
            #Max.append(max(b[:,1] - b[:,0]))
            band.append(b)            
        #SR_pathLink_cost.append(cost)
        SR_bandwidth.append(band)
        #MaxBand.append(Max)
        Hops.append(hops)
    global ALL_SD_bandwidth
    ALL_SD_bandwidth=SR_bandwidth
        
    return


# In[6]:



def edgeToLinks(path):
 pathlist=[]
 path_len=len(path)
 corePossib=7**path_len
 for i in range(corePossib):
     List=[]
     List.append((path[0]+(i%7,)))

     for j in range(1,path_len):
         List.append((path[j]+(int((i/(7**j)%7)),)))
     if(len(List)>0):
         pathlist.append(List)    
 return pathlist



TrafficCost=0
node_list=[0,1,2,3]
SR_pair=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(2,3),(3,2),(0,3),(3,0),(1,3),(3,1)]
edges_list=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(2,3),(3,2),(1,3),(3,1)]
"""
bypassEdges=[]
In_bypassEdges=[]
Out_bypassEdges=[]
for i in range(len(edges_list)):
 BY=[]
 BY0=[]
 BY1=[]
 
 for j in range(7):
     BY.append([])
     BY0.append([])
     BY1.append([])
 bypassEdges.append(BY)
 In_bypassEdges.append(BY0)
 Out_bypassEdges.append(BY1)
 
"""

#先对D=2的节点进行bypass设置，在对D=3的节点进行bypass设置
#每次设置都在该link中不存在

G=nx.DiGraph()
G.add_nodes_from(node_list)
G.add_edges_from(edges_list)
nx.draw(G)

#plt.show()


link_list=[] 

for k in range(len(edges_list)):
 link_list0=[]
 for i in range(7):
     link_list0.append(edges_list[k]+(i,)) 
 link_list.append(link_list0)

SD_path=[]

for i in range(len(SR_pair)):
 path0=[]
 for path in nx.all_simple_paths(G, source=SR_pair[i][0], target=SR_pair[i][1]):
     path_edges=[]            
     for j in range(len(path)-1):
         path_edges.append((path[j],path[j+1]))
     path0.append(path_edges)
 SD_path.append(path0)


     
SD_pathLink=[]  
for i in range(len(SD_path)):
 Links=[]
 for j in range(len(SD_path[i])):
     Links=Links+edgeToLinks(SD_path[i][j])
     
 SD_pathLink.append(Links)

 


# In[7]:


Traffic_SD_pairs=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(2,3),(3,2),(0,3),(3,0),(1,3),(3,1)]
Bandwidth=[1,2,4,16]
Traffic_num=3000
Traffic=[]
for i in range(Traffic_num):
    random_seed=np.random.randint(48)
    if(random_seed<24):
        S_band=Bandwidth[0]
    if(random_seed<36 and random_seed>23):
        S_band=Bandwidth[1]
    if(random_seed<48 and random_seed>35):
        S_band=Bandwidth[2]
    #if(random_seed<48 and random_seed>35):
        #S_band=Bandwidth[3]        
    Traffic.append(Traffic_SD_pairs[np.random.randint(len(Traffic_SD_pairs))]+(S_band,))
    
MyFile=open('traffic.csv','w')
writer=csv.writer(MyFile,lineterminator='\n')
for element in Traffic:
     writer.writerow(element)
MyFile.close()


# In[7]:


#prepare for the traffic here
Total_bandwidth=161
Resource=np.zeros([4,4,7,161]).tolist()
ALL_SD_bandwidth=[]
#MaxBand=[]
Hops=[]
for i in range(len(SD_pathLink)):
    Band=[]
    #Max=[]
    Hhops=[]
    for j in range(len(SD_pathLink[i])):
        Hhops.append(len(SD_pathLink[i][j]))
        b=pathBand(SD_pathLink[i][j])
        #Max.append(max(b[:,1] - b[:,0]))
        Band.append(b)
    ALL_SD_bandwidth.append(Band)
    #MaxBand.append(Max)
    Hops.append(Hhops)

    


# In[ ]:


Traffic=range(1,26)
traffic=[x*200 for x in Traffic]
Traffic_number=[1]+traffic
Success_list=[]
Tag_List=[]
True_List=[]
Traffic_length=[]
Bandwidth_list=[]
Link_status=[]
for tag_selection in range(len(Traffic_number)):
    for true_selection in range(1):
        print(tag_selection)
    
        Traffic_num=Traffic_number[tag_selection]
        filename='traffic_'+str(Traffic_num)+'.csv'
        #Tag=Tag_range[tag_selection]
        #choose=FalseorTrue[true_selection]
        list1=["hops","bypass","sum"]
        list2=[True,False,False]
        #for i in range(len(Tag)):
            #list1.append(Tag_list[Tag[i]])
        #for i in range(len(choose)):
            #list2.append(FalseTrueList[choose[i]])
        
        Total_bandwidth=161
        Resource=np.zeros([4,4,7,161]).tolist()
        ALL_SD_bandwidth=[]
        MaxBand=[]
        Hops=[]
        #Bypass_Type=[]
        for i in range(len(SD_pathLink)):
            Band=[]
            Max=[]
            Hhops=[]
            #Bypass=[]
            for j in range(len(SD_pathLink[i])):
                Hhops.append(len(SD_pathLink[i][j]))
                b=pathBand(SD_pathLink[i][j])
                Max.append(max(b[:,1] - b[:,0]))
                Band.append(b)
                #Bypass.append(bypassType(SD_pathLink[i][j]))
            ALL_SD_bandwidth.append(Band)
            MaxBand.append(Max)
            Hops.append(Hhops)
            #Bypass_Type.append(Bypass)
        

        
        
        # In[214]:
        
        
        Traffic=pd.read_csv(filename,sep=',',names=['S','D','band'])
        SuccessFlag=np.zeros(len(Traffic))
        #BypassFlag=np.zeros(len(Traffic))
        traffic_wave=[]
        traffic_path=[]
        traffic_feedback=[]
        Path_band_usage=[]
        Traffic_num=len(Traffic)
        for i in range(Traffic_num):
            traffic=(Traffic['S'][i],Traffic['D'][i])
            band=Traffic['band'][i]
            traffic_Path=SD_pathLink[SR_pair.index(traffic)]
            traffic_Hops=Hops[SR_pair.index(traffic)]
            traffic_Band=ALL_SD_bandwidth[SR_pair.index(traffic)]
            #traffic_Bypass=Bypass_Type[SR_pair.index(traffic)]
            wave=list(range(Total_bandwidth-1))
            while(len(wave)):
                PathFlag=0
                select_wave=randomWave(wave)
                #print(select_wave)
                selectPath=randomPath(traffic_Path,traffic_Hops,select_wave,band,traffic_Band)
                if(len(selectPath) and select_wave+band<=160):
                    Pathloc=traffic_Path.index(selectPath)
                    pathBand_Ocuupy(SR_pair.index(traffic),Pathloc,select_wave,band,i)
                    PathFlag=1
                    traffic_wave.append(select_wave)
                    traffic_path.append(selectPath)
                    pathUpdate()
                    if(PathFlag==1):
                        SuccessFlag[i]=1
                    break
                else:
                    wave.remove(select_wave)

                
                
        k=0
        for i in traffic_path:
            k+=len(i)
        Traffic_length.append(k/len(traffic_path))
        k=0
        for i in range(Traffic_num):
            if(SuccessFlag[i]==1):
                k+=Traffic['band'][i]
        Bandwidth_list.append(k)
        Link_status.append(Link_percentage())        
        Success_list.append(SuccessFlag.sum()) 


# In[ ]:


a=SuccessFlag.sum()
print(a)


# In[ ]:


print(Link_status)


# In[ ]:


c={'success':Success_list,
   'length': Traffic_length,
   'band':Bandwidth_list
    
}

test=pd.DataFrame(c)
test.to_csv('Contrast_nobypass.csv')

