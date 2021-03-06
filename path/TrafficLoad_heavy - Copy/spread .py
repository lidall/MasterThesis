
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib as plt
import numpy as np
import pandas as pd
from itertools import permutations


# In[2]:


def Spreadwave(wave_DF):
    waveList=[]
    df=wave_DF.sort_values(by=["waveUsage",'wave'],ascending=[True,True]) 
    df.reset_index(drop=True, inplace=True)#reset the index sequence
    #print(df)
    for i in range(len(df)):
        waveList.append(df["wave"][i])
        
    return waveList

def waveUpdate(Wave,Path,band):
    for i in range(band):
        wave_loc=Total_wave.index(Wave+i)
        wave_range[wave_loc]+=len(Path)
    #print(wave_range)
    #print(Total_wave)
    #print(randomseed)
    Wave_Infor={"wave":Total_wave,
               "waveUsage":wave_range
               }
    Wave_infor=pd.DataFrame(Wave_Infor)
    
    return Wave_infor
    


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



def bypassType(path):
    path_len=len(path)
    if(path_len==1):
        bypass_type=0
    elif(path_len==2):
        edgeLoc0=edges_list.index((path[0][0],path[0][1]))
        edgeLoc1=edges_list.index((path[1][0],path[1][1]))
        if(path[1] in Out_bypassEdges[edgeLoc0][path[0][2]] and path[0] in In_bypassEdges[edgeLoc1][path[1][2]] ) :
            bypass_type=2
        else:
            bypass_type=0
            
    elif(path_len==3):       
        edgeLoc0=edges_list.index((path[0][0],path[0][1]))
        edgeLoc1=edges_list.index((path[1][0],path[1][1]))
        edgeLoc2=edges_list.index((path[2][0],path[2][1]))
        if(path[1] in Out_bypassEdges[edgeLoc0][path[0][2]] and path[0] in In_bypassEdges[edgeLoc1][path[1][2]] ) :
            bypass_type=1
        elif(path[2] in Out_bypassEdges[edgeLoc1][path[1][2]] and path[1] in In_bypassEdges[edgeLoc2][path[2][2]] ):
            bypass_type=1
        else:
            bypass_type=0
                       
            
    return bypass_type
                
                
            


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


def CoreSetup():
    Nodes_list=list(G.nodes)
    #Link_bypass=[]
    for i in Nodes_list:
        neighborList=list(G.neighbors(i))
        #bypasscore_num=len(neighborList)-1# set the bypass core number within each fiber 
        for j in range(len(neighborList)):
            dis=1
            while(j+dis<len(neighborList)):
                input_loc=edges_list.index((neighborList[j],i))
                output_loc=edges_list.index((i,neighborList[j+dis]))
                successFlag=0
                for m in range(7):
                    if(len(bypassEdges[input_loc][m])==0):
                        for n in range(7):
                            if(len(bypassEdges[output_loc][n])==0):
                                bypassEdges[output_loc][n].append(((neighborList[j],i,m)))
                                bypassEdges[input_loc][m].append(((i,neighborList[j+dis],n)))
                                successFlag=1
                                break
                        if(successFlag==1):
                            break
                input_loc=edges_list.index((neighborList[j+dis],i))
                output_loc=edges_list.index((i,neighborList[j]))
                successFlag=0
                for m in range(7):
                    if(len(bypassEdges[input_loc][m])==0):
                        for n in range(7):
                            if(len(bypassEdges[output_loc][n])==0):
                                bypassEdges[output_loc][n].append(((neighborList[j+dis],i,m)))
                                bypassEdges[input_loc][m].append(((i,neighborList[j],n)))
                                successFlag=1
                                break
                        if(successFlag==1):
                            break
                                                                                                        
                dis+=1
    return;


# In[7]:


def edgeToLinks(path):
    pathlist=[]
    path_len=len(path)
    corePossib=7**path_len
    for i in range(corePossib):
        List=[]
        List.append((path[0]+(i%7,)))
        edgeLoc=edges_list.index(path[0])
        
        if(len(In_bypassEdges[edgeLoc][i%7])>0):
            List=[]
            continue
        for j in range(1,path_len):
            List.append((path[j]+((int(i/(7**j)%7)),)))
            edgeLoc0=edges_list.index(path[j-1])
            edgeLoc1=edges_list.index(path[j])
            if(len(Out_bypassEdges[edgeLoc0][List[j-1][2]])>0 and List[j] not in Out_bypassEdges[edgeLoc0][List[j-1][2]] ) :

                List=[]
                break
            if(len(In_bypassEdges[edgeLoc1][List[j][2]])>0 and List[j-1] not in In_bypassEdges[edgeLoc1][List[j][2]] ) :

                List=[]
                break
        edgeLoc=edges_list.index(path[path_len-1])
        if(len(Out_bypassEdges[edgeLoc][int(i/(7**(path_len-1)))])>0):            
            List=[]
            continue 
        if(len(List)>0):
            pathlist.append(List)    
    return pathlist


# In[8]:


def PathAlgorithm_1(PathList,hopsList,bandwidth,pathBandList,bypassList,wave,list1,list2):
    Max=np.zeros(len(pathBandList))
    Sum=np.zeros(len(pathBandList))
    for i in range(len(pathBandList)):
        #print (pathBandList[i][:,1] - pathBandList[i][:,0])
        #print(i)
        #print(PathList[0])
        Max[i]=max(pathBandList[i][:,1] - pathBandList[i][:,0])
        Sum[i]=sum(pathBandList[i][:,1] - pathBandList[i][:,0])
    
    infor={"place":range(len(PathList)),
           "bypass":bypassList,
           "bandwidth": Max,
           "hops": hopsList,
           "sum":Sum
         }
    
    Path_infor=pd.DataFrame(infor)     
    #print(Path_infor)
    df=Path_infor.sort_values(by=list1,ascending=list2) 
    df.reset_index(drop=True, inplace=True)#reset the index sequence
    #print(df)
    Path=[]
    
    for i in range(len(df)):
        successful_flag=0
        if(df["bandwidth"][i]>bandwidth):
            for j in pathBandList[df["place"][i]]:
                if(wave>=j[0] and wave+bandwidth<=j[1]):
                    Path=PathList[df["place"][i]]
                    traffic_feedback.append(df["bypass"][i])
                    successful_flag=1
                    break 
        if(successful_flag==1):
            break
    if(len(Path)==0):
        traffic_feedback.append(-1)
        #print(df)
    return Path


# In[9]:


def PathOccupy_percent():
    Bypass_2=[]
    Bypass_1=[]
    Bypass_0=[]
    for i in range(len(edges_list)):
        if(edges_list[i] in Traffic_SD_pairs):
            for j in range(len((SD_pathLink[i]))):
                percent=1-(sum(ALL_SD_bandwidth[i][j][:,1]-ALL_SD_bandwidth[i][j][:,0])-1)/160
                if(Bypass_Type[i][j]==2):
                    Bypass_2.append((percent))
                elif(Bypass_Type[i][j]==1):
                    Bypass_1.append((percent))
                elif(Bypass_Type[i][j]==0):
                    Bypass_0.append((percent))
    Bypass=[]
    Bypass.append(Bypass_0)
    Bypass.append(Bypass_1)
    
    Bypass.append(Bypass_2)
    
    return Bypass


# In[10]:


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
        


# In[11]:


def Path_percent():
    Bypass_2=[]
    Bypass_1=[]
    Bypass_0=[]
    for i in range(len(edges_list)):
        if(edges_list[i] in Traffic_SD_pairs):
            for j in range(len((SD_pathLink[i]))):
                #percent=1-(sum(ALL_SD_bandwidth[i][j][:,1]-ALL_SD_bandwidth[i][j][:,0])-1)/160
                if(Bypass_Type[i][j]==2):
                    Bypass_2.append((SD_pathLink[i][j]))
                elif(Bypass_Type[i][j]==1):
                    Bypass_1.append((SD_pathLink[i][j]))
                elif(Bypass_Type[i][j]==0):
                    Bypass_0.append((SD_pathLink[i][j]))
    Bypass=[]
    Bypass.append(Bypass_0)
    Bypass.append(Bypass_1)
    
    Bypass.append(Bypass_2)
    
    return Bypass


# In[12]:


TrafficCost=0
node_list=[0,1,2,3]
Traffic_SD_pairs=[(0,3),(0,1),(0,2),(1,2),(2,3),(1,3),(2,1)]
SR_pair=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(2,3),(3,2),(0,3),(3,0),(1,3),(3,1)]
edges_list=[(0,1),(1,0),(1,2),(2,1),(0,2),(2,0),(2,3),(3,2),(1,3),(3,1)]
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
SR_pathLink=[]
for i in range(len(SR_pair)):
    path0=[]
    for path in nx.all_simple_paths(G, source=SR_pair[i][0], target=SR_pair[i][1]):
        path_edges=[]            
        for j in range(len(path)-1):
            path_edges.append((path[j],path[j+1]))
        path0.append(path_edges)
    SD_path.append(path0)


CoreSetup() 
for i in range(len(bypassEdges)):
    Link=bypassEdges[i]
    for j in range(7):
        if(len(Link[j])>0):
            if(Link[j][0][0]==edges_list[i][1]):
                
                Out_bypassEdges[i][j]=Link[j]
            elif(Link[j][0][1]==edges_list[i][0]):
                In_bypassEdges[i][j]=Link[j]
                


# In[13]:


SD_pathLink=[]  
for i in range(len(SD_path)):
    Links=[]
    for j in range(len(SD_path[i])):
        Links=Links+edgeToLinks(SD_path[i][j])
        
    SD_pathLink.append(Links)


# In[14]:


Tag_range = list(permutations(range(3), 3))
Tag_list=["hops","bypass","sum"]
FalseorTrue=[True, True, False]
for i in range(8):
    k1=int(i%2)
    k2=int((i%4)/2)
    k3=int(i/4)
    FalseorTrue.append((k1,k2,k3))
    
print(FalseorTrue)

FalseTrueList=[False,True]


# In[46]:


Traffic=range(1,26)
traffic=[x*200 for x in Traffic]
Traffic_number=[1]+traffic


# In[ ]:


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
        list1=['hops', 'bypass', 'sum']
        list2= [True, False, True]
        #for i in range(len(Tag)):
            #list1.append(Tag_list[Tag[i]])
        #for i in range(len(choose)):
            #list2.append(FalseTrueList[choose[i]])
        Total_bandwidth=161
        Resource=np.zeros([4,4,7,161]).tolist()
        ALL_SD_bandwidth=[]
        MaxBand=[]
        Hops=[]
        Bypass_Type=[]
        for i in range(len(SD_pathLink)):
            Band=[]
            Max=[]
            Hhops=[]
            Bypass=[]
            for j in range(len(SD_pathLink[i])):
                Hhops.append(len(SD_pathLink[i][j]))
                b=pathBand(SD_pathLink[i][j])
                Max.append(max(b[:,1] - b[:,0]))
                Band.append(b)
                Bypass.append(bypassType(SD_pathLink[i][j]))
            ALL_SD_bandwidth.append(Band)
            MaxBand.append(Max)
            Hops.append(Hhops)
            Bypass_Type.append(Bypass)
        
        Total_wave=list(range(Total_bandwidth-1))
        wave_range=list(np.zeros(len(Total_wave)))
        randomseed=[]
        for i in range(len(Total_wave)):
            randomseed.append(np.random.randint(len(Total_wave)))
        Wave_Infor={"wave":Total_wave,
                   "waveUsage":wave_range
                   }
        Wave_infor=pd.DataFrame(Wave_Infor)

        
        
        # In[214]:
        
        
        Traffic=pd.read_csv(filename,sep=',',names=['S','D','band'])
        SuccessFlag=np.zeros(len(Traffic))
        BypassFlag=np.zeros(len(Traffic))
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
            traffic_Bypass=Bypass_Type[SR_pair.index(traffic)]
            wave=Spreadwave(Wave_infor)
            #print(wave)
            #print(Wave_infor)
            for j in range(len(wave)):
                PathFlag=0
                select_wave=wave[j]
                #print(select_wave)
                selectPath=PathAlgorithm_1(traffic_Path,traffic_Hops,band,traffic_Band,traffic_Bypass,select_wave,list1,list2)
                #print(selectPath)
                if(len(selectPath) and select_wave+band<=160):
                    Wave_infor=waveUpdate(select_wave,selectPath,band)
                    Pathloc=traffic_Path.index(selectPath)
                    pathBand_Ocuupy(SR_pair.index(traffic),Pathloc,select_wave,band,i)
                    PathFlag=1
                    traffic_wave.append(select_wave)
                    traffic_path.append(selectPath)            
                    pathUpdate()
                    if(PathFlag==1):
                        SuccessFlag[i]=1
                    break
                
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


print(Success_list)


# In[ ]:


print(Link_percentage())


# In[ ]:


occupy=0
for j in range(7):        
    
    for i in edges_list:
            #print(i)
            #print(list(np.nonzero(Resource[i[0]][i[1]][j])))
        k=np.array(Resource[i[0]][i[1]][j])
        S=np.count_nonzero(k)
        occupy+=S
print(occupy)
print(occupy/(7*10))


# In[ ]:


c={'success':Success_list,
   'length': Traffic_length,
   'band':Bandwidth_list
    
}

test=pd.DataFrame(c)
test.to_csv('spreadwavelength.csv')


# In[ ]:


print(Link_percentage())


# In[ ]:


[1.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 447.0, 486.0, 528.0, 570.0, 615.0, 652.0, 704.0, 710.0, 728.0, 793.0, 794.0, 871.0, 885.0, 895.0, 889.0, 925.0, 964.0, 930.0, 969.0, 990.0, 982.0, 1068.0, 977.0, 1009.0, 1048.0, 1059.0, 1077.0, 1050.0, 1102.0, 1013.0, 1135.0, 1062.0, 1125.0, 1056.0, 1124.0, 1103.0, 1062.0, 1118.0, 1056.0, 1108.0, 1067.0, 1071.0]


# In[16]:



print(Success_list)


# In[17]:


print(Tag_List[Success_list.index(max(Success_list))])
print(True_List[Success_list.index(max(Success_list))])


# In[20]:


print(max(Success_list))


# In[19]:


print(Link_percentage())


# In[52]:


import heapq


a=[1230.0, 1357.0, 1223.0, 1355.0, 1213.0, 1371.0, 1200.0, 1362.0, 1234.0, 1354.0, 1210.0, 1378.0, 1241.0, 1359.0, 1194.0, 1362.0, 1315.0, 1225.0, 1358.0, 1241.0, 1326.0, 1212.0, 1361.0, 1214.0, 1347.0, 1234.0, 1314.0, 1212.0, 1345.0, 1240.0, 1307.0, 1212.0, 1337.0, 1227.0, 1347.0, 1230.0, 1333.0, 1224.0, 1338.0, 1225.0, 1343.0, 1233.0, 1337.0, 1224.0, 1343.0, 1227.0, 1338.0, 1224.0]
tag=[['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'bypass', 'sum'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['hops', 'sum', 'bypass'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'hops', 'sum'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['bypass', 'sum', 'hops'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'hops', 'bypass'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops'], ['sum', 'bypass', 'hops']]
true=[[False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True], [False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True], [False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True], [False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True], [False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True], [False, False, False], [True, False, False], [False, True, False], [True, True, False], [False, False, True], [True, False, True], [False, True, True], [True, True, True]]
big_index=list(map(a.index,heapq.nlargest(5,a)))
print(len(true))
print(big_index)
for i in big_index:
    print(a[i])
    print(tag[i])
    print(true[i])


# In[43]:



m=0
n=0
p=0
q=0
for i in range(Traffic_num):
    if(SuccessFlag[i]==1):
        print(str(i)+' : '+str(Traffic['band'][i]))
        if(Traffic['band'][i]==1):
            m+=1
        elif(Traffic['band'][i]==2):
            n+=1
        elif(Traffic['band'][i]==4):
            p+=1 
        elif(Traffic['band'][i]==16):
            q+=1


# In[44]:


print(m)
print(n)
print(p)
print(q)

