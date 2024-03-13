import numpy as np
from matplotlib import pyplot as plt
from random import randint
import math
global data_noise       #目標資料
data_noise = np.load('full_data_npy/datanpy/simulation_Brain_Web_noise3_rf0_5mm_all_images.npy',allow_pickle=True).tolist()

def Masked_Data(index):      #去腦殼
    global data_gth
    data_gth = np.load('full_data_npy/gtnpy/slices_'+str(index)+'_groundtruth.npy',allow_pickle=True)
    data_gth = data_gth.tolist()    #讀gth
    mask = data_gth['B1'].copy()    #取得腦殼mask
    mask += data_gth['B2']
    mask += data_gth['B3']
    for i in range(len(mask[:,0])):
        for j in range(len(mask[0,:])):
                if(mask[i,j]!=0):
                    mask[i,j] = 1

    data_noise_PD = data_noise['AI_PD_n3_rf0'][:,:,index]*mask     #原圖相乘，取得三種noise的腦殼     
    data_noise_T1 = data_noise['AI_T1_n3_rf0'][:,:,index]*mask
    data_noise_T2 = data_noise['AI_T2_n3_rf0'][:,:,index]*mask
    Masked_data = np.array([data_noise_PD,data_noise_T1,data_noise_T2])
    return Masked_data

def get_center():        #取center 陣列
    center_array = np.zeros((3,3))
    for i in range(3):
        center = np.array([randint(80,150),randint(80,150),randint(80,150)])
        center_array[i] = center
    return center_array

def get_distribution(center,data):
    distribution = data[0,:,:].copy()
    for i in range(len(data[0,:,0])):
        for j in range(len(data[0,0,:])):
            if(data[:,i,j].max()!=0):       ##最大值=0代表不是感興趣的值
                dis1 = np.sum(np.square(data[:,i,j]-center[0,:]))  #取得兩點距離
                dis2 = np.sum(np.square(data[:,i,j]-center[1,:]))
                dis3 = np.sum(np.square(data[:,i,j]-center[2,:]))
                if(min(dis1,dis2,dis3) == dis1):        #分配
                    distribution[i,j] = 1
                elif(min(dis1,dis2,dis3) == dis2):
                    distribution[i,j] = 2
                else:
                    distribution[i,j] = 3
    return distribution

def get_new_center(distribution,data):
    new_center_array = np.zeros((3,3))
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(distribution[:,0])):
        for j in range(len(distribution[0,:])):
            if(distribution[i,j]==1):
                new_center_array[0,:]+=data[:,i,j]
                count1+=1
            elif(distribution[i,j]==2):
                new_center_array[1,:]+=data[:,i,j]
                count2+=1
            elif(distribution[i,j]==3):
                new_center_array[2,:]+=data[:,i,j]
                count3+=1
    
    if(count1==0):
        new_center_array[0,:]/=1
    else:
        new_center_array[0,:]/=count1
    if(count2==0):
        new_center_array[1,:]/=1
    else:
        new_center_array[1,:]/=count2
    if(count3==0):
        new_center_array[2,:]/=1
    else:
        new_center_array[2,:]/=count3
    return new_center_array

def get_result(distribution,data):
    result = data.copy()
    for i in range(len(distribution[:,0])):
        for j in range(len(distribution[0,:])):
            if(distribution[i,j]==1):
                result[0,i,j] = 255
                result[1,i,j] = 0
                result[2,i,j] = 0
            elif(distribution[i,j]==2):
                result[0,i,j] = 0
                result[1,i,j] = 255
                result[2,i,j] = 0
            elif(distribution[i,j]==3):
                result[0,i,j] = 0
                result[1,i,j] = 0
                result[2,i,j] = 255
    return result
def get_IoU(result):
    IoU_List = []
    gth = data_gth.copy()
    for i in range(3):
        temp_score = []
        for j in range(3):
            intersection = np.logical_and(result[i,:,:],gth['B'+str(j+1)][:,:])
            union = np.logical_or(result[i,:,:],gth['B'+str(j+1)][:,:])
            temp_score.append((np.sum(intersection)/np.sum(union)))
        IoU_List.append(max(temp_score))
    return IoU_List
def Show_Animate(brain):
    plt.clf()
    plt.subplot(2,3,1)
    plt.imshow(brain[0,:,:])
    plt.subplot(2,3,2)
    plt.imshow(brain[1,:,:])
    plt.subplot(2,3,3)
    plt.imshow(brain[2,:,:])
    plt.subplot(2,3,4)
    plt.imshow(data_gth['B1'][:,:])
    plt.subplot(2,3,5)
    plt.imshow(data_gth['B2'][:,:])
    plt.subplot(2,3,6)
    plt.imshow(data_gth['B3'][:,:])
########main##########
for i in range(18):
    center = get_center()      #取得center的陣列，3個center
    brain = Masked_Data(i)
    old_center = center.copy()
    plt.ion()
    Show_Animate(brain)
    plt.pause(0.1)
    plt.show()
    while(True):
        distribution = get_distribution(old_center,brain)
        temp_result = get_result(distribution,brain)
        Show_Animate(temp_result)
        plt.pause(0.1)
        new_center = get_new_center(distribution,brain)
        if(0 in new_center):
            new_center = get_center()
        if(np.array_equal(new_center,old_center)):
            break
        else:
            old_center = new_center
    result = get_result(distribution,brain)
    IoU_List = get_IoU(result)
    print("B1:"+str(math.floor(IoU_List[0]*100))+"%   B2:" + str(math.floor(IoU_List[1]*100)) + "%   B3:"+str(math.floor(IoU_List[2]*100))+"%")
    Show_Animate(result)
plt.ioff()
plt.show()

