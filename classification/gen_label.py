import cv2
from PIL import Image
import numpy as np
import pydensecrf.densecrf as dcrf
import multiprocessing
import os
from os.path import exists

palette = [0,0,0,  128,0,0,  0,128,0,  128,128,0,  0,0,128,  128,0,128,  0,128,128,  128,128,128,  
					 64,0,0,  192,0,0,  64,128,0,  192,128,0,  64,0,128,  192,0,128,  64,128,128,  192,128,128,  
					 0,64,0,  128,64,0,  0,192,0,  128,192,0,  0,64,128,  128,64,128,  0,192,128,  128,192,128,  
					 64,64,0,  192,64,0,  64,192,0, 192,192,0]

cats = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv']


data_path = './'
train_lst_path = data_path + 'data/train_cls.txt'
sal_path = data_path + 'data/saliency_aug/'
att_path = data_path + 'runs/exp2/attention/'
last_att_path = data_path + 'runs/exp1/attention/'
save_path = './pseudo_labels/'

if not exists(save_path):
	os.makedirs(save_path)
		
with open(train_lst_path) as f:
    lines = f.readlines()

def sal_divide(temp, bg, gt_cc, output_divide, bg_name):
    mask_part = temp==1
    check_flag = ((temp==1) & (bg > 1))   
    if np.any(check_flag) == False:
        gt_flt = gt_cc[temp==1]
        bin_result = np.bincount(gt_flt)
        label_ind = np.nonzero(bin_result)

        bin_flag = bin_result>0
        bin_sum = sum(bin_flag.astype(np.float))
        if bin_sum>0:
            if (bin_sum < 2) or (bin_sum < 3 and bin_result[0]>1):
                the_label = label_ind[-1].astype(np.uint8)
                #print(bg_name)
                fenzi = gt_flt[gt_flt == the_label[-1]]
                fenzi = len(fenzi)

                ratio = 1.0*fenzi/(len(gt_flt)+1e-8)
                if ratio > 0.95:
                    output_divide[mask_part] = the_label[-1]
                    

def gen_gt(index):
    line = lines[index]
    line = line[:-1]
    fields = line.split()
    name = fields[0]
    bg_name = sal_path + name + '.png'
    if not os.path.exists(bg_name):
        return
    sal = cv2.imread(bg_name, 0)
    sal_cc = sal.copy()            # this for cc process
    height, width = sal.shape
    gt = np.zeros((21, height, width), dtype=np.float32)
    added_gt = np.zeros((21, height, width), dtype=np.float32)                 
    added_gt[0] = 0.5                                                         
    sal = np.array(sal, dtype=np.float32)


    #begin boundary generation
    init_mask =  np.where(sal==255, 1.0, 0.0)
    kernel = np.ones((2, 2), np.uint8)   #ori=5
    last_mask = cv2.dilate(init_mask, kernel, iterations=1)
    flag = ((last_mask != 0) & (sal == 0))
    boundary_mask_global = np.where(flag, 0, 255)
    #end boundary generation
    
    # some thresholds. 
    conflict = 0.9
    fg_thr = 0.3
    # the below two values are used for generating uncertainty pixels
    bg_thr = 32
    att_thr = 0.8

    # use saliency map to provide background cues
    gt[0] = (1 - (sal / 255))
    init_gt = np.zeros((height, width), dtype=float) 
    sal_att = sal.copy()  
    
    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        # normalize attention to [0, 1] 
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        gt[cls+1] = att.copy()
        sal_att = np.maximum(sal_att, (att > att_thr) *255)
    

    gt_cc = gt.copy()                #for cc (connected component) process
    sal_att_cc  = sal_att.copy()     #for cc process
    # throw low confidence values for all classes
    gt[gt < fg_thr] = 0
    
    # conflict pixels with multiple confidence values
    bg = np.array(gt > conflict, dtype=np.uint8)  
    bg = np.sum(bg, axis=0)
    gt = gt.argmax(0).astype(np.uint8)
    gt[bg > 1] = 255
    
    # pixels regarded as background but confidence saliency values 
    bg = np.array(sal_att >= bg_thr, dtype=np.uint8) * np.array(gt == 0, dtype=np.uint8)
    gt[bg > 0] = 255  

    #POM
    for i in range(len(fields) - 1):
        k = i + 1
        cls = int(fields[k])
        att_name = last_att_path + name + '_' + str(cls) + '.png'
        if not exists(att_name):
            continue
        
        # normalize attention to [0, 1] 
        att = cv2.imread(att_name, 0)
        att = (att - np.min(att)) / (np.max(att) - np.min(att) + 1e-8)
        position = [gt==(cls+1)]

        temp = att[tuple(position)]
        if np.sum(temp)!=0: 
            flt_thr = np.median(temp) 
        else:                         
            position = [att > 0.3]
            if np.sum(position) != 0:
                temp = att[tuple(position)]
                temp_median = np.median(temp)  
                position = [att > temp_median]
                temp = att[tuple(position)]
                flt_thr = np.median(temp) 
            else:
                flt_thr = 1
        
        select_position = np.where(att > flt_thr, 1, 0)
        added_gt[cls+1] = select_position

    ignore = np.sum(added_gt, axis=0)
    added_gt = np.zeros((height, width), dtype=np.uint8)
    added_gt[ignore > 0.6] = 255                     # if there is a class, the background should be ignored

    flag = ((gt==0) & (added_gt == 255))

    gt = np.where(flag, 255, gt)

    # cc process begin

    # throw low confidence values for all classes
    gt_cc[gt_cc < fg_thr] = 0
    
    # conflict pixels with multiple confidence values
    bg = np.array(gt_cc > conflict, dtype=np.uint8)
    bg = np.sum(bg, axis=0)
    gt_cc = gt_cc.argmax(0).astype(np.uint8)
    gt_cc[bg > 1] = 255


    ret, binary = cv2.threshold(sal_cc, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)



    temp_1 = np.zeros((sal_cc.shape[0], sal_cc.shape[1]), np.uint8)
    output_divide = np.ones((sal_cc.shape[0], sal_cc.shape[1]), dtype=np.float32)*255
    for i in range(1, num_labels):
        cent = np.round(centroids[i]).astype(int)
        cent = cent[1],cent[0]

        mask = labels == i
        valid = np.array(mask, dtype=int).sum()
        ratio = float(valid) / float(height * width)
        if ratio > 0.01:
            
            temp_1[mask] = 1
            #left top
            temp_2 = np.zeros((sal_cc.shape[0], sal_cc.shape[1]), np.uint8)
            temp_2[:cent[0],:cent[1]] = 1
            temp = temp_1*temp_2
            sal_divide(temp, bg, gt_cc, output_divide, bg_name)
            

            #right top
            temp_2 = np.zeros((sal_cc.shape[0], sal_cc.shape[1]), np.uint8)
            temp_2[:cent[0],cent[1]:] = 1
            temp = temp_1*temp_2
            sal_divide(temp, bg, gt_cc, output_divide, bg_name)
        
            #left bottom
            temp_2 = np.zeros((sal_cc.shape[0], sal_cc.shape[1]), np.uint8)
            temp_2[cent[0]:,:cent[1]] = 1
            temp = temp_1*temp_2
            sal_divide(temp, bg, gt_cc, output_divide, bg_name)

            #right top
            temp_2 = np.zeros((sal_cc.shape[0], sal_cc.shape[1]), np.uint8)
            temp_2[cent[0]:,cent[1]:] = 1
            temp = temp_1*temp_2
            sal_divide(temp, bg, gt_cc, output_divide, bg_name)

    
    flag = (output_divide != 255)

    gt = np.where(flag, output_divide, gt)

    #boundary
    gt = np.where(boundary_mask_global==0, 0, gt)

    # we ignore the whole image for an image with a small ratio of semantic objects
    out = gt
    #out = output_divide
    valid = np.array((out > 0) & (out < 255), dtype=int).sum()
    ratio = float(valid) / float(height * width)
    if ratio < 0.01:
        out[...] = 255

    # output the proxy labels using the VOC12 label format
    out = Image.fromarray(out.astype(np.uint8), mode='P')
    out.putpalette(palette)
    out_name = save_path + name + '.png'
    out.save(out_name)

### Parallel Mode
pool = multiprocessing.Pool(processes=16)
pool.map(gen_gt, range(len(lines)))
#pool.map(gen_gt, range(100))
pool.close()
pool.join()

