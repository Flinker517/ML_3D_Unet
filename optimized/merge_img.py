import torch
import os
from torch.utils.data import Dataset
import nibabel as nib
from nibabel import nifti1
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import common
import csv
import pandas as pd
import multiprocessing

def label_get_2dlength(labelname):
    shape_csv = './shape.csv'
    with open(shape_csv) as f:
        csv_keyword = csv.reader(f)
        keywords = []
        for row in csv_keyword:
            keywords.append(row)
        for i in range(len(keywords)):
            if labelname == keywords[i][0]:
                return keywords[i][1]

      
def label_get_code(imgname, id_code):
    info_path = '../3dunet/dataset/info.csv'
    with open(info_path) as f:
        csv_keyword = csv.reader(f)
        keywords = []
        for row in csv_keyword:
            keywords.append(row)
        for i in range(len(keywords)):
            if imgname == keywords[i][0]:
                id_code.append(keywords[i])

def crop_0d(img, label, crop_arr, idx):
    return img[crop_arr[idx][0]:crop_arr[idx][1],:,:], label[crop_arr[idx][0]:crop_arr[idx][1],:,:]

def crop_plan(all_num, slice_num):
    '''
    all_num is the length of the num to be cropped.
    slice_num is the length of each slice.
    The function returns an array containing the beginning and ending of each slice.
    example:
    input: all_num = 90, slice_num = 20
    output:[[0,19],[20,39],[40,59],[60,79],[70,89]]
    Last element [70,89] must be the same length as slice_num.
    '''
    all_num = int(all_num)
    loop = all_num // slice_num
    #loop += 1
    arr = []
    for i in range(0, loop):
        bound = i*slice_num
        arr.append([bound, bound + slice_num - 1])
    if all_num % slice_num != 0:
        arr.append([all_num - slice_num, all_num - 1])
    return arr

def processing_img(imgname):
#for imgname in imglist: # Here only load label to get the transposed shape
    imgadr = input_path + "/images/" + imgname
    labeladr = input_path + "/labels/" + imgname.replace("-image.nii.gz","-label.nii.gz")
    rawlabel = nib.load(labeladr) # in order to use affine
    img = nib.load(imgadr).get_data()
    label = rawlabel.get_data()
    img = ndimage.zoom(img, zoom=zoom_size, order=3)
    label = ndimage.zoom(label, zoom=zoom_size, order=0)
    img = np.transpose(img, (2,0,1))
    label = np.transpose(label, (2,0,1))

    print(imgname)
    id_code = []
    label_get_code(imgname.replace("-image.nii.gz",""), id_code)
    d2length = label_get_2dlength(imgname.replace("-image.nii.gz","-label.nii.gz"))
    plans = crop_plan(d2length, 64)
    for idx in range(0, len(plans)):
        croppedimg, croppedlabel = crop_0d(img, label, plans, idx)
        for xx in range(len(croppedlabel)):
            for yy in range(len(croppedlabel[0])):
                for zz in range(len(croppedlabel[0][0])):
                    croppedlabel[xx][yy][zz] = 0 if int((id_code[int(croppedlabel[xx][yy][zz])][2])) == -1 or int((id_code[int(croppedlabel[xx][yy][zz])][2])) == 0 else 1 
        save_img = nib.Nifti1Image(croppedimg, rawlabel.affine)
        save_label = nib.Nifti1Image(croppedlabel, rawlabel.affine)
        subname = '-' + str(idx) + '-'
        nib.save(save_img, output_path + '/images/' + imgname.replace("-", subname))
        nib.save(save_label, output_path + '/labels/' + imgname.replace("-image.nii.gz", subname+"label.nii.gz"))


'''
input:[512,512,312],[512,512,213],...
output:[64,64,64]*5,[64,64,64]*4,...
input name: RibFrac1-image.nii.gz, RibFrac2-image.nii.gz, ... 
output name: RibFrac1-1-image.nii.gz, RibFrac1-2-image.nii.gz,..., RibFrac1-5image.nii.gz; RibFrac2-1-image.nii.gz, RibFrac2-2-image.nii.gz,..., RibFrac2-4image.nii.gz; ...

'''
#input_path = "./dataset-predicted64/ribfrac-val"
input_path = "./dataset-predicted/ribfrac-val"
output_path = "./dataset-predicted-merged/ribfrac-val"
#imglist = os.listdir(input_path + "/images")
#zoom_size = (0.125,0.125,1)
#name_list = []
#shape_list = []
#


imgname1 = "RibFrac"
for i in range(421,501):
    d2length = label_get_2dlength(imgname1 + str(i) + "-label.nii.gz")
    plans = crop_plan(d2length, 64)
    imgdata = np.empty(shape=(64,64,64), dtype = np.int16)
    print(imgname1 + str(i))
    firsttime = 1
    for j in range(0, len(plans)-1):
        imgname2 = imgname1 + str(i) + '-' + str(j) + '-label.nii.gz'
        img = nib.load(input_path + '/labels/' + imgname2).get_data()
        if firsttime == 1:
            imgdata = img
            firsttime = 0
        else:
            imgdata = np.concatenate((imgdata, img), axis=0, out=None)
    imgname2 = imgname1 + str(i) + '-' + str(len(plans)-1) + '-label.nii.gz'
    rawimg = nib.load(input_path + '/labels/' + imgname2)
    img = rawimg.get_data()
    start = plans[len(plans)-2][1] + 1
    end = plans[len(plans)-1][1] + 1
    remain = end - start
    imgdata = np.concatenate((imgdata, img[64 - remain:65,:,:]), axis = 0, out = None)
    save_img = nib.Nifti1Image(imgdata, rawimg.affine)
    nib.save(save_img, output_path + '/labels/' + imgname1 + str(i) + '-label.nii.gz')





#p = multiprocessing.Pool(8)
#
#for imgname in imglist: # Here only load label to get the transposed shape
#    #print(imgname)
#    p.apply_async(func = processing_img, args = (imgname,))
#
#p.close()
#p.join()
#

#dataframe = pd.DataFrame({'name':name_list,'shape':shape_list})
#dataframe.to_csv("./shape.csv",index=False,sep=',')

