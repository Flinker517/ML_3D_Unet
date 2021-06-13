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
    return img[crop_arr[idx][0]:crop_arr[idx][1]+1,:,:], label[crop_arr[idx][0]:crop_arr[idx][1]+1,:,:]

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
    #imgadr = input_path + "/images/" + imgname
    labeladr = input_path + "/labels/" + imgname.replace("-image.nii.gz","-label.nii.gz")
    rawlabel = nib.load(labeladr) # in order to use affine
#    img = nib.load(imgadr).get_data()
    label = rawlabel.get_data()
    #img = ndimage.zoom(img, zoom=zoom_size, order=3)
    #label = ndimage.zoom(label, zoom=zoom_size, order=0)
    #img = np.transpose(img, (2,0,1))
    #label = np.transpose(label, (2,0,1))

    print(imgname)
    id_code = []
    label_get_code(imgname.replace("-image.nii.gz",""), id_code)
#    d2length = label_get_2dlength(imgname.replace("-image.nii.gz","-label.nii.gz"))
#    plans = crop_plan(d2length, 64)
#    for idx in range(0, len(plans)):
#        croppedimg, croppedlabel = crop_0d(img, label, plans, idx)
    for xx in range(len(label)):
        for yy in range(len(label[0])):
            for zz in range(len(label[0][0])):
                label[xx][yy][zz] = 0 if int((id_code[int(label[xx][yy][zz])][2])) == -1 or int((id_code[int(label[xx][yy][zz])][2])) == 0 else 1 
#    save_img = nib.Nifti1Image(img, rawlabel.affine)
    save_label = nib.Nifti1Image(label, rawlabel.affine)
#    subname = '-' + str(idx) + '-'
#    nib.save(save_img, output_path + '/images/' + imgname)
    nib.save(save_label, output_path + '/labels/' + imgname.replace("-image.nii.gz", "-label.nii.gz"))


'''
input:[512,512,312],[512,512,213],...
output:[64,64,64]*5,[64,64,64]*4,...
input name: RibFrac1-image.nii.gz, RibFrac2-image.nii.gz, ... 
output name: RibFrac1-1-image.nii.gz, RibFrac1-2-image.nii.gz,..., RibFrac1-5image.nii.gz; RibFrac2-1-image.nii.gz, RibFrac2-2-image.nii.gz,..., RibFrac2-4image.nii.gz; ...

'''
'''
Only resize and save the img for comparing.(64*64*???)

'''

input_path = "../3dunet/dataset/ribfrac-train"
output_path = "./dataset-change-label-only/ribfrac-train"
imglist = os.listdir(input_path + "/images")
zoom_size = (0.125,0.125,1)
name_list = []
shape_list = []

p = multiprocessing.Pool(40)

for imgname in imglist: # Here only load label to get the transposed shape
    #print(imgname)
    p.apply_async(func = processing_img, args = (imgname,))

p.close()
p.join()


#dataframe = pd.DataFrame({'name':name_list,'shape':shape_list})
#dataframe.to_csv("./shape.csv",index=False,sep=',')

