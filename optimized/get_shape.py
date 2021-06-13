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

input_path1 = "../3dunet/dataset/ribfrac-train"
input_path2 = "../3dunet/dataset/ribfrac-val"
#output_path = "./dataset/train/images"
imglist1 = os.listdir(input_path1 + "/images")
imglist2 = os.listdir(input_path2 + "/images")
#zoom_size = (0.125,0.125,1)
name_list = []
shape_list = []

for imgname in imglist1: # Here only load label to get the transposed shape
    #imgadr = input_path1 + "/images/" + imgname
    labeladr = input_path1 + "/labels/" + imgname.replace("-image.nii.gz","-label.nii.gz")
    label = nib.load(labeladr).get_data()
    #label = np.transpose(label, (2,0,1))
    name_list.append(imgname.replace("-image.nii.gz","-label.nii.gz"))
    shape_list.append(label.shape[2])

for imgname in imglist2: # Here only load label to get the transposed shape
    #imgadr = input_path2 + "/images/" + imgname
    labeladr = input_path2 + "/labels/" + imgname.replace("-image.nii.gz","-label.nii.gz")
    label = nib.load(labeladr).get_data()
    #label = np.transpose(label, (2,0,1))
    name_list.append(imgname.replace("-image.nii.gz","-label.nii.gz"))
    shape_list.append(label.shape[2])

dataframe = pd.DataFrame({'name':name_list,'shape':shape_list})
dataframe.to_csv("./shape.csv",index=False,sep=',')


#    img = ndimage.zoom(img, zoom=zoom_size, order=3)
#    label = ndimage.zoom(label, zoom=zoom_size, order=0)
#    img = np.transpose(img, (2,0,1))
#    label = np.transpose(label, (2,0,1))
#
   



#
#
#imgadr = self.root + self.kind + 'images/' + self.imglist[index]
#labeladr = self.root + self.kind + 'labels/' + self.imglist[index].replace("-image.nii.gz","-label.nii.gz")
#
#img = nib.load(imgadr).get_data()
#label = nib.load(labeladr).get_data()
##label = sitk.ReadImage(labeladr)
##label = sitk.GetArrayFromImage(label)
##print(label.shape)
##print(img.shape)
#
##print("\n")
#
#img = ndimage.zoom(img, zoom=(0.125,0.125,1), order=3)
#label = ndimage.zoom(label, zoom=(0.125,0.125,1), order=0)#1
#img = np.transpose(img, (2,0,1))
#label = np.transpose(label, (2,0,1))
#
#
#id_code = []
#label_get_code(self.imglist[index].replace("-image.nii.gz",""), id_code)
#
#if self.transform is not None:
#    img = self.transform(img)
#
#img, label = common.random_crop_3d(img, label, [64,64,64])
#img = np.expand_dims(img, axis=0)
#
#for xx in range(len(label)):
#    for yy in range(len(label[0])):
#        #print("\n")
#        for zz in range(len(label[0][0])):
#            label[xx][yy][zz] = 0 if int((id_code[int(label[xx][yy][zz])][2])) == -1 or int((id_code[int(label[xx][yy][zz])][2])) == 0 else 1 #int((id_code[int(label[xx][yy][zz])][2]))
#
#
#
#
#
#
#def label_get_code(imgname, id_code):
#    info_path = './dataset/info.csv'
#    with open(info_path) as f:
#        csv_keyword = csv.reader(f)
#        keywords = []
#        for row in csv_keyword:
#            keywords.append(row)
#        for i in range(len(keywords)):
#            if imgname == keywords[i][0]:
#                id_code.append(keywords[i])
