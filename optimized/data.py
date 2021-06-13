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

class data1(Dataset):
    def __init__(self, root, split = 'train', transform=None):
        self.root = root
        self.transform = transform
        self.split = split
        if self.split == 'train':
            self.kind = '/ribfrac-train/'
            self.imglist = os.listdir(self.root + self.kind + "images")
            self.labellist = os.listdir(self.root + self.kind + "labels")
        if self.split == 'val':
            self.kind = '/ribfrac-val/'
            self.imglist = os.listdir(self.root + self.kind + "images")
            self.labellist = os.listdir(self.root + self.kind + "labels")
        if self.split == 'test':
            self.kind = '/ribfrac-test/'
            self.imglist = os.listdir(self.root + self.kind + "images")
            self.labellist = os.listdir(self.root + self.kind + "labels")

    def __getitem__(self, index):
        imgadr = self.root + self.kind + 'images/' + self.imglist[index]
        labeladr = self.root + self.kind + 'labels/' + self.imglist[index].replace("-image.nii.gz","-label.nii.gz")
        
        rawimg = nib.load(imgadr)
        affine = rawimg.affine
        img = rawimg.get_data()
        #label = nib.load(labeladr).get_data()
        #label = sitk.ReadImage(labeladr)
        #label = sitk.GetArrayFromImage(label)
        #print(label.shape)
        #print(img.shape)
        
        #print("\n")

        #img = ndimage.zoom(img, zoom=(0.125,0.125,1), order=3)
        #label = ndimage.zoom(label, zoom=(0.125,0.125,1), order=0)#1
        #img = np.transpose(img, (2,0,1))
        #label = np.transpose(label, (2,0,1))


        #id_code = []
        #label_get_code(self.imglist[index].replace("-image.nii.gz",""), id_code)
		
        if self.transform is not None:
            img = self.transform(img)

        #img, label = common.random_crop_3d(img, label, [64,64,64])
        #img = np.expand_dims(img, axis=0)
        #
        #for xx in range(len(label)):
        #    for yy in range(len(label[0])):
        #        #print("\n")
        #        for zz in range(len(label[0][0])):
        #            label[xx][yy][zz] = 0 if int((id_code[int(label[xx][yy][zz])][2])) == -1 or int((id_code[int(label[xx][yy][zz])][2])) == 0 else 1 #int((id_code[int(label[xx][yy][zz])][2]))
        
        #print(img.shape)
        #print(label.shape)

        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img), self.imglist[index], affine

    def __len__(self):
        return len(self.imglist)



def label_get_code(imgname, id_code):
    info_path = './dataset/info.csv'
    with open(info_path) as f:
        csv_keyword = csv.reader(f)
        keywords = []
        for row in csv_keyword:
            keywords.append(row)
        for i in range(len(keywords)):
            if imgname == keywords[i][0]:
                id_code.append(keywords[i])
