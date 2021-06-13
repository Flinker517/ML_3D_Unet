import numpy as np
import pandas as pd
from tqdm import trange
import os
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import unet2, logger, metrics, lossfuc, common, data
import nibabel as nib

#def train(model, train_loader, optimizer, criterion, device):
#    model.train()
#    train_loss = metrics.LossAverage()
#    train_dice = metrics.DiceAverage()
#    for batch_idx, (inputs, targets) in enumerate(train_loader):
#        inputs = inputs.float().to(device)
#        #print("inputs shape:", inputs.shape)
#        #print("targets shape1:", targets.shape)
#        targets = targets.long()
#        targets = common.to_one_hot_3d(targets)
#        #print("targets shape2:", targets.shape)
#        targets = targets.to(device)
#        outputs = model(inputs.to(device))
#        optimizer.zero_grad()
#        print(outputs[0,:,0,0,0])
#        #print("outputs shape:", outputs.shape)
#        #print("targets shape3:", targets.shape)
#        loss = criterion(outputs, targets)
#        loss.backward()
#        optimizer.step()
#
#        train_loss.update(loss.item(),inputs.size(0))
#        train_dice.update(outputs, targets)
#        print(loss)
#
#    return OrderedDict({'Train Loss': train_loss.avg, 'Train dice0': train_dice.avg[0],
#                       'Train dice1': train_dice.avg[1]})#,'Train dice2': train_dice.avg[2],
##                       'Train dice3': train_dice.avg[3],'Train dice4': train_dice.avg[4],
##                       'Train dice5': train_dice.avg[5]})
#
#
#def val(model, val_loader, optimizer, dir_path, epoch, criterion, device):
#    model.eval()
#    val_loss = metrics.LossAverage()
#    val_dice = metrics.DiceAverage()
#    with torch.no_grad():
#        for batch_idx, (inputs, targets) in enumerate(val_loader):
#            inputs = inputs.float().to(device)
#            targets = targets.long()
#            targets = common.to_one_hot_3d(targets)
#            targets = targets.to(device)
#            outputs = model(inputs.to(device))
#            loss=criterion(outputs, targets)
#            val_loss.update(loss.item(),inputs.size(0))
#            val_dice.update(outputs, targets)
#
#    state = {'net': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
#    path = os.path.join(dir_path, 'ckpt_%d_loss_%.5f.pth' % (epoch, loss))
#    torch.save(state, path)
#
#    return OrderedDict({'Val Loss': val_loss.avg, 'Val dice0': val_dice.avg[0],
#                        'Val dice1': val_dice.avg[1]})#,'Val dice2': val_dice.avg[2],
##                        'Val dice3': val_dice.avg[3],'Val dice4': val_dice.avg[4],
##                        'Val dice5': val_dice.avg[5]})
#
#
#
#def test(model, data_loader):
#    model.eval()
#    val_loss = metrics.LossAverage()
#    val_dice = metrics.DiceAverage()
#    with torch.no_grad():
#        for batch_idx, (inputs, targets) in enumerate(data_loader):
#            inputs = inputs.float().to(device)
#            targets = targets.float().to(device)
#            outputs = model(inputs.to(device))
#            loss=criterion(outputs, targets)
#            val_loss.update(loss.item(),data.size(0))
#            val_dice.update(outputs, targets)
#        print('test Loss: %.5f, Val dice0: %.5f, Val dice1: %.5f, Val dice2: %.5f' %(val_loss.avg, val_dice.avg[0], val_dice.avg[1], val_dice.avg[2]))
#
#
#input_root = './dataset'
#output_root = './output'
#if not os.path.exists(output_root):
#    os.mkdir(output_root)
#
        
#start_epoch = 0
#end_epoch = 50 
#batch_size = 32
#val_auc_list = []
#early_stop = 5
#
#print('Load data...')
## transform = 
#train_dataset = data.data1(root=input_root, split = 'train')
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=40)
#val_dataset = data.data1(root=input_root, split='val')
#val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=40)
## test_dataset = data1(root=input_root, split='test')
## test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#                                             
#print('train model...')
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = unet2.my3dunet(1,2).to(device)
#
## model = nn.DataParallel(model, device_ids=[0,1])
## model = model.module()
## torch.backends.cudnn.enabled = False
## model = torch.nn.DataParallel(model)
#
#optimizer = optim.Adam(model.parameters(), lr=0.001)
##criterion = lossfuc.DiceLoss(weight=np.array([0.2,0.3,0.5]))
#criterion = lossfuc.DiceLoss(weight=np.array([0.1,0.9]))
#    
#    
#log = logger.Train_Logger(output_root,"train_log")
#
#best = [0,np.inf]
#trigger = 0
#for epoch in trange(start_epoch, end_epoch):
#    # common.adjust_learning_rate(optimizer, epoch, args)
#    train_log = train(model, train_loader, optimizer, criterion, device)
#    print(train_log)
#    val_log = val(model, val_loader, optimizer, output_root, epoch, criterion, device)
#    # log.update(epoch,train_log,val_log)
#
#    print(val_log)
#    # trigger += 1
#    # if val_log['Val Loss'] < best[1]:
#    #     print('Saving best model')
#    #     torch.save(state, os.path.join(save_path, 'best_model.pth'))
#    #     best[0] = epoch
#    #     best[1] = val_log['Val Loss']
#    #     trigger = 0
#    # print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
#    # if trigger >= args.early_stop:
#    #     print("=> early stopping")
#    #     break
##restore_model_path = os.path.join(save_path, 'best_model.pth')
##model.load_state_dict(torch.load(restore_model_path)['net'])
##test(model, test_loader)

input_path1 = "./dataset64/ribfrac-val"
output_path1 = "./dataset-predicted/ribfrac-val"
input_root = './dataset64'

batch_size = 1
val_dataset = data.data1(root=input_root,split='val')
val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True, num_workers = 12)

print('Testing...')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = unet2.my3dunet(1,2).to(device)
model.load_state_dict(torch.load('../3dunet/output/ckpt_49_loss_0.13719.pth')['net'])

optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = lossfuc.DiceLoss(weight = np.array([0.1,0.9]))
#log = logger.Train_Logger(output_root, 'train_log')

model.eval()
val_loss = metrics.LossAverage()
val_dice = metrics.DiceAverage()

#with torch.no_grad():
#    imglist = os.listdir(input_path1 + '/images')
#    for imgname in imglist:
        
with torch.no_grad():
    for batch_idx, (inputs, imgname, affine) in enumerate(val_loader):
        print(imgname[0])
        inputs = inputs.float().to(device)
        #targets = targets.long()
        #targets = common.to_one_hot_3d(targets)
        #targets = targets.to(device)
        outputs = model(inputs.to(device))
        print(outputs.shape)
        #loss=criterion(outputs, targets)
        #val_loss.update(loss.item(),inputs.size(0))
        #val_dice.update(outputs, targets)
        labeldata = np.empty(shape=(64,64,64), dtype = np.int16)
        labeldata = outputs[0][1].cpu().numpy()
        print(labeldata.shape)
        #print(np.squeeze(affine.numpy()))
        save_label = nib.Nifti1Image(labeldata, np.squeeze(affine.numpy()))
        nib.save(save_label, output_path1 + '/labels/' + imgname[0].replace("-image.nii.gz", '-label.nii.gz'))
        #exit(0)
        #nib.save(





