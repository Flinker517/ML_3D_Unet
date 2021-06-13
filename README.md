# 3D Unet
1. Older Version  

First, `cd ./3dunet` and follow the instructions in `./3dunet/README.md` to train a model.

Then `cd ./optimized` and follow the instructions in `./optimized/README.md` to use the trained model to test FROC.  

2. Final Version  

Code folder includes our final version model.  

Code/odel folder includes our models. Unet.py defines 3D U-Net, losses.py defines loss functions and metrics.py defines used metrics.  

To train the model, use the command python train.py ---train_image_dir train_img1 --train_label_dir train_label01 --val_image_dir val_img1 --val_label_dir val_label01 --save_model True. It will save the model named model100.pth.  

To generate the predictions, use the command python predict.py --image_dir val_img1 --pred_dir pred --model_path model100.pth. It will save the predictions in pred folder.  

Finally, use https://github.com/M3DV/RibFrac-Challenge to evaluate. The results including model and predictions are submitted in the Canvans. The performance is shown in our report.
