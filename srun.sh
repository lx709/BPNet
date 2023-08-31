#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J bpnet
#SBATCH -o bpnet.%J.out
#SBATCH -e bpnet.%J.err
#SBATCH --mail-user=xiang.li.1@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=192:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:v100:8
#SBATCH --cpus-per-task=12

cd /ibex/ai/home/lix0i/3DCoMPaT/BPNet

export TOKENIZERS_PARALLELISM=false

source ~/.bashrc

conda init bash

conda activate BPNet

# module load openmpi

module load cuda/10.0.130  

export OMP_NUM_THREADS=12

# 24909401, 6xv100, batch=90
# sh ./tool/train_xiang.sh com10_v1 config/compat/bpnet_10.yaml 16

## debug 1: obj_cls uses unwanted softmax, 
# debug 2 : remap vox to point set for evaluation

# 25143551,  8xv100, batch=80, use_2d_classifier, killed
# sh ./tool/train_xiang.sh com10_v2 config/compat/bpnet_10_v2.yaml 16

# 25143569, 8xv100, batch=80, use_3d_classifier, killed
# sh ./tool/train_xiang.sh com10_v3 config/compat/bpnet_10.yaml 16

# 25196194, 4xv100, batch=48, use_2d_classifier, base_lr=0.002
# sh ./tool/train_xiang.sh com10_v4_v2 config/compat/bpnet_10_v4.yaml 16
# [2023-04-20 14:00:49,008 INFO train_xiang.py line 496 108711] Train result at epoch [49/50]: mIoU/mAcc/allAcc 0.8207/0.8497/0.9875.
# [2023-04-20 14:00:49,009 INFO train_xiang.py line 499 108711] Train result 2d at epoch [49/50]: mIoU/mAcc/allAcc 0.5645/0.6070/0.9951.
# [2023-04-20 14:00:49,009 INFO train_xiang.py line 503 108711] Train result cls at at epoch [49/50]: acc:0.9960/total:40368
# [2023-04-20 14:06:18,338 INFO train_xiang.py line 643 108711] Val result 3d: mIoU/mAcc/allAcc 0.3142/0.4153/0.7423.
# [2023-04-20 14:06:18,338 INFO train_xiang.py line 645 108711] Val result 2d : mIoU/mAcc/allAcc 0.2952/0.3834/0.9608.
# [2023-04-20 14:06:18,338 INFO train_xiang.py line 647 108711] Val result 2dmat: mIoU/mAcc/allAcc 0.6502/0.7554/0.9751.
# [2023-04-20 14:06:18,338 INFO train_xiang.py line 649 108711] Val result 3dmat: mIoU/mAcc/allAcc 0.0032/0.0418/0.0066.
# [2023-04-20 14:06:18,338 INFO train_xiang.py line 651 108711] Class ACC0.7955

# 25196393, 4xv100, batch=48, use_2d_classifier, base_lr=0.002, vox_size=0.01, 
# sh ./tool/train_xiang.sh com10_v5 config/compat/bpnet_10_v4.yaml 16

# [2023-04-21 11:03:02,949 INFO train_xiang.py line 499 84893] Train result at epoch [36/50]: mIoU/mAcc/allAcc 
# 0.8228/0.8491/0.9827.
# [2023-04-21 11:03:02,949 INFO train_xiang.py line 502 84893] Train result 2d at epoch [36/50]: mIoU/mAcc/allAcc 0.5721/0.6168/0.9952.
# [2023-04-21 11:03:02,949 INFO train_xiang.py line 506 84893] Train result cls at at epoch [36/50]: acc:0.9960/total:40368
# [2023-04-21 11:09:46,178 INFO train_xiang.py line 646 84893] Val result 3d: mIoU/mAcc/allAcc 0.3305/0.4260/0.7513.
# [2023-04-21 11:09:46,179 INFO train_xiang.py line 648 84893] Val result 2d : mIoU/mAcc/allAcc 0.2820/0.3795/0.9600.
# [2023-04-21 11:09:46,179 INFO train_xiang.py line 650 84893] Val result 2dmat: mIoU/mAcc/allAcc 0.6523/0.7540/0.9788.
# [2023-04-21 11:09:46,179 INFO train_xiang.py line 652 84893] Val result 3dmat: mIoU/mAcc/allAcc 0.0041/0.0566/0.0085.
# [2023-04-21 11:09:46,179 INFO train_xiang.py line 654 84893] Class ACC0.6933

# 25203647, 4xv100, batch=48, use_2d_classifier, base_lr=0.002, vox_size=0.01, aug=True, 
# sh ./tool/train_xiang.sh com10_v6 config/compat/bpnet_10_v5.yaml 16
# [2023-04-21 16:30:08,224 INFO train_xiang.py line 499 153801] Train result at epoch [37/50]: mIoU/mAcc/allAcc 0.7729/0.8135/0.9685.
# [2023-04-21 16:30:08,225 INFO train_xiang.py line 502 153801] Train result 2d at epoch [37/50]: mIoU/mAcc/allAcc 0.5187/0.5614/0.9934.
# [2023-04-21 16:30:08,225 INFO train_xiang.py line 506 153801] Train result cls at at epoch [37/50]: acc:0.9960/total:40368
# [2023-04-21 16:37:22,460 INFO train_xiang.py line 646 153801] Val result 3d: mIoU/mAcc/allAcc 0.2739/0.3773/0.7175.
# [2023-04-21 16:37:22,461 INFO train_xiang.py line 648 153801] Val result 2d : mIoU/mAcc/allAcc 0.2726/0.3655/0.9583.
# [2023-04-21 16:37:22,461 INFO train_xiang.py line 650 153801] Val result 2dmat: mIoU/mAcc/allAcc 0.6304/0.7319/0.9757.
# [2023-04-21 16:37:22,461 INFO train_xiang.py line 652 153801] Val result 3dmat: mIoU/mAcc/allAcc 0.0051/0.0583/0.0076.
# [2023-04-21 16:37:22,461 INFO train_xiang.py line 654 153801] Class ACC0.7066

# 25203657, 4xv100, batch=48, use_2d_classifier, base_lr=0.002, vox_size=0.01, aug=True, material seg based on part seg
# sh ./tool/train_xiang.sh com10_v7 config/compat/bpnet_10_v5.yaml 16
# [2023-04-21 14:55:32,201 INFO train_xiang.py line 499 33451] Train result at epoch [39/50]: mIoU/mAcc/allAcc 
# 0.7709/0.8077/0.9706.
# [2023-04-21 14:55:32,201 INFO train_xiang.py line 502 33451] Train result 2d at epoch [39/50]: mIoU/mAcc/allAcc 0.5077/0.5496/0.9931.
# [2023-04-21 14:55:32,201 INFO train_xiang.py line 506 33451] Train result cls at at epoch [39/50]: acc:0.9960/total:40368
# [2023-04-21 15:02:11,184 INFO train_xiang.py line 646 33451] Val result 3d: mIoU/mAcc/allAcc 0.2793/0.3848/0.7309.
# [2023-04-21 15:02:11,185 INFO train_xiang.py line 648 33451] Val result 2d : mIoU/mAcc/allAcc 0.2601/0.3485/0.9576.
# [2023-04-21 15:02:11,185 INFO train_xiang.py line 650 33451] Val result 2dmat: mIoU/mAcc/allAcc 0.6466/0.7448/0.9770.
# [2023-04-21 15:02:11,185 INFO train_xiang.py line 652 33451] Val result 3dmat: mIoU/mAcc/allAcc 0.0042/0.0555/0.0073.
# [2023-04-21 15:02:11,185 INFO train_xiang.py line 654 33451] Class ACC0.7104

# 25251404, 4xv100, batch=48, use_2d_classifier, base_lr=0.01, vox_size=0.01, aug=True, 
# [2023-04-24 20:57:35,778 INFO train_xiang.py line 499 27112] Train result at epoch [38/50]: mIoU/mAcc/allAcc 0.6445/0.7076/0.9278.
# [2023-04-24 20:57:35,778 INFO train_xiang.py line 502 27112] Train result 2d at epoch [38/50]: mIoU/mAcc/allAcc 0.3612/0.4095/0.8851.
# [2023-04-24 20:57:35,778 INFO train_xiang.py line 506 27112] Train result cls at at epoch [38/50]: acc:0.9907/total:40368
# [2023-04-24 21:04:22,209 INFO train_xiang.py line 646 27112] Val result 3d: mIoU/mAcc/allAcc 0.2119/0.3221/0.6581.
# [2023-04-24 21:04:22,210 INFO train_xiang.py line 648 27112] Val result 2d : mIoU/mAcc/allAcc 0.1944/0.2656/0.6645.
# [2023-04-24 21:04:22,210 INFO train_xiang.py line 650 27112] Val result 2dmat: mIoU/mAcc/allAcc 0.4845/0.6243/0.7701.
# [2023-04-24 21:04:22,210 INFO train_xiang.py line 652 27112] Val result 3dmat: mIoU/mAcc/allAcc 0.0039/0.0597/0.0063.
# [2023-04-24 21:04:22,210 INFO train_xiang.py line 654 27112] Class ACC0.6560

# 25251407, 4xv100, batch=48, use_2d_classifier, base_lr=0.01, vox_size=0.01, aug=True, arch_3d: MinkUNet34A
# [2023-04-24 20:56:25,617 INFO train_xiang.py line 499 162739] Train result at epoch [36/50]: mIoU/mAcc/allAcc 0.6073/0.6749/0.9184.
# [2023-04-24 20:56:25,617 INFO train_xiang.py line 502 162739] Train result 2d at epoch [36/50]: mIoU/mAcc/allAcc 0.3031/0.3532/0.8261.
# [2023-04-24 20:56:25,617 INFO train_xiang.py line 506 162739] Train result cls at at epoch [36/50]: acc:0.9844/total:40368
# [2023-04-24 21:03:13,160 INFO train_xiang.py line 646 162739] Val result 3d: mIoU/mAcc/allAcc 0.2134/0.3118/0.6522.
# [2023-04-24 21:03:13,160 INFO train_xiang.py line 648 162739] Val result 2d : mIoU/mAcc/allAcc 0.1483/0.2047/0.6070.
# [2023-04-24 21:03:13,160 INFO train_xiang.py line 650 162739] Val result 2dmat: mIoU/mAcc/allAcc 0.4389/0.6007/0.7209.
# [2023-04-24 21:03:13,160 INFO train_xiang.py line 652 162739] Val result 3dmat: mIoU/mAcc/allAcc 0.0037/0.0563/0.0056.
# [2023-04-24 21:03:13,160 INFO train_xiang.py line 654 162739] Class ACC0.5686


# 25282099, 
# sh ./tool/train_xiang_2donly.sh com10_2donly_v1 config/compat/bpnet_10_2donly.yaml 16

### lr=0.002, vox_size=0.01 (slightly better)
# 25287824, 4xv100, batch=48, use_2d_classifier, base_lr=0.002, vox_size=0.01, aug=False, 
# sh ./tool/train_xiang.sh com10_new_v1 config/compat/bpnet_10_v5.yaml 16

# [2023-04-28 16:00:59,680 INFO train_xiang.py line 499 154935] Train result at epoch [38/50]: mIoU/mAcc/allAcc 0.8885/0.9070/0.9902.
# [2023-04-28 16:00:59,681 INFO train_xiang.py line 502 154935] Train result 2d at epoch [38/50]: mIoU/mAcc/allAcc 0.7256/0.7638/0.9826.
# [2023-04-28 16:00:59,681 INFO train_xiang.py line 506 154935] Train result cls at at epoch [38/50]: acc:0.9960/total:40368
# [2023-04-28 16:07:46,591 INFO train_xiang.py line 646 154935] Val result 3d: mIoU/mAcc/allAcc 0.5414/0.6289/0.8630.
# [2023-04-28 16:07:46,592 INFO train_xiang.py line 648 154935] Val result 2d : mIoU/mAcc/allAcc 0.4651/0.5789/0.8714.
# [2023-04-28 16:07:46,592 INFO train_xiang.py line 650 154935] Val result 2dmat: mIoU/mAcc/allAcc 0.7686/0.8648/0.8947.
# [2023-04-28 16:07:46,592 INFO train_xiang.py line 652 154935] Val result 3dmat: mIoU/mAcc/allAcc 0.0050/0.0622/0.0099.
# [2023-04-28 16:07:46,592 INFO train_xiang.py line 654 154935] Class ACC0.7813

# 25287900, 4xv100, batch=48, use_2d_classifier, base_lr=0.002, vox_size=0.01, aug=True, com10_new_v2
# sh ./tool/train_xiang.sh com10_new_v2 config/compat/bpnet_10_v6.yaml 16
# [2023-04-28 15:01:31,932 INFO train_xiang.py line 499 141754] Train result at epoch [37/50]: mIoU/mAcc/allAcc 0.8697/0.8981/0.9813.
# [2023-04-28 15:01:31,932 INFO train_xiang.py line 502 141754] Train result 2d at epoch [37/50]: mIoU/mAcc/allAcc 0.7235/0.7626/0.9819.
# [2023-04-28 15:01:31,932 INFO train_xiang.py line 506 141754] Train result cls at at epoch [37/50]: acc:0.9960/total:40368
# [2023-04-28 15:07:44,846 INFO train_xiang.py line 646 141754] Val result 3d: mIoU/mAcc/allAcc 0.5490/0.6376/0.8755.
# [2023-04-28 15:07:44,846 INFO train_xiang.py line 648 141754] Val result 2d : mIoU/mAcc/allAcc 0.4834/0.5716/0.8822.
# [2023-04-28 15:07:44,846 INFO train_xiang.py line 650 141754] Val result 2dmat: mIoU/mAcc/allAcc 0.7818/0.8671/0.8842.
# [2023-04-28 15:07:44,846 INFO train_xiang.py line 652 141754] Val result 3dmat: mIoU/mAcc/allAcc 0.0049/0.0613/0.0094.
# [2023-04-28 15:07:44,846 INFO train_xiang.py line 654 141754] Class ACC0.7423

### TOTO
# sh ./tool/train.sh com10_coarse_new2 config/compat/bpnet_10_coarse.yaml 12

# sh ./tool/train.sh com10_coarse_new3 config/compat/bpnet_10_coarse.yaml 12
# running

# sh ./tool/test.sh com10_coarse_3dcls config/compat/bpnet_10_coarse_3dcls.yaml 12
# 26017696

# sh ./tool/train.sh com10_fine_v4 config/compat/bpnet_10_fine.yaml 12
# 26018855

# sh ./tool/train.sh com10_fine_3dcls_v2 config/compat/bpnet_10_fine_3dcls.yaml 12
# 26018860

### Test
# sh ./tool/test.sh com10_coarse config/compat/bpnet_10_coarse.yaml 16
# sh ./tool/test.sh com10_coarse_new2 config/compat/bpnet_10_coarse.yaml 16

# sh ./tool/test.sh com10_fine_v4 config/compat/bpnet_10_fine.yaml 16


#### 
sh ./tool/train.sh com50_coarse config/compat/bpnet_50_coarse.yaml 12
# 27143148

# sh ./tool/train.sh com20_coarse config/compat/bpnet_20_coarse.yaml 12
# 27142942

# sh ./tool/train.sh com50_fine config/compat/bpnet_50_fine.yaml 12
# 26748943

