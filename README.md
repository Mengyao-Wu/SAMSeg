# ST-DASegNet

This repo is the implementation of ["Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation"](https://arxiv.org/pdf/2301.05526.pdf). we refer to  [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMGeneration](https://github.com/open-mmlab/mmgeneration). Many thanks to SenseTime and their two excellent repos.

<table>
    <tr>
    <td><img src="PaperFigs\Fig1.png" width = "100%" alt="Cross-Domain RS Semantic Segmentation"/></td>
    <td><img src="PaperFigs\Fig4.png" width = "100%" alt="ST-DASegNet"/></td>
    </tr>
</table>

## Dataset Preparation

We select Postsdam, Vaihingen and LoveDA as benchmark datasets and create train, val, test list for researchers to follow. 

**In the following, we provide the detailed commands for dataset preparation.**

**Potsdam**
     
     Move the ‘3_Ortho_IRRG.zip’ and ‘5_Labels_all_noBoundary.zip’ to Potsdam_IRRG folder 
     Move the ‘2_Ortho_RGB.zip’ and ‘5_Labels_all_noBoundary.zip’ to Potsdam_RGB folder
     python tools/convert_datasets/potsdam.py yourpath/ST-DASegNet/data/Potsdam_IRRG/ --clip_size 512 --stride_size 512
     python tools/convert_datasets/potsdam.py yourpath/ST-DASegNet/data/Potsdam_RGB/ --clip_size 512 --stride_size 512
    
    SeaElm:
    python tools/convert_datasets/potsdam.py ./data/Potsdam_IRRG_DA/ -o=./data/Potsdam_IRRG_DA/ --clip_size=512 --stride_size=512
    python tools/convert_datasets/potsdam.py ./data/Potsdam_RGB_DA/ -o=./data/Potsdam_RGB_DA/ --clip_size=512 --stride_size=512



**Vaihingen**

     Move the 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' to Vaihingen_IRRG folder 
     python tools/convert_datasets/vaihingen.py yourpath/ST-DASegNet/data/Vaihingen_IRRG/ --clip_size 512 --stride_size 256

    SeaElm:
    python tools/convert_datasets/vaihingen.py /data2/yrz/data/Vaihingen/IRRG/ -o=/data2/yrz/data/Vaihingen/IRRG --clip_size 512 --stride_size 256

**LoveDA**
    
     Unzip Train.zip, Val.zip, Test.zip and create Train, Val and Test list for Urban and Rural

**Sentinel-2**
    
     python tools/convert_datasets/Sentinel-2.py ./data/yourpath  --out_dir ./data/Sentinel2

**GID (GF-2)**
    
     python tools/convert_datasets/GF-2.py ./data/yourpath/GID/Large-scale_Classification_5classes/image_NirRGB --out_dir ./data/GID_G2R/ --clip_size 1024 --stride_size 1024
     python tools/convert_datasets/GF-2.py ./data/yourpath/GID/Large-scale_Classification_5classes/label_5classes --out_dir ./data/GID_G2R/ --clip_size 1024 --stride_size 1024

**CITY-OSM: CHICAGO and PARIS**
    
    python tools/convert_datasets/CITY-OSM.py ./data/yourpath/CITY-OSM/paris/ --out_dir ./data/CITY_paris/ --clip_size 512 --stride_size 512
    python tools/convert_datasets/CITY-OSM.py ./data/yourpath/CITY-OSM/chicago/ --out_dir ./data/CITY_chicago/ --clip_size 512 --stride_size 512

## ST-DASegNet

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd ST-DASegNet
     
     pip install -e .
     
     chmod 777 ./tools/dist_train.sh
     
     chmod 777 ./tools/dist_test.sh
     ```

### Training

**mit_b5.pth** : [google drive](https://drive.google.com/drive/folders/1cmKZgU8Ktg-v-jiwldEc6IghxVSNcFqk?usp=sharing) For SegFormerb5 based ST-DASegNet training, we provide ImageNet-pretrained backbone here.

We select deeplabv3 and Segformerb5 as baselines. Actually, we use deeplabv3+, which is a more advanced version of deeplabv3. After evaluating, we find that deeplabv3+ has little modification compared to deeplabv3 and has little advantage than deeplabv3.

For LoveDA results, we evaluate on test datasets and submit to online server (https://github.com/Junjue-Wang/LoveDA) (https://codalab.lisn.upsaclay.fr/competitions/424). We also provide the evaluation results on validation dataset.

<table>
    <tr>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Urban.jpg" width = "100%" alt="LoveDA UDA Urban"/></td>
    <td><img src="PaperFigs\LoveDA_Leaderboard_Rural.jpg" width = "100%" alt="LoveDA UDA Rural"/></td>
    </tr>
</table>

1. Potsdam IRRG to Vaihingen IRRG:

**train code**
```shell
cd ST-DASegNet
conda 
./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2

// SeaElm:
# 第一个点
./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2
./tools/dist_train.sh ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 1
./tools/dist_train.sh ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2 295003
./tools/dist_train.sh ./experiments/deeplabv3plus/cowout/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 1 295013
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 3 295023
./tools/dist_train.sh ./experiments/deeplabv3plus/patchout/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2 295033
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_dis/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 3 295043
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 3 295063
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
--resume-from ''
# 第二个点
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_low/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2 295073
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 1 295073
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2 295083 # work 66.49

--resume-from /data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask/iter_22500.pth

# 第三个点
./tools/dist_train.sh ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py 2 295103


CUDA_VISIBLE_DEVICES=0 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
--resume-from '/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_2500.pth'

CUDA_VISIBLE_DEVICES=0 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
--resume-from '' \

CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
--resume-from '' \

CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
--resume-from '' \

"""
```
**test code**
```shell

# test 
#patchmix_coco_mask/iter_2500 ： 49.31
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask/iter_2500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
# patchmix_coco_mask/iter_22500 ：64.24
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask/iter_22500.pth \
--eval mIoU mFscore --gpu-id 1 --auto-show
# full 
python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_27500.pth \
--eval mIoU mFscore --gpu-id 2 --auto-show

# 第一个点的主实验
# cd ../ADA
# python test.py --device_id=3 --data=./data --config=./configs/pot2val.json -f /data/yrz/repos/ADA/checkpoints/pot6_2_vai6/MixUp/best_iou_57.72.pth
# output:
#0 impervious surface (imp suf, white) iou/accuracy: 69.56/87.69.
#1 building (blue) iou/accuracy: 71.28/77.78.
#2 tree (green) iou/accuracy: 63.07/75.05.
#3 car (yellow) iou/accuracy: 35.27/51.21.
#4 low vegetation (low veg, cyan) iou/accuracy: 49.42/66.06.
#Val result: IoU/Acc 57.7203/71.5553.

#python test.py --device_id=2 --data=./data --config=./configs/rural2urban.json -f /data/yrz/repos/ADA/checkpoints/rural_2_urban/MixUp/best_iou_47.78.pth
#output：
#0 backgroud iou/accuracy: 28.84/41.10.
#1 building iou/accuracy: 48.38/76.52.
#2 road iou/accuracy: 43.66/63.22.
#3 water iou/accuracy: 69.32/76.96.
#4 barren iou/accuracy: 45.65/68.89.
#5 forest iou/accuracy: 46.73/75.28.
#6 agricultural iou/accuracy: 51.89/62.59.
#Val result: IoU/Acc 47.7812/66.3643.

# 66.08
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_40000.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show


# ada test
# 75.08
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity/iter_50000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
# 75.16
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_region-relate-entropy&impurity_0.022_40/iter_30000.pth' \
--eval mIoU mFscore --gpu-id 1 --auto-show
# random 66.17
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_random/iter_30000.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
# entropy
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_test_0.022_40_pixel_entropy/iter_25000.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

```

**viz code**
```shell
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/ada_viz/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
--resume-from '/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Potsdam2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_region-relate-entropy&impurity_0.022_40/iter_30000.pth' 

```
2. Vaihingen IRRG to Potsdam IRRG:

**train code**
```shell
 cd ST-DASegNet
 
 ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2
 ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2
 
 // SeaElm:
 
./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2 294983
./tools/dist_train.sh ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2 294993
./tools/dist_train.sh ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 2 294003
./tools/dist_train.sh ./experiments/deeplabv3plus/cowout/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 1 294013
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 1 294023
./tools/dist_train.sh ./experiments/deeplabv3plus/patchout/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 3 294033
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_dis/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 1 294043
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 3 294053

./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 3 294063 # work, 57.89
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_coco_mask_dis/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 1 294073
./tools/dist_train.sh ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py 1 294083 

 ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
```
**test code**
```shell
# 44.21
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2Potsdam_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask/iter_2500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
# 74.44
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2Potsdam_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_random/iter_50000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

 ```

3. Potsdam RGB to Vaihingen IRRG:
**train**
```shell
cd ST-DASegNet

./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 2
./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_PotsdamRGB2Vaihingen.py 2

// SeaElm
./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 0 294983
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
```
**test**
```shell
# 54.75
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_2500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
# 20.24
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
# 72.1
python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 82.52 | 91.16 | 90.42  |    89.7   | 91.16  |
#|      building      | 90.58 | 95.37 | 95.06  |   94.75   | 95.37  |
#|   low_vegetation   | 68.41 | 74.16 | 81.24  |   89.82   | 74.16  |
#|        tree        | 81.33 | 94.09 |  89.7  |   85.71   | 94.09  |
#|        car         |  74.8 | 80.49 | 85.58  |   91.35   | 80.49  |
#|      clutter       | 34.93 | 98.26 | 51.78  |   35.15   | 98.26  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+------+-------+---------+------------+---------+
#|  aAcc | mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+------+-------+---------+------------+---------+
#| 89.03 | 72.1 | 88.92 |   82.3  |   81.08    |  88.92  |
#+-------+------+-------+---------+------------+---------+

# 54.75
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_12500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
# 69.23
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_20000.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
# 60.78
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

```
     
4. Vaihingen IRRG to Potsdam RGB:
**train**
 ```shell
 cd ST-DASegNet
 
 ./tools/dist_train.sh ./experiments/deeplabv3/config/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 2
 ./tools/dist_train.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2PotsdamRGB.py 2
 
 // SeaElm
./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 1 294993
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py

CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
 ```

**test**
```shell
#
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data2/wmy/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_30000.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
# 31.71
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

# 69.84
python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

# 47.89
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

# 49.33
python tools/test.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_cowmix/iter_12500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

# 67.68
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_region_relate_entropy&impurity/iter_40000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show
# 59.49
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_region_relate_entropy&impurity/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

```



5. LoveDA Rural to Urban

 ```shell
 cd ST-DASegNet
 
 ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py 2
 ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2U.py 2


CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/segformerb5/config_LoveDA/full/segformerb5_769x769_40k_R2U.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py \
--resume-from '' \

CUDA_VISIBLE_DEVICES=0 python tools/train.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
--resume-from '' \

# 38.16
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_7500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
#+--------------+-------+-------+--------+-----------+--------+
#|    Class     |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------+-------+-------+--------+-----------+--------+
#|  background  | 33.11 | 66.14 | 49.74  |   39.86   | 66.14  |
#|   building   | 28.29 | 30.14 | 44.11  |    82.2   | 30.14  |
#|     road     | 33.31 |  63.7 | 49.97  |   41.11   |  63.7  |
#|    water     | 64.51 | 68.95 | 78.43  |   90.93   | 68.95  |
#|    barren    | 33.61 | 83.18 | 50.31  |   36.06   | 83.18  |
#|    forest    | 34.96 | 39.44 |  51.8  |   75.47   | 39.44  |
#| agricultural | 39.32 | 42.65 | 56.44  |   83.43   | 42.65  |
#+--------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 53.78 | 38.16 | 56.31 |   54.4  |   64.15    |  56.31  |
#+-------+-------+-------+---------+------------+---------+


# 48.68
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 4 --auto-show
# 50.94
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show
#+--------------+-------+-------+--------+-----------+--------+
#|    Class     |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------+-------+-------+--------+-----------+--------+
#|  background  | 42.38 |  71.1 | 59.53  |    51.2   |  71.1  |
#|   building   | 48.98 | 56.45 | 65.75  |   78.73   | 56.45  |
#|     road     | 48.64 | 56.43 | 65.45  |   77.89   | 56.43  |
#|    water     | 65.05 | 79.28 | 78.83  |   78.38   | 79.28  |
#|    barren    | 48.76 | 71.55 | 65.56  |   60.49   | 71.55  |
#|    forest    | 53.51 | 73.83 | 69.72  |   66.04   | 73.83  |
#| agricultural | 49.23 | 57.19 | 65.98  |   77.96   | 57.19  |
#+--------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 65.85 | 50.94 | 66.55 |  67.26  |    70.1    |  66.55  |
#+-------+-------+-------+---------+------------+---------+
#45.21
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_7500.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------+-------+-------+--------+-----------+--------+
#|    Class     |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------+-------+-------+--------+-----------+--------+
#|  background  |  39.6 | 66.52 | 56.73  |   49.45   | 66.52  |
#|   building   | 37.06 | 41.24 | 54.08  |    78.5   | 41.24  |
#|     road     | 40.27 | 44.93 | 57.42  |    79.5   | 44.93  |
#|    water     | 69.44 | 78.52 | 81.96  |   85.71   | 78.52  |
#|    barren    |  46.0 | 55.61 | 63.02  |    72.7   | 55.61  |
#|    forest    | 40.47 |  79.6 | 57.63  |   45.16   |  79.6  |
#| agricultural | 43.66 | 54.05 | 60.79  |   69.44   | 54.05  |
#+--------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 60.74 | 45.21 | 60.07 |  61.66  |   68.64    |  60.07  |
#+-------+-------+-------+---------+------------+---------+


# 49.89
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

 ```
6. LoveDA Urban to Rural

 ```shell
 cd ST-DASegNet
 
 ./tools/dist_train.sh ./experiments/deeplabv3/config_LoveDA/ST-DASegNet_deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py 2
 ./tools/dist_train.sh ./experiments/segformerb5/config_LoveDA/ST-DASegNet_segformerb5_769x769_40k_U2R.py 2

CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_U2R.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_768x768_40k_U2R.py

CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/segformerb5/config_LoveDA/full/segformerb5_769x769_40k_U2R.py

# 31.74
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_768x768_40k_U2R.py \
'/data2/wmy/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full_test/iter_32500.pth' \
--eval mIoU mFscore --gpu-id 1 --auto-show
 ```


7. LoveDA R-G-B Rural to LandCoverNet Sentinel-2

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_S2LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2S.py 2
     ```

8. LoveDA R-G-B Rural to GID

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_GF2LoveDA/ST-DASegNet_segformerb5_769x769_40k_R2G.py 2
     ```

9. Paris to Chicago

     ```
     cd ST-DASegNet
     
     ./tools/dist_train.sh ./experiments/segformerb5/config_Paris2Chicago/ST-DASegNet_segformerb5_769x769_40k_P2C.py 2
     ```

### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. Testing commands

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mIoU   
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mFscore 
     ```

2. Testing cases: P2V_IRRG_64.33.pth and V2P_IRRG_59.65.pth : [google drive](https://drive.google.com/drive/folders/1qVTxY0nf4Rm4-ht0fKzIgGeLu4tAMCr-?usp=sharing)

    ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mFscore 
     ```
     
     ```
     cd ST-DASegNet
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Vaihingen2Potsdam.py 2 ./experiments/segformerb5/ST-DASegNet_results/V2P_IRRG_59.65.pth --eval mFscore 
     ```

The ArXiv version of this paper is release. [ST-DASegNet_arxiv](https://arxiv.org/pdf/2301.05526.pdf). This paper has been published on JAG, please refer to [Self-Training Guided Disentangled Adaptation for Cross-Domain Remote Sensing Image Semantic Segmentation](https://doi.org/10.1016/j.jag.2023.103646).

If you have any question, please discuss with me by sending email to lyushuchang@buaa.edu.cn.

# References
Many thanks to their excellent works
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [MMGeneration](https://github.com/open-mmlab/mmgeneration)
* [DAFormer](https://github.com/lhoyer/DAFormer)
