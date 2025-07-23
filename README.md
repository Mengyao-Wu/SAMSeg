# SAMseg

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



## SAMSeg

### Install

1. requirements:
    
    python >= 3.7
        
    pytorch >= 1.4
        
    cuda >= 10.0
    
2. prerequisites: Please refer to  [MMSegmentation PREREQUISITES](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

     ```
     cd SAMSeg
     
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


2. Vaihingen IRRG to Potsdam IRRG:

**train code**
```shell
 cd SAMSeg
 
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

python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2Potsdam_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask/iter_2500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2Potsdam_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_random/iter_50000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

 ```

3. Potsdam RGB to Vaihingen IRRG:
**train**
```shell
cd SAMSeg

./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py 0 294983
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py

CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py
```
**test**
```shell

python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_2500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show



python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_12500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_20000.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show

python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/PotsdamRGB2Vaihingen_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

```
     
4. Vaihingen IRRG to Potsdam RGB:
**train**
 ```shell
 cd SAMSeg
 
./tools/dist_train.sh ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py 1 294993
CUDA_VISIBLE_DEVICES=2 python tools/train.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py

CUDA_VISIBLE_DEVICES=3 python tools/train.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py
 ```

**test**
```shell

python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data2/wmy/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_30000.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show


python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show


python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_15000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show


python tools/test.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results/deeplabv3plus_r50-d8_4x4_512x512_40k_cowmix/iter_12500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show


python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_region_relate_entropy&impurity/iter_40000.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/Vaihingen2PotsdamRGB_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_region_relate_entropy&impurity/iter_22500.pth' \
--eval mIoU mFscore --gpu-id 0 --auto-show

```



5. LoveDA Rural to Urban

 ```shell
 cd SAMSeg
 
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


python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_7500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show




python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_Base/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 4 --auto-show

python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_7500.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show



python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results_ada/deeplabv3plus_r50-d8_4x4_512x512_40k_ADA_0.022_40_normalize2_region_relate_entropy&impurity_k=1/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

python tools/test.py ./experiments/deeplabv3plus/full/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py \
'/data/yrz/repos/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_full/iter_37500.pth' \
--eval mIoU mFscore --gpu-id 2 --auto-show

 ```




### Testing
  
Trained with the above commands, you can get a trained model to test the performance of your model.   

1. Testing commands

    ```
     cd SAMSeg
     
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mIoU   
     ./tools/dist_test.sh yourpath/config.py yourpath/trainedmodel.pth --eval mFscore 
     ```

2. Testing cases: P2V_IRRG_64.33.pth and V2P_IRRG_59.65.pth : [google drive](https://drive.google.com/drive/folders/1qVTxY0nf4Rm4-ht0fKzIgGeLu4tAMCr-?usp=sharing)

    ```
     cd SAMSeg
     
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mIoU   
     ./tools/dist_test.sh ./experiments/segformerb5/config/ST-DASegNet_segformerb5_769x769_40k_Potsdam2Vaihingen.py 2 ./experiments/segformerb5/ST-DASegNet_results/P2V_IRRG_64.33.pth --eval mFscore 
     ```
     
     ```
     cd SAMSeg
     
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
