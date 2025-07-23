# 毕业论文验收代码

## 环境

软件环境：[seaelm](https://pan.baidu.com/s/1tDgKuJgkTQfQpxcgLd_yxQ?pwd=1016)

## 数据集

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

## 权重

可通过链接进行下载：[url](https://pan.baidu.com/s/16nqk3_X4TU4hO7_57qheoQ?pwd=0116)

## 训练部分

## 测试部分

### chapter3
```shell
#chapter3
# Potsdam2Vaihingen
# cd ../ADA
python test.py --device_id=3 --data=./data --config=./configs/pot2val.json -f ./checkpoints/chapter3/Potsdam2Vaihingen_57.72.pth
# output:
#0 impervious surface (imp suf, white) iou/accuracy: 69.56/87.69.
#1 building (blue) iou/accuracy: 71.28/77.78.
#2 tree (green) iou/accuracy: 63.07/75.05.
#3 car (yellow) iou/accuracy: 35.27/51.21.
#4 low vegetation (low veg, cyan) iou/accuracy: 49.42/66.06.
#Val result: IoU/Acc 57.7203/71.5553.

#LoveDA_R2U
python test.py --device_id=3 --data=./data --config=./configs/rural2urban.json -f ./checkpoint/chapter3/LoveDA_R2U_47.78.pth
#output：
#0 backgroud iou/accuracy: 28.84/41.10.
#1 building iou/accuracy: 48.38/76.52.
#2 road iou/accuracy: 43.66/63.22.
#3 water iou/accuracy: 69.32/76.96.
#4 barren iou/accuracy: 45.65/68.89.
#5 forest iou/accuracy: 46.73/75.28.
#6 agricultural iou/accuracy: 51.89/62.59.
#Val result: IoU/Acc 47.7812/66.3643.
```

### chapter4
```shell
# Potsdam2Vaihingen
# 66.08
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'./checkpoints/chapter4/Potsdam2Vaihingen_66.08.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 80.71 | 89.86 | 89.33  |    88.8   | 89.86  |
#|      building      | 89.78 | 94.68 | 94.62  |   94.55   | 94.68  |
#|   low_vegetation   | 60.36 | 74.99 | 75.28  |   75.56   | 74.99  |
#|        tree        | 70.47 | 80.52 | 82.68  |   84.95   | 80.52  |
#|        car         | 63.35 | 86.09 | 77.57  |   70.57   | 86.09  |
#|      clutter       | 31.78 | 96.82 | 48.23  |   32.11   | 96.82  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 85.18 | 66.08 | 87.16 |  77.95  |   74.42    |  87.16  |
#+-------+-------+-------+---------+------------+---------+

# Vaihingen2Potsdam
# 57.89
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'./checkpoints/chapter4/Vaihingen2Potsdam_57.89.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 71.11 | 80.12 | 83.12  |   86.35   | 80.12  |
#|      building      |  71.4 | 87.71 | 83.31  |   79.34   | 87.71  |
#|   low_vegetation   | 52.74 | 71.21 | 69.06  |   67.04   | 71.21  |
#|        tree        | 52.81 | 68.97 | 69.12  |   69.27   | 68.97  |
#|        car         | 80.81 | 94.52 | 89.39  |   84.79   | 94.52  |
#|      clutter       | 18.46 | 24.47 | 31.17  |   42.92   | 24.47  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 76.28 | 57.89 | 71.17 |  70.86  |   71.62    |  71.17  |
#+-------+-------+-------+---------+------------+---------+

# PotsdamRB2Vaihingen
# 54.75
python tools/test.py ./experiments/deeplabv3plus/patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'./checkpoints/chapter4/PotsdamRGB2Vaihingen_54.75.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 74.51 | 85.74 |  85.4  |   85.06   | 85.74  |
#|      building      | 80.19 | 93.27 | 89.01  |   85.12   | 93.27  |
#|   low_vegetation   | 41.47 | 69.78 | 58.62  |   50.55   | 69.78  |
#|        tree        | 39.49 | 43.26 | 56.62  |   81.93   | 43.26  |
#|        car         | 62.07 | 84.33 |  76.6  |   70.16   | 84.33  |
#|      clutter       |  30.8 |  97.0 | 47.09  |   31.09   |  97.0  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+------+---------+------------+---------+
#|  aAcc |  mIoU | mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+------+---------+------------+---------+
#| 72.74 | 54.75 | 78.9 |  68.89  |   67.32    |   78.9  |
#+-------+-------+------+---------+------------+---------+

# Vaihingen2PotsdamRGB
# 49.33
python tools/test.py ./experiments/deeplabv3plus/cowmix/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'./checkpoints/chapter4/Vaihingen2PotsdamRGB_49.33.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 59.64 | 75.32 | 74.72  |   74.12   | 75.32  |
#|      building      | 66.61 | 79.45 | 79.96  |   80.47   | 79.45  |
#|   low_vegetation   | 47.95 | 80.09 | 64.82  |   54.45   | 80.09  |
#|        tree        | 38.35 | 44.38 | 55.44  |   73.84   | 44.38  |
#|        car         | 79.19 | 87.19 | 88.39  |   89.62   | 87.19  |
#|      clutter       |  4.25 |  5.04 |  8.15  |   21.29   |  5.04  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 69.46 | 49.33 | 61.91 |  61.91  |   65.63    |  61.91  |
#+-------+-------+-------+---------+------------+---------+

# LovaDA R2U
# 48.68(似乎还有更高的。。)
python tools/test.py ./experiments/deeplabv3plus/base/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py \
'./checkpoints/chapter4/LoveDA_R2U_48.68.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------+-------+-------+--------+-----------+--------+
#|    Class     |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------+-------+-------+--------+-----------+--------+
#|  background  | 37.18 | 60.16 | 54.21  |   49.33   | 60.16  |
#|   building   | 50.44 | 58.57 | 67.06  |   78.42   | 58.57  |
#|     road     | 40.03 | 67.06 | 57.17  |   49.83   | 67.06  |
#|    water     | 60.14 | 69.33 | 75.11  |   81.94   | 69.33  |
#|    barren    | 46.66 | 71.21 | 63.63  |   57.51   | 71.21  |
#|    forest    | 53.18 | 70.05 | 69.44  |   68.84   | 70.05  |
#| agricultural | 53.12 | 62.77 | 69.38  |   77.56   | 62.77  |
#+--------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 63.93 | 48.68 | 65.59 |  65.14  |    66.2    |  65.59  |
#+-------+-------+-------+---------+------------+---------+

```

### chapter5
```shell
# Potsdam2Vaihingen
# 75.16
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Potsdam2Vaihingen.py \
'./checkpoints/chapter5/Potsdam2Vaihingen_75.16.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 83.32 |  89.0 |  90.9  |   92.89   |  89.0  |
#|      building      | 91.66 |  96.7 | 95.65  |   94.62   |  96.7  |
#|   low_vegetation   | 68.29 | 78.56 | 81.16  |   83.93   | 78.56  |
#|        tree        | 79.85 | 91.37 | 88.79  |   86.36   | 91.37  |
#|        car         |  77.1 | 85.19 | 87.07  |   89.03   | 85.19  |
#|      clutter       | 50.72 | 97.86 |  67.3  |   51.29   | 97.86  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 89.14 | 75.16 | 89.78 |  85.15  |   83.02    |  89.78  |
#+-------+-------+-------+---------+------------+---------+

# Vaihingen2Potsdam
# 74.44
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2Potsdam.py \
'./checkpoints/chapter5/Vaihingen2Potsdam_74.44.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 83.13 | 90.21 | 90.79  |   91.37   | 90.21  |
#|      building      | 89.51 | 93.88 | 94.47  |   95.06   | 93.88  |
#|   low_vegetation   | 73.22 | 87.97 | 84.54  |   81.37   | 87.97  |
#|        tree        |  75.7 |  85.5 | 86.17  |   86.85   |  85.5  |
#|        car         | 89.32 | 93.96 | 94.36  |   94.77   | 93.96  |
#|      clutter       | 35.75 | 48.38 | 52.67  |    57.8   | 48.38  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 88.22 | 74.44 | 83.31 |  83.83  |   84.54    |  83.31  |
#+-------+-------+-------+---------+------------+---------+

# PotsdamRB2Vaihingen
# 69.23
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_PotsdamRGB2Vaihingen.py \
'./checkpoints/chapter5/PotsdamRGB2Vaihingen_69.23.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 81.44 | 87.06 | 89.77  |   92.66   | 87.06  |
#|      building      | 89.52 | 94.57 | 94.47  |   94.37   | 94.57  |
#|   low_vegetation   | 61.28 | 69.79 | 75.99  |   83.41   | 69.79  |
#|        tree        |  76.8 | 94.15 | 86.88  |   80.65   | 94.15  |
#|        car         | 71.01 | 76.86 | 83.05  |   90.32   | 76.86  |
#|      clutter       |  35.3 | 96.53 | 52.18  |   35.75   | 96.53  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 86.79 | 69.23 | 86.49 |  80.39  |   79.53    |  86.49  |
#+-------+-------+-------+---------+------------+---------+

# Vaihingen2PotsdamRGB
# 67.68
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_512x512_40k_Vaihingen2PotsdamRGB.py \
'./checkpoints/chapter5/Vaihingen2PotsdamRGB_67.68.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------------+-------+-------+--------+-----------+--------+
#|       Class        |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------------+-------+-------+--------+-----------+--------+
#| impervious_surface | 76.58 | 89.74 | 86.74  |   83.93   | 89.74  |
#|      building      | 81.88 | 95.12 | 90.04  |   85.47   | 95.12  |
#|   low_vegetation   | 61.81 | 73.38 |  76.4  |   79.66   | 73.38  |
#|        tree        | 71.09 | 79.68 |  83.1  |   86.83   | 79.68  |
#|        car         | 90.18 | 94.07 | 94.83  |   95.61   | 94.07  |
#|      clutter       | 24.57 | 29.77 | 39.44  |   58.42   | 29.77  |
#+--------------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 83.65 | 67.68 | 76.96 |  78.42  |   81.65    |  76.96  |
#+-------+-------+-------+---------+------------+---------+

# LovaDA R2U
# 49.89
python tools/test.py ./experiments/deeplabv3plus/ada_base/deeplabv3plus_r50-d8_4x4_768x768_40k_R2U.py \
'./checkpoints/chapter5/LoveDA_R2U_49.89.pth' \
--eval mIoU mFscore --gpu-id 3 --auto-show
#+--------------+-------+-------+--------+-----------+--------+
#|    Class     |  IoU  |  Acc  | Fscore | Precision | Recall |
#+--------------+-------+-------+--------+-----------+--------+
#|  background  | 36.21 | 53.19 | 53.17  |   53.14   | 53.19  |
#|   building   | 56.61 | 74.97 | 72.29  |   69.79   | 74.97  |
#|     road     | 46.85 | 67.37 |  63.8  |    60.6   | 67.37  |
#|    water     | 61.56 | 70.85 |  76.2  |   82.43   | 70.85  |
#|    barren    | 43.62 | 72.33 | 60.74  |   52.36   | 72.33  |
#|    forest    | 50.32 | 73.42 | 66.95  |   61.53   | 73.42  |
#| agricultural | 54.07 | 64.25 | 70.19  |   77.33   | 64.25  |
#+--------------+-------+-------+--------+-----------+--------+
#Summary:
#
#+-------+-------+-------+---------+------------+---------+
#|  aAcc |  mIoU |  mAcc | mFscore | mPrecision | mRecall |
#+-------+-------+-------+---------+------------+---------+
#| 65.18 | 49.89 | 68.06 |  66.19  |   65.31    |  68.06  |
#+-------+-------+-------+---------+------------+---------+
```
