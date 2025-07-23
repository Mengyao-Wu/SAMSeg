import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.cuda.set_per_process_memory_fraction(0.6, 0)  # 只用60%的显存，后面的0是指GPU 0
torch.cuda.empty_cache()


from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os

# === Step 1: 路径配置 ===
config_file = '/data2/wmy/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/deeplabv3plus_r50-d8_4x4_512x512_40k_R2U.py'
checkpoint_file = '/data2/wmy/ST-DASegNet/checkpoints/deeplabv3plus/LoveDA_R2U_results/deeplabv3plus_r50-d8_4x4_512x512_40k_patchmix_coco_mask_tea/iter_32500.pth'
image_path = '/data/yrz/datasets/LoveDAtest/Val/Urban/images_png/3517.png'

# === Step 2: 初始化模型 ===
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# === Step 3: 推理单张图 ===
result = inference_segmentor(model, image_path)

# === Step 4: 显示或保存可视化结果 ===
out_file = 'vis_result.png'
model.show_result(image_path, result, out_file=out_file, opacity=0.5)
print(f'Result saved to {out_file}')
