# few-shot-self-prompt-SAM
This is the official repo for "Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation"
# To Do
Yaoo
1. Add codes for Kvasir dataset/ ISIC2018
2. Write env setup
3. Add dataset link
4. Decorate the git repo add a github page
5. Add abstract and Model picture and results
6. Modify abstract: No revolutionizing(reviewer 3)
7. Comparing to Zhou et al.???

## Experiment
1. Unet -- Yao
2. Original SAM, prompt sam --Qi
3. Use log reg on the image directly

## Requirements
The codes is tested on 
- Python 3.11.4
- PyTorch 2.0.1
- Nvidia GPU (RTX 3090) with CUDA version 11.7
1. First run ```conda env create -f environment.yml```
2. Following guidelines from the official repo of [Segment Anything](https://github.com/facebookresearch/segment-anything/tree/main),
```pip install git+https://github.com/facebookresearch/segment-anything.git```
```pip install opencv-python pycocotools matplotlib onnxruntime onnx```
3. Download the checkpoints of the ViT model for SAM and put it under ```./checkpoints```
- ```vit_b(default)```: [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- ```vit_l```: [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- ```vit_h```: [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
