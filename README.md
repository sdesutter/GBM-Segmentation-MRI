# MRI-based glioblastoma segmentation

## Overview

This repository contains the code used for the segmentation models discussed in the paper:

#### [Modality redundancy for MRI-based glioblastoma segmentation](doi.org/10.1007/s11548-024-03238-4)

The models included here have been trained on the [BraTS2021](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) dataset using various combinations of modalities to evaluate the impact of the different modalities on automated glioblastoma segmentation, and are based on nnU-Net and SwinUNETR architectures. Implementations were made using [MONAI](https://github.com/Project-MONAI) version 1.2.0.

<!-- ![Summary of the workflow.](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs11548-024-03238-4/MediaObjects/11548_2024_3238_Fig1_HTML.png?as=webp) -->

## Training script

The training script is found under '**training.py**'. Choices in architecture and modality configuration can be made by adapting variables `arch` and `modalities` accordingly.

## Trained models

Trained models for all modality configurations and both networks can be downloaded [here](https://drive.google.com/drive/folders/15tiAk2PcOAgedWPIRqg0HGwkklxRA1mO?usp=sharing).  

## Citations

If you find this repository useful in your research, please consider citing our paper:

*De Sutter, S., Wuts, J., Geens, W. et al. Modality redundancy for MRI-based glioblastoma segmentation. Int J CARS (2024). https://doi.org/10.1007/s11548-024-03238-4* 
