# Camouflage Object Segmentation with Attention-Guided Pix2Pix and Boundary Awareness

This repository contains the implementation of models presented in the paper **"Camouflage Object Segmentation with Attention-Guided Pix2Pix and Boundary Awareness."** The focus of this paper is on developing advanced models for the segmentation of camouflaged objects using a modified Pix2Pix architecture.

## Models

### 1. Discriminator

The `Discriminator` model is designed to distinguish between real and generated segmentation maps. It employs a PatchGAN architecture, evaluating local patches instead of the entire image. This enables more precise discrimination, particularly useful in handling high-frequency details.


### 2. Generator

The `Generator` model utilizes a Pix2Pix framework augmented with a VGG16 backbone for robust feature extraction. Convolutional Block Attention Modules (CBAMs) are integrated to enhance focus on critical details during the segmentation process.


## Future Work
- **Loss Boundary** 
- **Training Scripts** 
- **Evaluation Metrics** 
- **Extended Documentation** 
- **Full Pipeline** 




## Paper Reference

For more details on the methodology and evaluation metrics, please refer to the paper:

**Camouflage Object Segmentation with Attention-Guided Pix2Pix and Boundary Awareness**  
Erfan Akbarnezhad Sany, Fatemeh Naserizadeh, Parsa Sinichi, Seyyed Abed Hosseini  



## Status

🚧 **Project in Progress** 🚧

