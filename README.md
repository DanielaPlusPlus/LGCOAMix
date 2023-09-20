# LGCOAMix

LGCOAMix: Local and Global Context-and-Object-part-Aware Superpixel-based Data Augmentation for Deep Visual Recognition   
==============================================================================================================

![image](https://github.com/DanielaPlusPlus/LGCOAMix/blob/main/framework.png)

The source pytorch codes and some trained models are available here.

we propose LGCOAMix, an efficient local and global context-and-object-part-aware superpixel-based grid mixing data augmentation with cut-and-paste strategy and a training framework for Deep Visual Recognition. The motivation is to improve deep encoder learning through image data augmentation.


(1) We discuss the potential shortcomings of existing cutmix-based data augmentation methods for image classification.

(2) We propose an efficient object part-aware superpixel-based grid mixing method for data augmentation. Unlike existing cutmix-based data augmentation methods, we propose for the first time a superpixel-attention-based semantic label mixing strategy that efficiently requires only a single forward propagation, does not require pre-trained modules, and performs label mixing based on attention without destroying the augmentation diversification.

(3) We propose a novel framework for training a strong classifier that is context and object oriented as well as efficient. To the best of our knowledge, this is the first instance of learning local features from discriminative superpixel regions and cross-image local superpixel contrasts.

(4) We present extensive evaluations of LGCOAMix on several benchmarks and backbone encoders. These evaluations show that LGCOAMix outperforms existing cutmix-based methods for data augmentation.

Some trained models:
-------------------
[LGCOAMix + CUB200-2011 + ResNet18](https://github.com/DanielaPlusPlus/CCAMix/blob/main/CUB_224_R18_CCAMix_best.pt)

[OcCaMix + CUB200-2011 + ResNet18](https://github.com/DanielaPlusPlus/OcCaMix/blob/main/CUB_R18_OcCaMix.pt)

[OcCaMix + CUB200-2011 + ResNeXt50](https://github.com/DanielaPlusPlus/OcCaMix/blob/main/CUB_RX50_OcCaMix.pt)

[OcCaMix + TinyImageNet + ResNet18](https://github.com/DanielaPlusPlus/CCAMix/blob/main/TinyImageNet_R18_OcCaMix_best%20.pt)


The top.1 accuracy for classification:
--------------------------------------------------
<table align="left">
  <tr><th align="center">Method</th><th align="center">Dataset</th><th align="center">ResNet18</th><th align="center">ResNeXt50</th></tr>
  <tr><th align="center">OcCaMix</th><th align="center">TinyImageNet</th><th align="center">67.35%</th><th align="center">72.23%</th></tr>
  <tr><th align="center">OcCaMix</th><th align="center">CUB200-2011</th><th align="center">78.40%</th><th align="center">83.69%</th></tr>
  <tr><th align="center">CCAMix</th><th align="center">TinyImageNet</th><th align="center">68.27%</th><th align="center">73.08%</th></tr>
  <tr><th align="center">CCAMix</th><th align="center">CUB200-2011</th><th align="center">78.87%</th><th align="center">84.37%</th></tr>
</table>
