# LGCOAMix

LGCOAMix: Local and Global Context-and-Object-part-Aware Superpixel-based Data Augmentation for Deep Visual Recognition
==============================================================================================================

![image](https://github.com/DanielaPlusPlus/LGCOAMix/blob/main/framework.png)

The source pytorch codes and some trained models are available here.

we propose CCAMix, an efficient context-and-contour-aware superpixel-based grid mixing approach with a cut-and-paste strategy for data augmentation in the input space for a strong classifier. By addressing the above shortcomings, we have achieved the following practical benefits.

(ⅰ) Context-aware. First, we introduce the local context mapping classification to strengthen the representation capability based on selected local discriminative superpixels. Second, we propose superpixel pooling and reweighting to learn contextual information of superpixel features in an image, and selectively focus on more discriminative superpixel-based local regions to generate holistic contextual information. Third, we perform contrastive learning to achieve alignment and consistency of the discriminative local superpixel context across images.

(ⅱ}) Contour-aware. On the one hand, we generate the augmented samples by cutting and pasting superpixel-based local regions, that can preserve the contour information in the input space. On the other hand, we perform superpixel pooling and reweighting based on the superpixel regions, which can preserve the contour information in the feature space.

(ⅲ) Efficient. We propose an augmentation approach with complete randomness for the largest diversification based on superpixel-based grid mixing. To avoid label mismatch problems, we mix labels for augmented samples with superpixel-wise semantic attention. In this way, we only need a single forward propagation and preserve the maximum diversification, which is efficient.

Some trained models:
-------------------
[CCAMix + CUB200-2011 + ResNet18](https://github.com/DanielaPlusPlus/CCAMix/blob/main/CUB_224_R18_CCAMix_best.pt)

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
