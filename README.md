# CCAMix

CCAMix: Context-and-Contour-Aware Superpixel-Based Data Augmentation with Local and Global Context Learning
==============================================================================================================

![image](https://github.com/DanielaPlusPlus/CCAMix/blob/main/Fig2a.png)
![image](https://github.com/DanielaPlusPlus/CCAMix/blob/main/Fig2b.png)

The source pytorch codes and some trained models are available here.

we propose CCAMix, an efficient context-and-contour-aware superpixel-based grid mixing approach with a cut-and-paste strategy for data augmentation in the input space for a strong classifier. By addressing the above shortcomings, we have achieved the following practical benefits.

(ⅰ) Context-aware. First, we introduce the local context mapping classification to strengthen the representation capability based on selected local discriminative superpixels. Second, we propose superpixel pooling and reweighting to learn contextual information of superpixel features in an image, and selectively focus on more discriminative superpixel-based local regions to generate holistic contextual information. Third, we perform contrastive learning to achieve alignment and consistency of the discriminative local superpixel context across images.

(ⅱ}) Contour-aware. On the one hand, we generate the augmented samples by cutting and pasting superpixel-based local regions, that can preserve the contour information in the input space. On the other hand, we perform superpixel pooling and reweighting based on the superpixel regions, which can preserve the contour information in the feature space.

(ⅲ) Efficient. We propose an augmentation approach with complete randomness for the largest diversification based on superpixel-based grid mixing. To avoid label mismatch problems, we mix labels for augmented samples with superpixel-wise semantic attention. In this way, we only need a single forward propagation and preserve the maximum diversification, which is efficient.


