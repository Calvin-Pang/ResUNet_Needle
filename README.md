# ResUNet_Needle

This is the implementation for the final project of Bioengineering 224B (Advances in Imaging Informatics) at UCLA.

### Abstract
Needle segmentation in lung CT scans is a critical task in medical imaging, which is aimed at improving the efficiency of diagnostics and interventions in pulmonary diseases. Deep-learning-based medical image segmentation is a promising solution that provides efficient and reliable automatic segmentation results for clinical use. In this report, we propose to use a deep residual UNet (ResUNet) along with a straightness regularization in the loss function to achieve needle segmentation in lung CT scans. By evaluating our method on the validation set and the hidden test set, we show that our method can effectively segment the needle inside lung CT scans with a high Dice score and sensitivity. Furthermore, the additional straightness regularization term in the optimization objective significantly reduces false positive cases during inference.

### Qualitative Results
![image](https://github.com/Calvin-Pang/ResUNet_Needle/assets/72646258/f04b2b42-4d5b-4c86-929a-865a380ee3dd)


### Quantitative Results
![image](https://github.com/Calvin-Pang/ResUNet_Needle/assets/72646258/51fb203e-74b5-485d-b6b7-b5fb14a2feeb)

### Training
To reproduce the three baseline compared in the project, please run the following commands.
