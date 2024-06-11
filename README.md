# ResUNet_Needle

This is the implementation for the final project of Bioengineering 224B (Advances in Imaging Informatics) at UCLA.

## Abstract
Needle segmentation in lung CT scans is a critical task in medical imaging, which is aimed at improving the efficiency of diagnostics and interventions in pulmonary diseases. Deep-learning-based medical image segmentation is a promising solution that provides efficient and reliable automatic segmentation results for clinical use. In this report, we propose to use a deep residual UNet (ResUNet) along with a straightness regularization in the loss function to achieve needle segmentation in lung CT scans. By evaluating our method on the validation set and the hidden test set, we show that our method can effectively segment the needle inside lung CT scans with a high Dice score and sensitivity. Furthermore, the additional straightness regularization term in the optimization objective significantly reduces false positive cases during inference.

## Qualitative Results
![image](https://github.com/Calvin-Pang/ResUNet_Needle/assets/72646258/f04b2b42-4d5b-4c86-929a-865a380ee3dd)

## Quantitative Results
![image](https://github.com/Calvin-Pang/ResUNet_Needle/assets/72646258/51fb203e-74b5-485d-b6b7-b5fb14a2feeb)

## Training and Inference
**Before start any trainig or inference, please remember to modify the path to the `meta_file`, `train_set`, `val_set` and `test_set` in the config!**

To reproduce the training process of three baseline compared in the project, please run the following commands and replace `your_exp_name` with any customized experiment name.

- Vanilla U-Net with StraightnessLoss: `python main.py --config ./configs/unet_straight_config.yaml --exp_name your_exp_name`
- ResUNet without StraightnessLoss: `python main.py --config ./configs/resunet_nonstraight_config.yaml --exp_name your_exp_name`
- ResUNet with StraightnessLoss: `python main.py --config ./configs/resunet_straight_config.yaml --exp_name your_exp_name`

If you want to do inference on the test images after training, please run the following commands. **Please match the `your_exp_name` here with the experiment name you used for training, so that the code can find the correct folder to get the trained model.**

- Vanilla U-Net with StraightnessLoss: `python inference.py --config ./configs/unet_straight_config.yaml --exp_name your_exp_name`
- ResUNet without StraightnessLoss: `python inference.py --config ./configs/resunet_nonstraight_config.yaml --exp_name your_exp_name`
- ResUNet with StraightnessLoss: `python inference.py --config ./configs/resunet_straight_config.yaml --exp_name your_exp_name`

## References
This codebase is inspired by the repositories [SwinYNet](https://github.com/Zch0414/Liver-Tumor-Segmentation-and-Recognition/tree/swin-ynet), [ResUNet](https://github.com/rishikksh20/ResUnet.git) and [LIIF](https://github.com/yinboc/liif).
