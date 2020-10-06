# Image-Dehazing-using-GMAN-net  

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/LICENSE)   [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)   

## Introduction  
Generic Model-Agnostic Convolutional Neural Network(GMAN) is a convolutional neural network proposed for haze removal and clear image restoration. It is an end-to-end deep learning system that employs the encoder-decoder network for denoising image. I've used Kaggle notebook for the purpose of implementation and training. Dataset used for training and validation is SOTS outdoor [available here](https://www.kaggle.com/wwwwwee/dehaze).  

**Detailed explanation and documentation [here](tinyurl.com/gman-dehaze-net).**

Note: Incase notebook is not loading on GitHub, you can check notebook with validation output upto 10 epochs [here](https://nbviewer.jupyter.org/github/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/Notebook/gman-net-for-image-dehazing.ipynb).  

## Requirements  
- Python(3.6+)  
- Tensorflow(2+)  
- GPU: Nvidia Tesla P100(provided by Kaggle)  

## How to use on your images
1. Download the [saved model](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/tree/master/saved_model).  
2. Give model path and image path to [test.py](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/test.py) and run.  
(Note: Saved model folder, test.py and images should be in same folder.)  

## Evaluation  
I've used naturally hazed images downloaded randomly from google and some images are from dataset. You can see the dehazed test images against hazy images [here](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/tree/master/results/test%20results), some of them are below. Dehazed test images with good resolution are [available here](https://drive.google.com/drive/folders/1UxGa7cpHT9rHrmdKPYje15lJeMJnWyiZ?usp=sharing).  

![test_104](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/results/test%20results/test_104.png)  
![test_104](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/results/test%20results/test_111.png)  
![test_104](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/results/test%20results/test_105.png)  
![test_104](https://github.com/sanchitvj/Image-Dehazing-using-GMAN-net/blob/master/results/test%20results/test_100.png)  

### Citation
```
@article{liu2019single,
  title={Single Image Dehazing with a Generic Model-Agnostic Convolutional Neural Network},
  author={Liu, Zheng and Xiao, Botao and Alrabeiah, Muhammad and Wang, Keyan and Chen, Jun},
  journal={IEEE Signal Processing Letters},
  volume={26},
  number={6},
  pages={833--837},
  year={2019},
  publisher={IEEE}
}
```
https://github.com/Seanforfun/GMAN_Net_Haze_Removal
