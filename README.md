# NIMA
This project is a self implemented Artificial Neural Network based on Google's research paper
["NIMA: Neural Image Assessment"](https://arxiv.org/pdf/1709.05424.pdf).
You can find a quick introduction on their [Research Blog](https://research.googleblog.com/2017/12/introducing-nima-neural-image-assessment.html).

NIMA tries to predict the quality of images under two aspects, aesthetically and technically via transfer learning.

This implementation of NIMA model can be build based on the following Keras Imagenet base models:
- Xception
- VGG16
- VGG19
- ResNet50
- InceptionV3
- InceptionResNetV2
- MobileNet

## Datasets
This project uses two Datasets to train the NIMA model.
1. [AVA](https://github.com/mtobeiyf/ava_downloader) (Warning: Dataset is not complete) used for aesthetic rating
2. [TID2013](http://www.ponomarenko.info/tid2013.htm) used for technical rating
