# Exploring the Effect of Data Augmentation on Image Classification Tasks

### Abstract

This paper investigates the impact of various data augmentation techniques on model performance across different datasets. We employ ResNet-18 and Vision Transformer as base model architectures, conducting experiments on CIFAR-10, CIFAR-100, and Fashion-MNIST datasets. Our study encompasses both tra-ditional and deep learning-based augmentation methods. Results indicate that among traditional data augmentation methods, Crop demonstrates superior per-formance, while Flip and Rotation exhibit comparable effectiveness. Both tech-niques enhance model generalization and mitigate overfitting effects. How-ever, pixel-level augmentation methods do not consistently yield positive out-comes. And the DiffuseMix method achieves improvements over the base-line while utilizing one-fifth of the augmented data.  



#### dataset

| Dataset       |      |      |
| ------------- | ---- | ---- |
| Cifar10       |      |      |
| Cifar100      |      |      |
| Fashion-Mnist |      |      |



#### Augmented data

![Cifar100](img/tradition.png)