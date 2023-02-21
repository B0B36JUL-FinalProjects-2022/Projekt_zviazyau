# Facial Keypoints Detection

[Kaggle Facial Keypoints Detection](https://www.kaggle.com/competitions/facial-keypoints-detection)

# Installation
```julia
(@v1.8) pkg> add https://github.com/B0B36JUL-FinalProjects-2022/Projekt_zviazyau
```
# Neural network specification
## SimpleNet
**Layers:**
1. Input layer
2. Hidden Dense layer with 100 neurons and ReLU activation function
3. Output layer with Identity function

**Optimization method:** Gradient descent

**Loss function:**  Mean squared error

**Parameters initialization:** Xavier initialization


## MediumNet
**Layers:**
1. Input layer
2. Hidden Dense layer with 200 neurons and ReLU activation function
2. Hidden Dense layer with 100 neurons and softplus activation function
3. Output layer with Identity function

**Optimization method:** Gradient descent

**Loss function:**  Mean squared error

**Parameters initialization:** Xavier initialization


# Citation
+ [juliateachingctu - Julia for Optimization and Learning](https://juliateachingctu.github.io/Julia-for-Optimization-and-Learning/stable/)

+ [Samson Zhang - Building a neural network FROM SCRATCH](https://youtu.be/w8yWXqWQYmU)

+ [deeplizard - Deep Learning Fundamentals - Intro to Neural Networks](https://youtube.com/playlist?list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU)

+ [3Blue1Brown - Neural networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

+ [Daniel Nouri - Using convolutional neural nets to detect facial keypoints tutorial](https://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
)

+ [LAVANYA S - Face Key-point Recognition Using CNN](https://www.analyticsvidhya.com/blog/2021/07/face-key-point-recognition-using-cnn/)


+ [Aladdin Persson - Detecting Facial Keypoints with Deep Learning](https://youtu.be/84Lwv5PpWJA)

+ [dinghe - Facial Keypoint Detection](https://dinghe.github.io/CV_project.html)
