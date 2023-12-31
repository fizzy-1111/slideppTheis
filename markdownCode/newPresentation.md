---
marp: true
html: true
theme: gaia
paginate: true
class: 
style: |
    img[alt~="left"] {
      position: absolute;
      top: 200px;
      left: 50px;
    }
    img[alt~="right"] {
      position: absolute;
      top: 200px;
      right: 50px;
    }
    img[alt~="center"] {
      display: block;
      margin-left: auto;
      margin-right: auto;
      width: 100%;
      margin-top: 100px
    }
    img[alt~="centernotop"] {
      display: block;
      margin-left: auto;
      margin-right: auto;
      width: 100%;
    }
    [alt~="centersmalltop"] {
      display: block;
      margin-left: auto;
      margin-right: auto;
      width: 100%;
      margin-top: 50px
    }
    img[alt~="farLeft"]{
      position: absolute;
      top: 150px;
      left: 150px;
    }
    img[alt~="farRight"]{
      position: absolute;
      top: 150px;
      right: 0px;
      left: 750px;
    }
    img[alt~="full"] {
      position: absolute;
      top: 0px;
      right: 0px;
      width: 100%;
      height: 100%
    }
    img[alt~="fullpadtop"] {
      position: absolute;
      top: 100px;
      right: 0px;
      width: 100%;
      height: 100%
    }
footer: 26/08/2023 
math: mathjax
---
<div style="text-align:center">

# <span style="font-size:40px">THESIS</span>

### <span style="font-size:50px">DEVELOPING A VIRTUAL TRY-ON CLOTHES APPLICATION</span>

<span style="font-size:25px">NGUYỄN HOÀNG LINH 19125103 - NGUYỄN PHẠM TÙNG LÂM 19125056</span>

#### <span style="font-size:40px">THESIS ADVISORS</span>

<span style="font-size:30px">Mr. PHẠM MINH HOÀNG - Dr. VÕ HOÀI VIỆT</span>

</div>

---
# Outline
- Introduction
- Related works 
- Our approach 
- Experiments
- Conclusion

---
<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Introduction</span>
</div>

---
# Motivation

|                 |                     |
|-----------------|---------------------|
| **Traditional Shopping** | **Online Shopping**     |
| Require visit stores physically | Convenience         |
| Limited inventory | Wider range of products
| Able to try-on clothes | Unable to try on clothes |
| <div style="text-align:center"><img src="Traditional%20Shopping.png" alt="Traditional Shopping" width="400"/> | <div style="text-align:center"><img src="Online%20Shopping.png" alt="Online Shopping" width="350"/> |

---
# Problem Statements
- Virtual try-on task: Digitally try-on clothes or accessories
- Require only a human image and a cloth image
- Front view, with a clear background and minimal noise

![width:700 centernotop](app-introduce.png)

---

<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Related Works</span>
</div>

---

## Several approaches to Virtual Try-on

|                 |                     |  |
|-----------------|---------------------|--|
| <div style="width:320px">**Image-based (2D) Virtual Try-on** | <div style="width:300px">**3D Virtual Try-on**  | <div style="width:370px"> **Multi-pose guided virtual try-on** |
| <span style="font-size:30px">Use a 2D image of the person and the clothing item</span>| <span style="font-size:30px">Employs 3D models to simulate clothing on a person's body </span> | <span style="font-size:30px">Transfer clothes onto a person image under diverse poses </span>|
|<span style="font-size:30px">Suitable for mobile apps and e-commerce platforms </span> | <span style="font-size:30px">Commonly used in virtual reality (VR) applications and high-end fashion industry </span>|<span style="font-size:30px">Used in various applications for versatile try-on experiences </span>|

<!-- - Virtual try-on with diffusion models -->

---
### Image-based Virtual Try-on
<div style="text-align:center"><img src="vton-overview.png" style="width:50%;float:right;"></div>

- <span style="font-size:30px">VITON (2018): An Image-based Virtual Try-on Network [1]</span>

- <span style="font-size:30px">VITON-HD (2021): High-Resolution Virtual Try-On via Misalignment-Aware Normalization [2]</span>

- <span style="font-size:30px">HR-VTON (2022): High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions [3]</span>

---
### 3D Virtual Try-on 

- <span style="font-size:30px">DeepWrinkles (2018): Accurate and Realistic Clothing Modeling [4]</span>
- <span style="font-size:30px">TailorNet (2020): Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style [5]</span>
- <span style="font-size:30px">M3D-VTON (2021): A Monocular-to-3D Virtual Try-On Network [6]</span>

<div style="text-align:center"><img src="tailornet.png" style="width:48%;"></div>

---
### Multi-Pose Guided Virtual Try-on

<div style="text-align:center"><img src="MG-VTON.png" style="width:70%;float:right;"></div>

- <span style="font-size:30px">MG-VTON (2019): Towards Multi-pose Guided Virtual Try-on Network [7]</span>

- <span style="font-size:30px">SPG-VTON (2021): Semantic Prediction Guidance for Multi-pose Virtual Try-on [8]</span>


<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Implementation</span>
</div>

---
<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Our Approach</span>
</div>

---

# Objectives

- Focus on HR-VITON[1] model, an promising image-based virtual try-on approach
- Investigate and improve the performance of the model by exploring different loss functions
- A virtual try-on web application applying the model

---

<div style="display:flex;justify-content:left;align-items:center;height:5px">
  
### <span style="font-size:60px;">System Overview</span>
</div>

![w:1100 centersmalltop](framework.png)

---


# Preprocessing Module

![width:800 centernotop](pre-processing.png)

<!-- ---

![width:850 centernotop](pre-processing_agnostic.png)

<div style="color: black; font-size: 30px; margin-top: 30px; text-align: center; ">
  <b>Clothing-agnostic Processing Flow </b>
</div> -->

---

### Try-On Condition Module

![width:900 centernotop](tryon-condition.png)


---

## Generator architecture
![w:1050 centernotop](Generator.png)

<!-- <div style="color: white; font-size: 30px; margin-top: 150px; margin-left:700px ">
<b>Generator Architecture</b>

- Two encoders  
- Four feature fusion blocks
- Condition Aligning stage --> -->

<!-- </div> -->
---

![w:1300 centernotop](Encoder.png)

---

### Feature Fusion Blocks

<!-- - Has two routes: the flow pathway and the seg pathway.
- Takes two inputs, $F_{f_{i-1}}$ and $F_{s_{i-1}}$.
- The two pathways communicate with each other to determine $F_{f_i}$ and $F_{s_i}$ simultaneously. -->


![w:1000 centernotop](Feature_Fusion_Block.png)


---
### Condition Aligning
![h:500 w:700 centernotop](ConditionAligning.png)

---
## Discriminator architecture
![w:1100 centernotop](Discriminator.png)
<!-- ![h:500 farRight](SubDiscriminator.png) -->

---

### Training Try-On Condition module

<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

<span style="font-size:30px;">Let :</span>

- <span style="font-size:25px;">D is the discriminator network, G is the generator network </span>
- <span style="font-size:25px;">Ground truth seg map $S$, generated seg map $\hat{S}$ </span>
- <span style="font-size:25px;">Synthetic data sample $z$ </span>

The loss function will entail a type of loss characteristic of GANs.

###### $\underset{G}{min} \; \underset{D}{max} V(D,G)=E_{S\sim p_{data}(S)} log\left(D(S)\right) + E_{z\sim p_{z}(z)} \left(1-\left(D\left(\hat{S}\right)\right)\right) \tag{4.3}$
</div>
</div>

---
### Training Try-On Condition module
<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

- <div style="font-size:30px;height:10px">Cross-entropy loss</div>
### <div style="font-size:25px;">$\mathcal{L}_{CE} = \sum_{i=1}^H\sum_{j=1}^W\sum_{k=1}^CS_{ijk}log(\hat{S}_{ijk})$</div>

- <div style="font-size:30px;;height:10px">L1 loss</div>
### <div style="font-size:25px;"> $\mathcal{L}_{L1} =  \sum_{i=0}^3 w_i  .\left| \left|W(c_m,F_{f_i})-S_c \right| \right|_1 +||\hat{S_c}- S_c||_1 \tag{4.5}$</div>

- <div style="font-size:30px;;height:10px">VGG loss</div>
### <div style="font-size:25px;"> $\mathcal{L}_{VGG} = \sum_{i=0}^3 w_i  . \phi(W(c,F_{f_i}),I_c) + \phi(\hat{I_c},I_c) \tag{4.6}$</div>
</div>
<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

- <span style="font-size:25px;">Ground truth map $S$, generated seg map $\hat{S}$</span>
- <span style="font-size:25px;">Predicted cloth segmentation mask $\hat{S_c}$, ground truth $S_c$</span>
- <span style="font-size:25px;">Flow pathway $F_{f_i}$ </span>
- <span style="font-size:25px;">Predicted warped clothing image $\hat{I}_c$</span>
- <span style="font-size:25px;">Ground truth warped clothing image $I_c$ </span>
-  <span style="font-size:25px;">$\phi$ represent a loss function that measures the difference between two images</span>
</div>
</div>

---

### Training Try-On Condition module

<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

- <div style="font-size:30px;height:10px">Loss TV</div>
### <div style="font-size:25px;">$\mathcal{L}_{TV}= ||\nabla F_{f4}|| \tag{4.7}$</div>

- <div style="font-size:30px;;height:10px">Least square GAN loss</div>
### <div style="font-size:25px;">$\mathcal{L}_{cGAN}=\underset{G}{min}V_{LS}(G)= \frac{1}{2}E_{z\sim p_{z}(z)} \left[\left(D\left(G(z)\right)-1\right)^2\right] \tag{4.9}$</div>
</div>
<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

- <span style="font-size:25px;">$F_{f4}$ is the $4^{th}$ flow pathway   </span>
- <span style="font-size:25px;">$\nabla F_{f4}$ calculates the gradient of the flow pathway </span>
- <span style="font-size:25px;">Cloth mask $c_m$, original cloth image $c$ </span>
- <span style="font-size:25px;">D is the discriminator network, G is the generator network </span>
- <span style="font-size:25px;">Synthetic data sample $z$  </span>
</div>
</div>


---

# Training Try-On Condition module
<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

- <div style="font-size:30px;height:10px">Generator loss:</div>
### <div style="font-size:25px;">$\mathcal{L}_{TOCG} = \lambda_{CE} \mathcal{L}_{CE} + \mathcal{L}_{cGAN} + \lambda_{L1}\mathcal{L}_{L1} + \mathcal{L}_{VGG} + \lambda_{TV}\mathcal{L}_{TV}$</div>

- <div style="font-size:30px;;height:10px">Discriminator loss:</div>
### <div style="font-size:25px;">$\mathcal{L}_{D}^{LS} = \frac{1}{2}\mathbb{E}_{S\sim p_{data}(S)}[(D(S)-1)^2] + \frac{1}{2}\mathbb{E}_{z\sim p_z(z)}[D(G(z))^2]$</div>
</div>
<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

- <span style="font-size:25px;">Generator loss is the combination of above loss </span>
- <span style="font-size:25px;">D is the discriminator network, G is the generator network  </span>
- <span style="font-size:25px;">Synthetic data sample $z$ </span>
</div>
</div>


---


### Try-On Image Module
![width:750 centernotop](tryon-image.png)

---


### Try-on Image Generator architecture

![width:1050 centernotop](ImageGenerator.png)


---
### SPADE Residual Block

![width:1200 centernotop](ResBlock.png)

---

## Discriminator architecture
![w:1100 centernotop](DiscriminatorImage.png)

---
### Training Try-On Image


<span style="font-size:30px;">Let :</span>

- <span style="font-size:25px;">D is the discriminator network, G is the generator network </span>
- <span style="font-size:25px;">Ground truth Image $I$, generated try on image $\hat{I}$ </span>
- <span style="font-size:25px;">Synthetic data sample $z$ </span>
</div>

The loss function also involve a type of loss that is typical of GANs. 

###### $\underset{G}{min} \; \underset{D}{max} V(D,G)=E_{I\sim p_{data}(I)} log\left(D(I)\right) + E_{z\sim p_{z}(z)} \left(1-\left(D\left(\hat{I}\right)\right)\right) \tag{4.11}$

<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">
</div>

---
### Training Try-On Image
<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

<span style="font-size:25px;">$L1$ loss</span>

### <span style="font-size:25px;">$\mathcal{L}_{L1} =  ||\hat{I}- I||_1 \tag{4.12}$</span>
<span style="font-size:25px;">Feature Matching loss </span>

### <span style="font-size:25px;">$\mathcal{L}_{FM}=\frac{1}{k}\sum_{i=0}^{k-1}||Di(G(z)) - Di(I_i)||_1$</span>
<span style="font-size:25px;">VGG loss</span>

### <div style="font-size:25px;"> $\mathcal{L}_{VGG} = \phi(\hat{I},I) \tag{4.6}$</div>
</div>
<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

- <span style="font-size:25px;">Ground truth $I$, generated try on image $\hat{I}$</span>
- <span style="font-size:25px;">$I_i$ is the real image from the $i^{th}$ layer.</span>
- <span style="font-size:25px;">$D_i(I_i)$ is the feature map from the $i^{th}$ layer of the discriminator </span>
-  <span style="font-size:25px;">$\phi$ represent a loss function that measures the difference between two images</span>
</div>
</div>

---
### Training Try-On Image
Apply different loss to $\mathcal{L}_{TOIG}^{cGAN}$ in each experiment
- Hinge Loss
- Least square loss
- Cross Entropy

---

### Training Try-On Image
<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

- <div style="font-size:30px;height:10px">Generator loss:</div>
### <div style="font-size:25px;height:0px">$\mathcal{L}_{TOIG} = \mathcal{L}_{TOIG}^{cGAN} + \lambda_{TOIG}^{VGG}\mathcal{L}_{TOIG}^{VGG} + \lambda_{TOIG}^{FM}\mathcal{L}_{TOIG}^{FM} + \lambda_{TOIG}^{L1}\mathcal{L}_{TOIG}^{L1}$</div>

- <div style="font-size:25px;;height:0px">Discriminator loss Hinge:</div>
### <div style="font-size:25px;height:0px">$\mathcal{L}_{D}^{H} = -\mathbb{E}_{I\sim p_{data}}[\text{max}(0, -1 + D(I))] - \mathbb{E}_{z\sim p_z}[\text{max}(0, -1 - D(\hat{I}))]$</div>
- <div style="font-size:25px;;height:0px">Discriminator loss Least Square:</div>
### <div style="font-size:25px;height:0px">$\mathcal{L}_{D}^{LS} = \frac{1}{2}\mathbb{E}_{I\sim p_{data}(I)}[(D(I)-1)^2] + \frac{1}{2}\mathbb{E}_{z\sim p_z(z)}[D(\hat{I})^2]$</div>
- <div style="font-size:25px;;height:0px">Discriminator loss Cross Entropy:</div>
### <div style="font-size:25px;">$\mathcal{L}_{D}^{CE}= E_{I\sim p_{data}(I)} log\left(D(I)\right) + E_{z\sim p_{z}(z)} \left(1-\left(D\left(\hat{I}\right)\right)\right)$</div>

</div>
<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

- <span style="font-size:25px;">Generator loss is the combination of above loss </span>
- <span style="font-size:25px;">Original image $I$, generated try on image $\hat{I}$</span>
- <span style="font-size:25px;">D is the discriminator network, G is the generator network  </span>
- <span style="font-size:25px;">Synthetic data sample $z$ </span>
</div>
</div>

---

<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Experiments</span>
</div>

---

# Dataset

- High-resolution virtual try-on dataset from VITON-HD [2]
- 13,679 frontal-view woman and top clothing image pairs
- 1024 x 768 resolution
- 11,647 pairs for training, 2,032 for testing

<div style="display:flex;justify-content:center">

<img src="human_01.jpg" alt="Image 1" width="200"/>
<img src="cloth_01.jpg" alt="Image 2" width="200"/>
<img src="human_02.jpg" alt="Image 3" width="200"/>
<img src="cloth_02.jpg" alt="Image 4" width="200"/>

</div>

---
 <!-- ## Evaluation Metrics

 - Structural Similarity Index (SSIM)

 $SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$

 - Mean Squared Error (MSE)

 $MSE(x, y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - y_i)^2$

 - Learned Perceptual Image Patch Similarity (LPIPS)

 $LPIPS(x, y) = \frac{1}{N}\sum_{i=1}^{N}|f_i(x) - f_i(y)|_2$

---

 ## Evaluation Metrics
<div style="display:flex;flex-direction:row;">
<div style="display:flex;gap:0px;flex-direction:column;align-items:left">

<span style="font-size:30px;">

 - Structural Similarity Index (SSIM)
 
 $SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$

 - Mean Squared Error (MSE)

 $MSE(x, y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - y_i)^2$

 - Learned Perceptual Image Patch Similarity (LPIPS)

 $LPIPS(x, y) = \frac{1}{N}\sum_{i=1}^{N}|f_i(x) - f_i(y)|_2$

</span>
</div>

<div style="display:flex;flex-direction:column;justify-content:center;align-items:right">

<span style="font-size:30px;"> 

With $x$ and $y$ are the two images being compared

</span>

</div>
</div>

--- -->

 ## Evaluation Metrics

Structural Similarity Index (SSIM)
 
$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$

With $x$ and $y$ are the two images being compared
- $\mu_x$ and $\mu_y$ are the mean values of $x$ and $y$
- $\sigma_x$ and $\sigma_y$ are the standard deviations of $x$ and $y$
- $\sigma_{xy}$ is the covariance between $x$ and $y$
- $C_1$ and $C_2$ are constants to stabilize the division

---

 ## Evaluation Metrics

Mean Squared Error (MSE)

$MSE(x, y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - y_i)^2$

With $x$ and $y$ are the two images being compared
- $n$ is the total number of pixels in the images
- $x_i$ and $y_i$ are the pixel values at position $i$ in $x$ and $y$, respectively.

---

 ## Evaluation Metrics

Learned Perceptual Image Patch Similarity (LPIPS)

$LPIPS(x, y) = \frac{1}{N}\sum_{i=1}^{N}|f_i(x) - f_i(y)|_2$

With $x$ and $y$ are the two images being compared
- $N$ is the number of image patches
- $f_i(x)$ and $f_i(y)$ are the feature representations of the $i$-th patch in $x$ and $y$, respectively

--- 

# Experiments

- Goal: Investigate and improve generator model performance by exploring different loss functions
- Focus: Loss function of Try-On Image module includes **GAN loss**, **L1 loss**, and **Feature Matching (FM) loss**.
- GAN loss function:
  - Cross-Entropy (CE) GAN loss
  - Least Square (LS) GAN loss
  - Hinge GAN loss

---

# Experiments
Two experiments conducted:
- Experiment 1:
Investigate impact of L1 and FM losses on generator performance and find optimal set of lambda values for generator loss function

- Experiment 2:
Analyze specific impact of each GAN loss function in combination with L1 and FM on performance of generator model

---

## Experiment 1: L1 vs. FM Loss

<span style="font-size:30px">

Try different lambda values for the L1 and FM losses:
- No L1 and no FM losses
- Fix the FM lambda at 10 and vary L1 lambda between 10 and 40
- Fix the L1 lambda at 10 and vary FM lambda between 10 and 40

Models trained using original paper parameters with:
- $512\times384$ resolution
- 8 batch size
- Training steps: 30,000
</span>

---

## Experiment 1: L1 vs. FM Loss

![width:650 centernotop](Exp1-table.png)

---
## Experiment 1: L1 vs. FM Loss

![width:1150 centernotop](Exp1-chart.png)
- L1 and FM losses improve performance, and FM is more impactful.
- Lambda values optimal for L1 and FM losses are 10 and 30.

---

### Experiment 2: GAN Losses combine with L1 and FM

<span style="font-size:28px">

- Try different GAN loss functions: **CE GAN loss**, **LS GAN loss** and **Hinge GAN loss**
- As for each GAN loss function:
  - No L1 and no FM
  - Include L1 without FM
  - Include FM without L1
  - Include both L1 and FM
- Models trained using original paper parameters with:
  - $512\times384$ resolution
  - 8 batch size
  - Training steps: 30,000
  - Optimal lambda values found in Experiment 1: 10 for L1 and 30 for FM
</span>

---

### Experiment 2: GAN Losses combine with L1 and FM

![width:500 centernotop](Exp2-table.png)

---

### Experiment 2: GAN Losses combine with L1 and FM

![width:1200 centernotop](Exp2-chart.png)
- GAN loss combined with L1 and FM can significantly impact the performance of a generator
- Cross-Entropy (CE) GAN loss function is the most effective

--- 

## Application

![width:1100 centernotop](application.png)

---

## Application

![width:1070 centernotop](app-result.png)

<!-- ---

## Application Overview -->

<!-- - Architecture: Microservice 

- Programming language: Python

- Communication between services: gRPC

- User interface: Streamlit

- Deploy: Docker -->
<!-- ![width:900 centernotop](app-overview.png) -->

---

#### Application Pipeline

![width:1020 centernotop](app-pipeline.png)

---

<div style="display:flex;justify-content:center;align-items:center;height:100%">
  
### <span style="font-size:64px;text-align:center">Conclusion</span>
</div>

---

# Conclusion

- Achieved a virtual try-on application 
- Provided insights into effectiveness of loss functions when training HR-VITON model

- Future research: 
  - Improve the application performance
  - Optimizing the pre-processing steps
  - Exploring alternative models

---
# References

<span style="font-size:30px">
[1]: X. Han, Z. Wu, Z. Wu, R. Yu, and L. S. Davis, “Viton: An image-based virtual try-on network,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 7543–7552.

[2]: S. Choi, S. Park, M. Lee, and J. Choo, “Viton-hd: High-resolution virtual try-
on via misalignment-aware normalization,” in Proc. of the IEEE conference on
computer vision and pattern recognition (CVPR), 2021.

[3]: S. Lee, G. Gu, S. Park, S. Choi, and J. Choo, “High-resolution virtual try-on with misalignment and occlusion-handled conditions,” 2022.
</span>

--- 

# References

<span style="font-size:30px">
[4]: Z. Lahner, D. Cremers, and T. Tung, “Deepwrinkles: Accurate and realistic clothing modeling,” in Proceedings of the European conference on computer
vision (ECCV), 2018, pp. 667–684.

[5]: C. Patel, Z. Liao, and G. Pons-Moll, “Tailornet: Predicting clothing in 3d as a function of human pose, shape and garment style,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 7365–7375.

[6]: F. Zhao, Z. Xie, M. Kampffmeyer, H. Dong, S. Han, T. Zheng, T. Zhang, and X. Liang, “M3d-vton: A monocular-to-3d virtual try-on network,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 13 239–13 249.
</span>

---


# References

<span style="font-size:30px">
[7]: H. Dong, X. Liang, X. Shen, B. Wang, H. Lai, J. Zhu, Z. Hu, and J. Yin, “Towards multi-pose guided virtual try-on network,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 9026–9035.

[8]: B. Hu, P. Liu, Z. Zheng, and M. Ren, “Spg-vton: Semantic prediction guidance for multi-pose virtual try-on,” IEEE Transactions on Multimedia, vol. 24, pp. 1233–1246, 2022.
</span>