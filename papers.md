# Papers

## 2015 U-Net

- size of dataset and network limited success of CNN

- CNN can be used for classification, for biomedical image, label per pixel is used (localization)

  - per pixel labeling is hard, so people use sliding window that label local region patch
    - local patches give more training data and able to localize, thus won EM segmentation challenge in 2012
    - but slow because network run separately for each patch
    - also redundancy because overlapping patches
    - also trade off between localization accuracy and context because large patch requires max pooling.
      - we can solve by accounting features from multiple layers

- This paper propose to:

  - use upsampling to replace pooling in usual contracting network
  - upsampling uses a large number of feature channels

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220501_1651400124.png" alt="image-20220501111523232" style="zoom: 80%;" />

## 2016 Image-to-Image Translation with Conditional Adversarial Networks

> https://arxiv.org/pdf/1611.07004.pdf

**Image-to-image translation** is the task of translating one possible representation of a scene into another, given sufficient training data, e.g.

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/04/upgit_20220430_1651332653.png" alt="image-20220430163051150" style="zoom: 67%;" />

This paper goal is to define a common framework for these problems.

- CNN is good but need to design effective loss function
- loss such as euclidean distance tend to produce blurry result $\rightarrow$ it minimize averaged plausible result
  - for sharp and realistic result requires expert knowledge
- we want high level goal and automatically learn a loss
- GAN is good for learning loss, and does not tolerate blurry image as it look obviously fake.
- Previous work focus on specific application
- Thus this paper:
  - explore GAN with conditional setting for image-to-image translation
  - present a simple framework sufficient to achieve good results on a wide variety of problems,
  - and to analyze the effects of several important architectural choices.

**Related Works**

**Loss** 

Structured losses for image modeling is often per-pixel classification or regression, pixels are independently conditioned. Losses includes:

- conditional random fields
- SSIM
- feature matching
- nonparametric losses 
- convolutional pseudo-prior
- matching covariance statistics

cGANs can penalize structure difference usin.

**cGAN**

- conditioned on discrete label, text, image, frame prediction, product photo generation, sparse annotations
- good for inpainting, future state prediction, image manipulation guided by user constraints, style transfer, and superresolution

This paper used more general and simpler setup. U-Net for generator and PatchGAN for discriminator

- l1 loss for less blurring.
- for nondeterministic output, they did not find gaussian noise effective as the generator learn to ignore noise, so they use drop out only on both train and test time, but also with little stochastic output

Generator:

- encoder-decoder network
- skip connections for layer i and n-i, like unet, for shuttling information directly, concatenates all channels 

Discriminator:

- l2 is good for low frequnecy but l1 is also enough for low frequency
- run on a NxN patch, generated image is run convolution-ally, average response and provide ultimate output of D
  - model image as a markov random field, assumed independence for pixels separated more than the patch diameter, this is commonly used for texture/style loss

Training:

- generator maximize $\log D(x,G(x,z))$ instead of minimize $\log(1-D(x,G(x,z))$.
- loss / 2 for discriminator to slow down its learning
- SGD, lr 0.0002, beta1=0.5, beta2=0.999
- for discriminator patch size N, 70 is good, 16 is enough for sharp output, but also tiling artifacts, larger N such as 286 does not improve visual quality, and got lower FCN score

Analysis

- more structural error in rigid geometry (photo to map) than map to photo, which is more chaotic.
- colorize produces grayscale or desaturated result

## 2017 CycleGAN

> https://arxiv.org/abs/1703.10593





## 2020 RIFE

> https://arxiv.org/abs/2011.06294'

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220502_1651523485.png" alt="image-20220502213123175" style="zoom: 80%;" />

video frame interpolation

- hard due to non linear motions and illumination changes

- recently flow-based algorithms achieved impressive result

  1. warping the input frames according to approximated optical flows
  2. fusing and refining the warped frames using CNN

  - But we do not have intermediate flow
    - some compute bi-directional flow using model and refine them to intermediate flow
      - but flaw in motion boundary, because now we only have one direction of the flow
    - some proposes [voxel flow](https://arxiv.org/abs/1702.02463) to jointly model intermediate flow and occlusion mask

This paper build 

- light weight pipeline that achieve sota performance
  - 3x3 conv only, no expensive cost volume 

- no additional component like image depth mode, flow refinement model and flow reversal layer
  - they are introduced to compensate for the defects of flow estimation
- direct supervision for the approximated intermediate flow
  - intermediate supervision is important
    - end to end construction loss does not work
    - used a privileged distillation scheme that employs a teacher model with access to the intermediate frames to guide the student to learn.
      - from knowledge distillation method that aims to transfer knowledge from a large model to a smaller one.
      - the teacher model gets more input than the student model, such as scene depth, images from other views, and even image annotation can guide student to learn

Optical flow estimation

- milestone are  flow net based on u net, and RAFT that iteratively update flow field

Video frame interpolation

- SuperSlomo uses the linear combination of the two bi-directional flows to estimate intermediate flows and refine with U-Net.
- DAIN made it weighted
- SoftSplat forward-warp frames and their feature map using softmax splatting
- QVI exploit four consecutive frames and flow reversal filter to get the intermediate flows. and EQVI extends with rectified quadratic flow prediction.

- flow free methods got improvements
  - phase information to learn the motion relationship
  - spatially adaptive convolution whose convolution kernel is generated using a CNN
  - deformable separable convolution
  - efficient flowfree method named CAIN, which employs the PixelShuffle operator and channel attention to capture the motion information implicitly

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220502_1651528347.png" alt="image-20220502225225841" style="zoom: 50%;" />

- uses RefineNet to refine the highfrequency area of warpped image

<img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/05/upgit_20220503_1651573043.png" alt="image-20220503111721688" style="zoom: 80%;" />

Training

- Vimeo90K dataset, 51312 triplets 448x256
- fixed timestep=0.5
- adamW weight=10-4
- 300 epochs
- batchsize 64
- lr=3x10-4 to 3x10-5, cosine annealing
- titan x pascal gpu 16hrs

Augmentation

- horizontal and vertical flipping
- temporal order reversal
- rotate 90 degrees



For arbitrary time steps, use Vimeo90K-Septuplet contains 7 consecutive frames, randomly select 3 frames and train it.

- use PSNR for evaluation

## 2014 Conditional Generative Adversarial Nets

