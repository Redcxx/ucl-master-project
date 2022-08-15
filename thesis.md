### Tobias Suggestion

- start now



- most useful: pros and cons
- research side: newest papers, directions, SOTA
- abilities to make edits



- describe data
- describe tasks
- overleaf latex



- for employer
  - simple story
  - hand drawn cartoon
  - try to get things into same logic
    - make pre-trained, show same logic
    - temporal fine tuning, show it is better
    - cleanup -> find dirty sketches to clean up
    - produce rough sketches
      - three steps -> fit everything into same logic



- setup stuff in latex, nice layout title name
- write some chapters about what i am trying to do without saying what did I do
  - explain settings, 
  - ch 1 and 2

- another section for result
  - input output, how to quantify, table, numbers
  - outcomes, measurement of success
- architectures are for later

### NoGhost Suggestion

- 







# Outlines

## colorization

#### Challenges & Solutions

anime sketch colorization is a difficult problem because neither depth or semantic information is presented in the sketches, there are infinite number of ways to produce feasible result, lack information presents in grayscale image, and there often lack of authentic pair of training data,

- learned prior, pretraining makes it easier and simpler, but can result in artifacts, because shadow can be missing in the noghost dataset images
- unet only output: https://medium.com/mlearning-ai/anime-illustration-colorization-with-deep-learning-part-2-62b1068ef734
- algorithmic methods can produce quality results but requires detailed color stroke inputs, it is time consuming and manually intensitve, https://dl.acm.org/doi/pdf/10.1145/1015706.1015780
- overfitting in synthetic data is not well handled, previous method addressed this in the cost of global shading coherent and realistic texture
  - https://github.com/pfnet/PaintsChainer
  - e.g.: https://petalica-paint.pixiv.dev/index_en.html
  - <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/08/upgit_20220814_1660467248.png" alt="image-20220814095408263" style="zoom:33%;" />
- grayscale colorization can achieve real-time: https://arxiv.org/pdf/1705.02999.pdf
  - <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/08/upgit_20220814_1660466835.png" alt="image-20220814094715835" style="zoom:33%;" />
- bad sketch can result in artifacts: https://arxiv.org/pdf/1704.08834.pdf
  - "dirty" coloring
  - <img src="https://raw.githubusercontent.com/redcxx/note-images/master/2022/08/upgit_20220814_1660466495.png" alt="image-20220814094134997" style="zoom:33%;" />

### Method

WGAN GP + perceptual (content) loss (https://arxiv.org/abs/1603.08155).

### Miscellance

#### WGAN GP

> **Wasserstein GAN + Gradient Penalty**, or **WGAN-GP**, is a generative adversarial network that uses the Wasserstein loss formulation plus a gradient norm penalty to achieve Lipschitz continuity.
>
> The original [WGAN](https://paperswithcode.com/method/wgan) uses weight clipping to achieve 1-Lipschitz functions, but this can lead to undesirable behaviour by creating pathological value surfaces and capacity underuse, as well as gradient explosion/vanishing without careful tuning of the weight clipping parameter c.
>
> A Gradient Penalty is a soft version of the Lipschitz constraint, which follows from the fact that functions are 1-Lipschitz iff the gradients are of norm at most 1 everywhere. The squared difference from norm 1 is used as the gradient penalty.

> Wasserstein GAN, or WGAN, is a type of generative adversarial network that minimizes an approximation of the Earth-Mover's distance (EM) rather than the Jensen-Shannon divergence as in the original GAN formulation. It leads to more stable training than original GANs with less evidence of mode collapse, as well as meaningful curves that can be used for debugging and searching hyperparameters.

