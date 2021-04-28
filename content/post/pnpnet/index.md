---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "PnPNet: End-to-End Perception and Prediction With Tracking in the Loop"
subtitle: "A review of a CVPR paper in the paradigm of autonomous driving"
summary: ""
authors: []
tags: [Deep Learning, Computer Vision]
categories: []
date: 2021-04-25T19:36:52+06:00
lastmod: 2021-04-25T19:36:52+06:00
featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: "Smart"
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
In this post, we will be looking at the paper [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/abs/2005.14711), by *Liang et al.*,  which was published in CVPR 2020 [[1]](#1). After defining our task and discussing some related research in this field, we will be looking at the methodology of the paper. Then we will analyse the quantitative results and have a look at the qualitative results. Finally, we will finish it off with some remarks and possible ideas for extension.

# Introduction

In the context of self-driving vehicles, predicting the motion of other vehicles is a critical task. Approximating the trajectory of neighbouring agents in the future is equally as important as detecting them in the current time frame. To do this task, so far three paradigms have been proposed.


The first divides this problem into three separate sub-tasks, which are handled by completely independent sub-systems. These three tasks, namely, object detection, object tracking, and motion forecasting are done sequentially. As they are developed separately, they need more computing power and cannot correct mistakes from upstream tasks. The second paradigm  tries to solve the detection and prediction task with a single neural network. This yields more efficient computation but these models suffer from limited use of temporal history and are vulnerable to occlusion.

![image](autonomy_stacks.png)
<figcaption>

Three paradigms of perception and prediction problems [[1]](#1)

</figcaption>

This paper introduces the new third paradigm. It argues that, for sequential modelling tasks such as motion forecasting, past data is very important. To that end, it proposes PnPNet which combines multi-object tracking with joint perception and prediction models. We will go into the details of the model after discussing some other works that try to tackle different aspects of our problem.

# Related works

### 3D object detection

The use of depth sensors such as LiDARs have been shown to have better performance [[2]](#2) than cameras for 3D detection. Some works also explore a fusion of LiDAR point clouds and camera inputs [[3]](#3).
![image](fusion_model.png)
<figcaption>

A qualitative result of fusion models [[3]](#3)

</figcaption>

### Multi-Object Tracking

Multi-Object tracking is a system to track multiple objects at the same time. It consists of a discrete data association problem and a continuous trajectory estimation problem [[4]](#4). There have been efforts to handle occlusion with hand crafted heuristics [[5]](#5) and single object tracking [[6]](#6) to handle occlusion. To handle the trajectory problem, some approaches also use sensor features but only use up to 3 seconds of temporal history [[7]](#7).

![image](multi-object_tracking.png)

<figcaption>

An example of multi-object tracking takes from [[8]](#8)

</figcaption>

### Motion Forecasting

Different methods try to approach the multi agent motion forecasting problem. *Alahi et al.* propose lstm based social pooling to model motion [[9]](#9). Social-GAN [[10]](#10) improves on it using adversarial training. The use of sensor features are also explored, but these methods usually have generalization issues when applied to noisy data [[11]](#11).

### Joint models for Perception and Prediction

The FAF paper [[12]](#12) of *Luo et al.* serves as a direct predecessor and an evaluation baseline for PnPNet. This model uses a single convolutional backbone to detect and predict future motion. NeuralMP [[13]](#13) shares motion planning features with perception and prediction to allow end to end training.

![image](neuralmp.png)
<figcaption>

Some qualitative results from NeuralMP [[13]](#13)

</figcaption>

# Proposed method : PnPNet

### Technical Contributions

The discussed related research overlook an important aspect of our problem. The do not take temporal characteristics of actors into account. To allow for that, this paper makes two major technical contributions :
  1. It introduces a new trajectory representation based on a sequence of detections through time.
  2. It proposes a multi-object tracker that solves both the association and the trajectory estimation problem.

## Model Overview

PnPNet consists of three separate modules :
1. 3D Object detection module
2. Discrete-Continuous Tracking module
3. Motion forecasting module

![image](model.png)
<figcaption>

A summary of PnPNet workflow [[1]](#1)

</figcaption>

### 3D Object detection module

The detection module takes multi-sweep LiDAR point cloud representation in bird-eye-view and an HD map as input. Optionally, geometric and semantic information of the HD map can also be used. 2D convolutional neural network based backbone is applied to the input, which generates the intermediate BEV features that will be used in downstream tasks. A convolutional detection header is then used on the intermediate features to create dense object detections at each time step.

![image](3dmodule.png)
<figcaption>

Workflow of the 3D detection module

</figcaption>

### Discrete-Continuous Tracking module

As discussed in the related works, there are two separate challenges in multi-object tracking. Previous works mostly focus on the discrete problem of data association but PnPNet also takes the continuous problem of trajectory estimation into account. The paper argues that it helps to prevent association errors from accumulating over time and reduces the variance in motion history. To that end, it proposes a two stage tracking framework. To use the framework, rich and concise trajectory level object representation need to be learned. 

![image](trackingrep.png)
<figcaption>

Trajectory level representation generation

</figcaption>

The representation learning problem is formulated as a sequence modelling problem. We use the intermediate FEV features from the backbone network and the location information of each object at time t to run a Bilinear Interpolation. This output is regarded as the representation of motion of each object from the start frame to the current frame. Along with this, the absolute velocities of each object is passed into an MLP/feed-forward network. The merged features from this FF network is then used as the sequential input for our LSTM. This LSTM hidden state is our trajectory level representation at each time step.

#### Data Association

First stage of the tracking framework is the discrete tracker. At time t, given N<sub>t</sub> detections and M<sub>t-1</sub> trajectories, it tries to determine the associations between them. This association problem is formulated as a bipartite graph matching problem. As a result, a one-to-one matching is guaranteed. The edge values for our bipartite matching problem are the affinity values, which represent how likely is a track M<sub>t-1</sub> to belong to detection N<sub>t</sub>. The affinity matrix :
$$
 C_{i,j} =
    \begin{cases}
      MLP_{pair}(f(D_{i}^t); h(P_{j}^{t−1})) & if\ 1 ≤ j ≤ M_{t−1}\\\\
      MLP_{unary}(f(D_{i}^t)) & if\ j = M_{t−1} + i \\\\
      −inf & otherwise \\\\
    \end{cases}       
$$
The affinity values are calculated with binary or unary MLPs. If we have more or equal number of tracks at the previous step than detections at this step, we use the binary MLP. Otherwise the unary MLP is used. This bipartite system is solved optimally using the Hungarian algorithm. We use single object tracking for older unmatched tracks. Combining results from bipartite matching and SOT yields a final set of tracks P<sub>t</sub>.

##### Single object tracking
The single object tracker used in the PnPNet paper follows nearly the same methodology as the Siamese tracker [[14]](#14). The Siamese tracker has two networks. One of these twin networks receive an exemplar image as input, other one receives a search image as input. The task for the twin network is to try and find the exemplar image within the search image.

![image](sot.png)
<figcaption>

Siamese tracker from [[14]](#14).

</figcaption>

Siamese trackers usually have a cross-correlation layer at the end. PnPNet chooses to replace this layer with an MLP with learnable parameters.

##### Hungarian Algorithm

The Hungarian algorithm is an optimization algorithm that produces the best one-to-one matching when applied to a bipartite graph. In our context, two sets of nodes are the detections and the tracks, denoted as **N** and **M** respectively. The edges between these sets of nodes are the affinity values, denoted by **a**. Here we see an example formulation :

```mermaid
graph TD;
id1(("N<sub>1</sub>"))-- "a<sub>1</sub>" ---id2(("M<sub>1</sub>"));
id1(("N<sub>1</sub>"))-- "a<sub>4</sub>" ---id3(("M<sub>2</sub>"));
id1(("N<sub>1</sub>"))-- "a<sub>7</sub>" ---id4(("M<sub>3</sub>"));

id6(("N<sub>2</sub>"))-- "a<sub>2</sub>" ---id2(("M<sub>1</sub>"));
id6(("N<sub>2</sub>"))-- "a<sub>5</sub>" ---id3(("M<sub>2</sub>"));
id6(("N<sub>2</sub>"))-- "a<sub>8</sub>" ---id4(("M<sub>3</sub>"));

id7(("N<sub>3</sub>"))-- "a<sub>3</sub>" ---id2(("M<sub>1</sub>"));
id7(("N<sub>3</sub>"))-- "a<sub>6</sub>" ---id3(("M<sub>2</sub>"));
id7(("N<sub>3</sub>"))-- "a<sub>9</sub>" ---id4(("M<sub>3</sub>"));
```

Hungarian algorithm will find the best matching that maximizes the overall affinity value throughout the graph.

#### Trajectory Estimation


### Motion Forecasting module


### End-to-End Training


# Results


## Datasets


## Metrics


## Quantitative results


## Ablation studies


## Qualitative Results


# Summary

# References

<p id="1">
[1] Ming Liang, Bin Yang, Wenyuan Zeng, Yun Chen, Rui Hu, Sergio Casas, Raquel Urtasun, "PnPNet: End-to-End Perception and Prediction with Tracking in the Loop", in CVPR, 2020.
</p>

<p id="2">
[2] Bin Yang, Wenjie Luo, and Raquel Urtasun. Pixor: Realtime 3d object detection from point clouds. In CVPR, 2018.
</p>

<p id="3">
[3] Ming Liang, Bin Yang, Shenlong Wang, and Raquel Urtasun. Deep continuous fusion for multi-sensor 3d object detection. In ECCV, 2018.
</p>

<p id="4">
[4] Anton Milan, Konrad Schindler, and Stefan Roth. Multitarget tracking by discrete-continuous energy minimization. TPAMI, 38(10):2054–2068, 2015
</p>

<p id="5">
[5] Hasith Karunasekera, Han Wang, and Handuo Zhang. Multiple object tracking with attention to appearance, structure, motion and size. IEEE Access 7:104423–104434, 2019.
</p>

<p id="6"> 
[6] Peng Chu and Haibin Ling. Famnet: Joint learning of feature, affinity and multi-dimensional assignment for online multiple object tracking. In ICCV, 2019.
</p>

<p id ="7">
[7] Wenwei Zhang, Hui Zhou, Shuyang Sun, Zhe Wang, Jianping Shi, and Chen Change Loy. Robust multi-modality multi-object tracking. In ICCV, 2019.
</p>

<p id="8">
[8] Michael D Breitenstein, Fabian Reichlin, Bastian Leibe, Esther Koller-Meier, and Luc Van Gool. Online multiperson tracking-by-detection from a single, uncalibrated camera. TPAMI, 33(9):1820–1833, 2010.
</p>

<p id="9">
[9] Alexandre Alahi, Kratarth Goel, Vignesh Ramanathan, Alexandre Robicquet, Li Fei-Fei, and Silvio Savarese. Social lstm: Human trajectory prediction in crowded spaces. In CVPR, 2016.
</p>

<p id="10">
[10] Agrim Gupta, Justin Johnson, Li Fei-Fei, Silvio Savarese, and Alexandre Alahi. Social gan: Socially acceptable trajectories with generative adversarial networks. In CVPR, 2018.
</p>

<p id="11"> 
[11] Nicholas Rhinehart, Rowan McAllister, Kris Kitani, and Sergey Levine. Precog: Prediction conditioned on goals invisual multi-agent settings. In ICCV, 2019. 
</p>

<p id="12">
[12] Wenjie Luo, Bin Yang, and Raquel Urtasun. Fast and furious: Real time end-to-end 3d detection, tracking and motion forecasting with a single convolutional net. In CVPR, 2018.
</p>

<p id="13">
[13] Wenyuan Zeng, Wenjie Luo, Simon Suo, Abbas Sadat, Bin Yang, Sergio Casas, and Raquel Urtasun. End-to-end interpretable neural motion planner. In CVPR, 2019.
</p>

<p id="14">
[14]  Luca Bertinetto, Jack Valmadre, Joao F Henriques, Andrea Vedaldi, and Philip HS Torr. Fully-convolutional siamese networks for object tracking. In ECCV, 2016.
</p>