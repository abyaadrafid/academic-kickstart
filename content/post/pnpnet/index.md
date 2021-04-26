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
In this post, we will be looking at the paper [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/abs/2005.14711), by *Liang et al.*,  which was published in CVPR 2020 [[1]](http://abyaadrafid.github.io/post/pnpnet/#1). After defining our task and discussing some related research in this field, we will be looking at the methodology of the paper. Then we will analyze the quantative results and have a look at the qualitative results. Finally, we will finish it off with some remarks and possible ideas for extension.

# Introduction

In the context of self-driving vehicles, predicting the motion of other vehicles is a critical task. Approximating the trajectory of neighboring agents in the future is equally as important as detecting them in the current time frame. To do this task, so far three paradigms have been proposed.


The first divides this problem into three separate sub-tasks, which are handled by completely independent sub-systems. These three tasks, namely, object detection, object tracking, and motion forecasting are done sequentially. As they are developed separately, they need more computing power and cannot correct mistakes from upstream tasks. The second paradigm  tries to solve the detection and prediction task with a single neural network. This yields more efficient computation but these models suffer from limited use of temporal history and are vulnerable to occlusion.

![image](autonomy_stacks.png)
<figcaption>

Three paradigms of perception and prediction problems [[1]](http://abyaadrafid.github.io/post/pnpnet/#1)

</figcaption>

This paper introduces the new third paradigm. It argues that, for sequential modeling tasks such as motion forecasting, past data is very important. To that end, it proposes PnPNet which combines multi-object tracking with joint perception and prediction models. We will go into the details of the model after discussing some other works that try to tackle different aspects of our problem.

# Related works

### 3D object detection

The use of depth sensors such as LiDARs have been shown to have better performance [[2]](http://abyaadrafid.github.io/post/pnpnet/#2) than cameras for 3D detection. Some works also explore a fusion of LiDAR point clouds and camera inputs [[3]](http://abyaadrafid.github.io/post/pnpnet/#3).
![image](fusion_model.png)
<figcaption>

A qualitative result of fusion models [[3]](http://abyaadrafid.github.io/post/pnpnet/#3)

</figcaption>

### Multi-Object Tracking

Multi-Object tracking is a system to track multiple objects at the same time. It consists of a discrete data association problem and a continuous trajectory estimation problem [[4]](http://abyaadrafid.github.io/post/pnpnet/#4). There have been efforts to handle occulsions with hand crafted heuristics [[5]](http://abyaadrafid.github.io/post/pnpnet/#5) and single object tracking [[6]](http://abyaadrafid.github.io/post/pnpnet/#6) to handle occulsion. To handle the trajectory problem, some approaches also use sensor features but only use upto 3 seconds of temporal history [[7]](http://abyaadrafid.github.io/post/pnpnet/#7).

![image](multi-object_tracking.png)

<figcaption>

An example of multi-object tracking takes from [[8]](http://abyaadrafid.github.io/post/pnpnet/#8)

</figcaption>

### Motion Forecasting

### Joint models for Perception and Prediction

# References

<div id="1">
[1] Ming Liang, Bin Yang, Wenyuan Zeng, Yun Chen, Rui Hu, Sergio Casas, Raquel Urtasun, "PnPNet: End-to-End Perception and Prediction with Tracking in the Loop", in CVPR, 2020.
</div>

<div id="2">
[2] Bin Yang, Wenjie Luo, and Raquel Urtasun. Pixor: Realtime 3d object detection from point clouds. In CVPR, 2018.
</div>

<div id="3">
[3] Ming Liang, Bin Yang, Shenlong Wang, and Raquel Urtasun. Deep continuous fusion for multi-sensor 3d object detection. In ECCV, 2018.
</div>

<div id="4">
[4] Anton Milan, Konrad Schindler, and Stefan Roth. Multitarget tracking by discrete-continuous energy minimization. TPAMI, 38(10):2054–2068, 2015
</div>

<div id="5">
Hasith Karunasekera, Han Wang, and Handuo Zhang. Multiple object tracking with attention to appearance, structure, motion and size. IEEE Access 7:104423–104434, 2019.
</div>

<div id="6"> 
Peng Chu and Haibin Ling. Famnet: Joint learning of feature, affinity and multi-dimensional assignment for online multiple object tracking. In ICCV, 2019.
</div>

<div id ="7">
Wenwei Zhang, Hui Zhou, Shuyang Sun, Zhe Wang, Jianping Shi, and Chen Change Loy. Robust multi-modality multi-object tracking. In ICCV, 2019.
</div>

<div id="8">
Michael D Breitenstein, Fabian Reichlin, Bastian Leibe, Esther Koller-Meier, and Luc Van Gool. Online multiperson tracking-by-detection from a single, uncalibrated camera. TPAMI, 33(9):1820–1833, 2010.
</div>