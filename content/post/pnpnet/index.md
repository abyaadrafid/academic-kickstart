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
In this post, we will be looking at the paper [PnPNet: End-to-End Perception and Prediction with Tracking in the Loop](https://arxiv.org/abs/2005.14711), by *Liang et al.*,  which was published in CVPR 2020 [[1]](http://localhost:1313/post/pnpnet/#1). After discuss our task and discussing some related research in this field, we will be looking at the methodology of the paper. Then we will analyze the quantative results and have a look at the qualitative results. Finally, we will finish it off with some remarks and possible ideas for extension.

# Introduction

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque gravida dignissim suscipit. Integer quis faucibus felis. Pellentesque consectetur tellus odio, ornare ultrices massa ultrices nec. Nam sit amet tincidunt eros. Phasellus id posuere est. Sed accumsan accumsan risus vel posuere. Morbi at nibh ultricies, ultrices nisi non, congue nisl. Phasellus rhoncus ligula vel sem iaculis, sed bibendum lectus commodo. Fusce in aliquam tortor. Ut iaculis sapien enim, non mollis lectus cursus vitae. Aenean dictum ex at arcu volutpat elementum. Ut in elit molestie, convallis mauris id, mollis felis. Ut mattis leo non elit fermentum, vel interdum massa faucibus. Maecenas elit nibh, luctus ut massa sit amet, efficitur ornare magna.

Curabitur viverra, mi eu vestibulum cursus, tellus nibh lobortis ex, at congue felis ante in ante. Aliquam erat volutpat. Nulla facilisi. Sed id gravida est. Aliquam sem enim, rutrum ac rutrum id, pharetra quis sapien. Quisque malesuada lacus eget tortor pharetra ullamcorper. Integer non metus eros. Curabitur ornare efficitur sollicitudin. Nullam pharetra nisl sed leo scelerisque, ut cursus turpis fermentum.

Aliquam id convallis lorem, vehicula fringilla nisl. Sed euismod porttitor tempor. Vivamus at tortor quis ex faucibus ullamcorper sit amet a orci. Nullam nisi metus, rutrum at urna quis, lobortis venenatis orci. Nunc mollis lacus tortor, vitae pharetra dolor suscipit vitae. Praesent nec malesuada purus. Duis id magna eu lacus elementum faucibus.

Quisque scelerisque lectus dui. Fusce id ipsum sit amet ante consectetur consequat et ut justo. Mauris orci elit, sagittis in posuere vitae, euismod dignissim diam. Nulla ac velit pellentesque, gravida arcu tristique, feugiat elit. Quisque ultricies ut est non imperdiet. Vivamus blandit luctus ligula sagittis consequat. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Ut eu velit ac sapien convallis luctus.

Proin ornare non enim at dapibus. Aliquam erat volutpat. Aenean malesuada pharetra lacus, euismod suscipit mauris gravida quis. Morbi id pulvinar libero. Pellentesque imperdiet dignissim gravida. Morbi commodo augue non dolor lacinia aliquam. Nulla erat lorem, aliquam nec lacinia eu, efficitur in lacus.

# Related works

<div id="1">
[1] Ming Liang, Bin Yang, Wenyuan Zeng, Yun Chen, Rui Hu, Sergio Casas, Raquel Urtasun, "PnPNet: End-to-End Perception and Prediction with Tracking in the Loop", in CVPR, 2020.
</div>