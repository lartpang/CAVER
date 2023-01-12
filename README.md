# (TIP 2023) CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection

[![](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/lartpang/CAVER?style=flat-square)
![](https://img.shields.io/github/issues/lartpang/CAVER?style=flat-square)
![](https://img.shields.io/github/stars/lartpang/CAVER?style=flat-square)
[![](https://img.shields.io/badge/Arxiv-Paper-red?style=flat-square)](https://arxiv.org/abs/2112.02363)
[![](https://img.shields.io/badge/IEEE-Paper-red?style=flat-square)](https://ieeexplore.ieee.org/document/10015667)
[![](https://img.shields.io/badge/Page-Project-pink?style=flat-square)](https://lartpang.github.io/docs/caver.html)

```
@article{CAVER-TIP2023,
  author={Pang, Youwei and Zhao, Xiaoqi and Zhang, Lihe and Lu, Huchuan},
  journal={IEEE Transactions on Image Processing},
  title={CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection},
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3234702}
}
```

## Download

- Predictions: https://github.com/lartpang/CAVER/releases/tag/rgbd-rgbt-results
- Pre-trained parameters: https://github.com/lartpang/CAVER/releases/tag/rgbd-rgbt-models

## Code

Code will come soon!

## Method Detials

![](./assets/net.jpg)

*The overview of the proposed model. This is a dual-stream encoder-decoder architecture with a very simple and straightforward form. Note that the dashed line denotes an optional path for the decoder. In our model, the CMIU4 only contains two inputs $f^{4}_{rgb}$ and $f^{4}_{d/t}$ and $\hat{f}^{4}_{rgb-d/t}=\tilde{f}^{4}_{rgb-d/t}$. The feature $f^{i+1}_{rgb-d/t}$ exists in CMIU1-3, which is upsampled using bilinear interpolation in the 2D form.*

<img src="./assets/ptre.jpg" width="30%">

*Patch-wise token re-embedding (PTRE). Before matrix multiplication, the parameter-free PTRE is used to reshape features. Thus, pixel-wise tokens are aggregated and converted into patch-wise tokens.*

![](./assets/vma.jpg)

## Comparison with SOTA

PySODEvalToolkit: A Python-based Evaluation Toolbox for Salient Object Detection and Camouflaged Object Detection: <https://github.com/lartpang/PySODEvalToolkit>

<img src="./assets/flops-params-fps.jpg" width="50%">

![](./assets/rgbd-results-0.jpg)

![](./assets/rgbd-results-1.jpg)

<img src="./assets/rgbt-results.jpg" width="50%">

![](./assets/prfm.jpg)
