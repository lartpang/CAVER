# CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection, TIP

[Arxiv Link](https://arxiv.org/abs/2112.02363)

## Download

- Predictions: https://github.com/lartpang/CAVER/releases/tag/rgbd-rgbt-results
- Pre-trained parameters: https://github.com/lartpang/CAVER/releases/tag/rgbd-rgbt-models

## Code

Code will come soon!

## Method Detials

![](./assets/net.jpg)

*The overview of the proposed model. This is a dual-stream encoder-decoder architecture with a very simple and straightforward form. Note that the dashed line denotes an optional path for the decoder. In our model, the CMIU4 only contains two inputs $f^{4}_{rgb}$ and $f^{4}_{d/t}$ and $\hat{f}^{4}_{rgb-d/t}=\tilde{f}^{4}_{rgb-d/t}$. The feature $f^{i+1}_{rgb-d/t}$ exists in CMIU1-3, which is upsampled using bilinear interpolation in the 2D form.*

![](./assets/ptre.jpg)

*Patch-wise token re-embedding (PTRE). Before matrix multiplication, the parameter-free PTRE is used to reshape features. Thus, pixel-wise tokens are aggregated and converted into patch-wise tokens.*

![](./assets/vma.jpg)


## Comparison with SOTA

PySODEvalToolkit: A Python-based Evaluation Toolbox for Salient Object Detection and Camouflaged Object Detection: <https://github.com/lartpang/PySODEvalToolkit>

![](./assets/flops-params-fps.jpg)

![](./assets/rgbd-results-0.jpg)

![](./assets/rgbd-results-1.jpg)

![](./assets/rgbt-results.jpg)

![](./assets/prfm.jpg)
