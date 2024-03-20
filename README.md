# VQCT
Codebook Transfer with Part-of-Speech for Vector-Quantized Image Modeling

# Codebook Transfer with Part-of-Speech for Vector-Quantized Image Modeling
This repository contains the code for the paper:
<br>
[**Codebook Transfer with Part-of-Speech for Vector-Quantized Image Modeling**](https://arxiv.org/pdf/2307.16424.pdf)
<br>
Baoquan Zhang, Huaibin Wang, Luo Chuyao, Xutao Li, Liang Guotao, Yunming Ye, Xiaochen Qi, Yao He
<br>
CVPR 2024

### Abstract

Vector-Quantized Image Modeling (VQIM) is a fundamental research problem in image synthesis, which aims to represent an image with a discrete token sequence. Existing studies effectively address this problem by learning a discrete codebook from scratch and in a code-independent manner to quantize continuous representations into discrete tokens. However, learning a codebook from scratch and in a code-independent manner is highly challenging, which may be a key reason causing codebook collapse, i.e., some code vectors can rarely be optimized without regard to the relationship between codes and good codebook priors such that die off finally. In this paper, inspired by pretrained language models, we find that these language models have actually pretrained a superior codebook via a large number of text corpus, but such information is rarely exploited in VQIM. To this end, we propose a novel codebook transfer framework with part-of-speech, called VQCT, which aims to transfer a well-trained codebook from pretrained language models to VQIM for robust codebook learning. Specifically, we first introduce a pretrained codebook from language models and part-of-speech knowledge as priors. Then, we construct a vision-related codebook with these priors for achieving codebook transfer. Finally, a novel codebook transfer network is designed to exploit abundant semantic relationships between codes contained in pretrained codebooks for robust VQIM codebook learning. Experimental results on four datasets show that our VQCT method achieves superior VQIM performance over previous state-of-the-art methods.

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{zhang2022metadiff,
	author    = {Zhang, Baoquan and Wang, Huaibin and Luo, Chuyao and Li, Xutao and Liang, Guotao and Ye, Yunming and Qi, Xiaochen and He, Yao},
	title     = {Codebook Transfer with Part-of-Speech for Vector-Quantized Image Modeling},
	booktitle = {CVPR},
	year      = {2024},
}
```
