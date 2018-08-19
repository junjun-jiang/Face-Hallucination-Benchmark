# Face Hallucination Benchmark
A list of face hallucination (super-resolution) resources collected by Junjun Jiang.

Some classical algorithms (including NE, LSR, SR, LcR, LINE, and EigTran) that I
implemented can be found [here](https://github.com/junjun-jiang/TLcR-RL).



#### Classical Patch-based Methods

-   Hallucinating face, FG2000, S. Baker and T. Kanade.
    [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=840616)]

-   [NE] Super-resolution through neighbor embedding, CVPR2004, Chang et al.
    [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [LSR] Hallucinating face by position-patch, PR2010, Ma et al.
    [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [SR] Position-patch based face hallucination using convex optimization,
    SPL2010, Jung et al. [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [LcR] Noise robust face hallucination via locality-constrained
    representation, TMM2104, Jiang et al.
    [[Web](https://github.com/junjun-jiang)]

-   [LINE] Multilayer Locality-Constrained Iterative Neighbor Embedding,
    TIP2014, Jiang et al. [[Web](https://github.com/junjun-jiang)]

-   Face Hallucination Using Linear Models of Coupled Sparse Support, TIP2017,
    Reuben A. Farrugia et al.
    [[PDF](https://ieeexplore.ieee.org/document/7953547/)][[Web](https://www.um.edu.mt/staff/reuben.farrugia)]

-   Hallucinating Face Image by Regularization Models in High-Resolution Feature
    Space, TIP2018, Jingang Shi et al.
    [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8310603)]

#### Classical Global Face Methods

-   [EigTran] Hallucinating face by eigentransformation, TSMC-C2005, Xiaogang
    Wang et al. [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   Super-resolution of face images using kernel PCA-based prior, TMM2007, A.
    Chakrabarti et al. [[PDF](http://ieeexplore.ieee.org/document/4202583/)]

-   A Bayesian Approach to Alignment-Based Image Hallucination, ECCV2012, Ce Liu
    et al.
    [[PDF](https://people.csail.mit.edu/celiu/pdfs/ECCV12-ImageHallucination.pdf)]

-   A convex approach for image hallucination, AAPRW2013, P. Innerhofer et al.
    [[Code](http://www.escience.cn/system/file?fileId=88901)]

-   Structured face hallucination, CVPR2013, Y. Yang et al.
    [[Web](https://eng.ucmerced.edu/people/cyang35/CVPR13/CVPR13.html)]

#### Classical Two-Step Methods

-   A two-step approach to hallucinating faces: global parametric model and
    local nonparametric model, CVPR2001, Ce Liu et al.
    [[Web](https://people.csail.mit.edu/celiu/FaceHallucination/fh.html)]

-   Hallucinating faces: LPH super-resolution and neighbor reconstruction for
    residue compensation, PR2007, Yuting Zhuang et al.
    [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320307001355)]

-   [CCA] Super-resolution of human face image using canonical correlation
    analysis, PR2010, Hua Huang et al.
    [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320310000853)]

#### Deep Learning Method

-   [CBN] Deep cascaded bi-network for face hallucination, ECCV2016, S. Zhu et
    al. [[Web](https://github.com/Liusifei/ECCV16-CBN)]

-   [R-DGN] Ultra-resolving face images by discriminative generative networks,
    in ECCV2016, Xin Yu et al. [[Web](https://github.com/XinYuANU)]

-   [LCGE] Learning to hallucinate face images via component generation and
    enhancement, IJCAI2017, Y. Song et al.
    [[Web](http://www.cs.cityu.edu.hk/~yibisong/)]

-   [TDAE] Hallucinating very low-resolution unaligned and noisy face images,
    CVPR2017, Xin Yu et al. [[Web](https://github.com/XinYuANU)]

-   [TDN] Hallucinating very low-resolution unaligned and noisy face images by
    transformative discriminative autoencoders, AAAI2017, Xin Yu et al.
    [[Web](https://github.com/XinYuANU)]

-   [FaceAttr] Super-resolving very low-resolution face images with
    supplementary attributes, CVPR2018, Xin Yu et al.
    [[Web](https://github.com/XinYuANU)]

-   [FSRNet] FSRNet: End-to-End learning face super-resolution with facial
    priors, CVPR, 2018 Yu Chen et al. [[Web](https://github.com/tyshiwo/FSRNet)]

-   Super-FAN: integrated facial landmark localization and super-resolution of
    real-world low resolution faces in arbitrary poses with GANs, CVPR2018,
    Adrian Bulat et al. [[Web](https://github.com/1adrianb)]

-   Attribute-Guided Face Generation Using Conditional CycleGAN, ECCV2018,
    Yongyi Lu et al.
    [[PDF](https://arxiv.org/pdf/1705.09966.pdf)][[Web](http://www.cse.ust.hk/~yluaw/)]

-   Face super-resolution guided by facial component heatmaps, ECCV2018, Xin Yu
    et al.
    [[PDF](https://ivul.kaust.edu.sa/Documents/Publications/2018/Face%20Super%20resolution%20Guided%20by%20Facial%20Component%20Heatmaps.pdf)]
    [[Web](https://github.com/XinYuANU)]

-   [GFSRNet] Learning Warped Guidance for Blind Face Restoration, ECCV2018,
    Xiaoming Li et al. [[Web](https://github.com/csxmli2016/GFRNet)]

-   To learn image super-resolution, use a GAN to learn how to do image
    degradation first, ECCV2018, Adrian Bulat et al.
    [[PDF](https://arxiv.org/abs/1807.11458)][[Web](https://www.adrianbulat.com/)]

#### Discriminative face hallucination

-   SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face
    Hallucination, arXiv2018, Hsu et al.
    [[PDF](https://arxiv.org/pdf/1807.08370)]

-   Face hallucination using cascaded super-resolution and identity priors,
    arXiv2018, K. Grm et al. [[PDF](https://arxiv.org/pdf/1805.10938)]

-   Super-Identity Convolutional Neural Network for Face Hallucination,
    ECCV2018, Kaipeng Zhang et al. [[Web]](http://kpzhang93.github.io/)

#### Image Quality Measurement

-   RMSE, PSNR, SSIM

-   Face recognition rate

-   Mean Opinion Score (MOS)

#### Databases

###### Classical databases

-   [FERET](http://www.nist.gov/itl/iad/ig/colorferet.cfm)

-   [CMU-PIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)

-   [CAS-PEAL-R1](http://www.jdl.ac.cn/peal/JDL-PEAL-Release.htm)

-   [FEI](https://fei.edu.br/~cet/facedatabase.html)

###### Largescale databases

-   [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

-   [Menpo](https://www.menpo.org/)

-   [Widerface](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

-   [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

-   [VGGFace2](https://arxiv.org/abs/1710.08092)

-   [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
