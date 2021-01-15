# Face-Hallucination-Benchmark
A list of face hallucination/face super-resolution resources collected by [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun).

Some classical algorithms (including NE, LSR, SR, LcR, LINE, TLcR-RL, and EigTran) that I
implemented can be found [here](https://github.com/junjun-jiang/TLcR-RL).

# Recently, we have released a survey of deep learning-based face super-resolution at [arXiv](https://arxiv.org/abs/2101.03749).


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
    
-   [TLcR-RL] Context-Patch based Face Hallucination via Thresholding Locality-Constrained Representation and Reproducing Learning, TCYB2018, Junjun Jiang et al. [[PDF]](https://arxiv.org/abs/1809.00665)[[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   Face Hallucination via Coarse-to-Fine Recursive Kernel Regression Structure, Jingang Shi et al. TMM 2109.

-   Robust Face Image Super-Resolution via Joint Learning of Subdivided Contextual Model, Liang Chen et al. TIP2019. [[PDF](https://ieeexplore.ieee.org/abstract/document/8733990)]

-   SSR2: Sparse signal recovery for single-image super-resolution on faces with extreme low resolutions, RamziAbiantun et al. PR2019. [[https://www.sciencedirect.com/science/article/abs/pii/S0031320319300597]]



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

-   [R-DGN] Ultra-resolving face images by discriminative generative networks,
    in ECCV2016, Xin Yu et al. [[Web](https://github.com/XinYuANU)]


-   [TDAE] Hallucinating very low-resolution unaligned and noisy face images,
    CVPR2017, Xin Yu et al. [[Web](https://github.com/XinYuANU)]    

-   [TDN] Hallucinating very low-resolution unaligned and noisy face images by
    transformative discriminative autoencoders, AAAI2017, Xin Yu et al.
    [[Web](https://github.com/XinYuANU)]
    
-   Face Hallucination by Attentive Sequence Optimization with Reinforcement Learning, Yukai Shi et al. PAMI 2019.

-   Joint Face Hallucination and Deblurring via Structure Generation and Detail Enhancement, Yibing Song et al. IJCV 2019. [[Web](https://ybsong00.github.io/ijcv19_fhd/)]

-   Wavelet Domain Generative Adversarial Network for Multi-scale Face Hallucination, Huaibo Huang et al., IJCV 2109. [[Code](https://github.com/hhb072/WaveletSRNet)]

-   Face hallucination from low quality images using definition-scalable inference, Xiao Hu et al. PR 2019.

-   Can We See More? Joint Frontalization and Hallucination of Unaligned Tiny Faces, Xin Yu et al. PAMI 2019.

    
###### Structure prior based method (componet, landmarks, attention, salience, heatmaps, etc)

-   [CBN] Deep cascaded bi-network for face hallucination, ECCV2016, S. Zhu et
    al. [[PDF](https://arxiv.org/abs/1607.05046)][[Web](https://github.com/Liusifei/ECCV16-CBN)]
    
-   [LCGE] Learning to hallucinate face images via component generation and
    enhancement, IJCAI2017, Y. Song et al.
    [[PDF](https://arxiv.org/abs/1708.00223)][[Web](http://www.cs.cityu.edu.hk/~yibisong/)]
    
-   Attention-Aware Face Hallucination via Deep Reinforcement Learning,
    CVPR2017, Qingxing Cao et al. [[PDF](https://arxiv.org/abs/1708.03132)][[Web](https://github.com/ykshi/facehallucination)]
    
-   Deep CNN Denoiser and Multi-layer Neighbor Component Embedding for Face Hallucination, IJCAI2018, Junjun Jiang et al.
    [[PDF](https://arxiv.org/abs/1806.10726)][[Web](https://github.com/junjun-jiang/IJCAI-18)] 

-   [FSRNet] FSRNet: End-to-End learning face super-resolution with facial
    priors, CVPR, 2018 Yu Chen et al. [[PDF](https://arxiv.org/abs/1711.10703)][[Web](https://github.com/tyshiwo/FSRNet)]

-   Super-FAN: integrated facial landmark localization and super-resolution of
    real-world low resolution faces in arbitrary poses with GANs, CVPR2018,
    Adrian Bulat et al. [[PDF](https://arxiv.org/abs/1712.02765)][[Web](https://github.com/1adrianb)]

-   Face super-resolution guided by facial component heatmaps, ECCV2018, Xin Yu
    et al.
    [[PDF](https://ivul.kaust.edu.sa/Documents/Publications/2018/Face%20Super%20resolution%20Guided%20by%20Facial%20Component%20Heatmaps.pdf)]
    [[Web](https://github.com/XinYuANU)]
    
-   A coarse-to-fine face hallucination method by exploiting facial prior knowledge, ICIP2018,
    Mengyan Li et al. [[PDF](https://github.com/lemoner20/ICIP2018/blob/master/Li.pdf)][[Web](https://github.com/lemoner20/ICIP2018)]
    
-   Residual Attribute Attention Network for Face Image Super-Resolution, Jingwei Xin et al. AAAI2019. [[https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4937]]
    
-   Progressive Face Super-Resolution via Attention to Facial Landmark, ECCV2019, Deokyun Kim et al. [[PDF](https://arxiv.org/pdf/1908.08239.pdf)][[Code](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution)]
    
###### Attribute-Guided method

-   [FaceAttr] Super-resolving very low-resolution face images with
    supplementary attributes, CVPR2018, Xin Yu et al.
    [[PDF](https://basurafernando.github.io/papers/XinYuCVPR18.pdf)][[Web](https://github.com/XinYuANU)]
    
-   Attribute-Guided Face Generation Using Conditional CycleGAN, ECCV2018,
    Yongyi Lu et al.
    [[PDF](https://arxiv.org/pdf/1705.09966.pdf)][[Web](http://www.cse.ust.hk/~yluaw/)]
    
-   Attribute Augmented Convolutional Neural Network for Face Hallucination, CVPRW2018,
    Cheng-Han Lee et al.
    [[PDF](http://openaccess.thecvf.com/content_cvpr_2018_workshops/supplemental/Lee_Attribute_Augmented_Convolutional_CVPR_2018_supplemental.pdf)][[Web](https://steven413d.github.io/)] 
    
-   Exemplar Guided Face Image Super-Resolution without Facial Landmarks, CVPRW 2019.

###### Blind face hallucination

-   [GFSRNet] Learning Warped Guidance for Blind Face Restoration, ECCV2018,
    Xiaoming Li et al. [[PDF](https://arxiv.org/pdf/1804.04829)][[Web](https://github.com/csxmli2016/GFRNet)]

-   To learn image super-resolution, use a GAN to learn how to do image
    degradation first, ECCV2018, Adrian Bulat et al.
    [[PDF](https://arxiv.org/abs/1807.11458)][[Web](https://github.com/jingyang2017/Face-and-Image-super-resolution)]

###### Discriminative face hallucination

-   SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face
    Hallucination, arXiv2018, Hsu et al.
    [[PDF](https://arxiv.org/pdf/1807.08370)]

-   Face hallucination using cascaded super-resolution and identity priors,
    arXiv2018, K. Grm et al. [[PDF](https://arxiv.org/pdf/1805.10938)]

-   Super-Identity Convolutional Neural Network for Face Hallucination,
    ECCV2018, Kaipeng Zhang et al. [[PDF]](https://arxiv.org/pdf/1811.02328.pdf)[[Web]](http://kpzhang93.github.io/)
    
-   Verification of Very Low-Resolution Faces Using An Identity-Preserving Deep Face Super-resolution Network,
    TR2018-116, Esra Ataer-Cansizoglu et al. [[PDF]](http://www.merl.com/publications/docs/TR2018-116.pdf)
    
-   FH-GAN: Face Hallucination and Recognition using Generative Adversarial Network, arXiv 2019.
    
-   SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination, Chih-Chung Hsu et al., TIP 2019. [[Code](https://github.com/jesse1029/SiGAN)]
    

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

-   [CMU+MIT](https://github.com/junjun-jiang/Face-Hallucination-Benchmark)*

-   [WHU-SCF](https://github.com/junjun-jiang/Face-Hallucination-Benchmark)* The last two databases are collected by myself.


###### Largescale databases

-   [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

-   [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/)

-   [Menpo](https://www.menpo.org/)

-   [Widerface](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

-   [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

-   [VGGFace2](https://arxiv.org/abs/1710.08092)

-   [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
