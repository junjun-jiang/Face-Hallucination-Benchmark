# Face-Hallucination-Benchmark
A list of face hallucination/face super-resolution resources collected by [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun).

#### Survey paper
J. Jiang, C. Wang, X. Liu, and J. Ma, “Deep Learning-based Face Super-resolution: A Survey,” accepted to ACM Computing Surveys. [arXiv](https://arxiv.org/abs/2101.03749).

#### Classical Methods
Some classical algorithms (including NE, LSR, SR, LcR, LINE, TLcR-RL, and EigTran) 
implemented by myself can be found [here](https://github.com/junjun-jiang/TLcR-RL).

###### Classical Patch-based Methods

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

-   SSR2: Sparse signal recovery for single-image super-resolution on faces with extreme low resolutions, RamziAbiantun et al. PR2019. [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300597)]

-   Robust face hallucination via locality-constrained multiscale coding, L. Liu et al., INS 2020.

-   Face hallucination via multiple feature learning with hierarchical structure, L. Liu et al., INS 2020.

-   Hallucinating Color Face Image by Learning Graph Representation in Quaternion Space, L. Liu et al., TCYB 2021.



###### Classical Global Face Methods

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

###### Classical Two-Step Methods

-   A two-step approach to hallucinating faces: global parametric model and
    local nonparametric model, CVPR2001, Ce Liu et al.
    [[Web](https://people.csail.mit.edu/celiu/FaceHallucination/fh.html)]

-   Hallucinating faces: LPH super-resolution and neighbor reconstruction for
    residue compensation, PR2007, Yuting Zhuang et al.
    [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320307001355)]

-   [CCA] Super-resolution of human face image using canonical correlation
    analysis, PR2010, Hua Huang et al.
    [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320310000853)]

#### Deep learning-based Methods
###### General FSR Methods

-   [URDGN] Ultra-resolving face images by discriminative generative networks,
    in ECCV2016, Xin Yu et al. [[Web](https://github.com/XinYuANU)]

-   [TDAE] Hallucinating very low-resolution unaligned and noisy face images,
    CVPR2017, Xin Yu et al. [[Web](https://github.com/XinYuANU)]    

-   [TDN] Hallucinating very low-resolution unaligned and noisy face images by
    transformative discriminative autoencoders, AAAI2017, Xin Yu et al.
    [[Web](https://github.com/XinYuANU)]
    
-   Face Hallucination by Attentive Sequence Optimization with Reinforcement Learning, Yukai Shi et al. PAMI 2019.

-   Joint Face Hallucination and Deblurring via Structure Generation and Detail Enhancement, Yibing Song et al. IJCV 2019. [[Web](https://ybsong00.github.io/ijcv19_fhd/)]

-   [WaSRNet] Wavelet-SRNet: A Wavelet-Based CNN for Multi-Scale Face Super Resolution, H. Huang et al., ICCV 2017. 

-   Face hallucination from low quality images using definition-scalable inference, Xiao Hu et al. PR 2019.

-   [Attention-FH] Attention-Aware Face Hallucination via Deep Reinforcement Learning,
    CVPR2017, Qingxing Cao et al. [[PDF](https://arxiv.org/abs/1708.03132)][[Web](https://github.com/ykshi/facehallucination)]

-   Can We See More? Joint Frontalization and Hallucination of Unaligned Tiny Faces, Xin Yu et al. PAMI 2019.

-   Copy and paste GAN: Face hallucination from shaded thumbnails, Y. Zhang et al., CVPR 2020.

-   Recursive Copy and Paste GAN: Face Hallucination from Shaded Thumbnails, Y. Zhang et al., PAMI 2021.

-   Face Hallucination With Finishing Touches, Y. Zhang et al., TIP 2021.

-   [DPDFN] Dual-path deep fusion network for face image hallucination, K. Jiang, TMM 2020.

-   Cross-spectral face hallucination via disentangling independent factors, B. Duan et al., CVPR 2020.

-   Hallucinating Unaligned Face Images by Multiscale Transformative Discriminative Networks, X. Yu et al., IJCV 2021.

-   To learn image super-resolution, use a GAN to learn how to do image
    degradation first, ECCV2018, Adrian Bulat et al.
    [[PDF](https://arxiv.org/abs/1807.11458)][[Web](https://github.com/jingyang2017/Face-and-Image-super-resolution)]


    
###### Prior-guided FSR Methods

-   [CBN] Deep cascaded bi-network for face hallucination, S. Zhu et
    al., ECCV 2016. [[PDF](https://arxiv.org/abs/1607.05046)][[Web](https://github.com/Liusifei/ECCV16-CBN)]
    
-   [KPEFH] Face Hallucination Based on Key Parts Enhancement, K. Li et al., ICASSP 2018.
    
-   [LCGE] Learning to hallucinate face images via component generation and
    enhancement, IJCAI2017, Y. Song et al.
    [[PDF](https://arxiv.org/abs/1708.00223)][[Web](http://www.cs.cityu.edu.hk/~yibisong/)]
    
-   [MNCEFH] Deep CNN Denoiser and Multi-layer Neighbor Component Embedding for Face Hallucination, IJCAI2018, Junjun Jiang et al.
    [[PDF](https://arxiv.org/abs/1806.10726)][[Web](https://github.com/junjun-jiang/IJCAI-18)] 

-   [FSRNet] FSRNet: End-to-End learning face super-resolution with facial
    priors, CVPR, 2018 Yu Chen et al. [[PDF](https://arxiv.org/abs/1711.10703)][[Web](https://github.com/tyshiwo/FSRNet)]

-   Super-FAN: integrated facial landmark localization and super-resolution of
    real-world low resolution faces in arbitrary poses with GANs, CVPR2018,
    Adrian Bulat et al. [[PDF](https://arxiv.org/abs/1712.02765)][[Web](https://github.com/1adrianb)]

-   [FSRGFCH] Face super-resolution guided by facial component heatmaps, ECCV2018, Xin Yu
    et al.
    [[PDF](https://ivul.kaust.edu.sa/Documents/Publications/2018/Face%20Super%20resolution%20Guided%20by%20Facial%20Component%20Heatmaps.pdf)]
    [[Web](https://github.com/XinYuANU)]
    
-   A coarse-to-fine face hallucination method by exploiting facial prior knowledge, ICIP2018,
    Mengyan Li et al. [[PDF](https://github.com/lemoner20/ICIP2018/blob/master/Li.pdf)][[Web](https://github.com/lemoner20/ICIP2018)]
    
-   [PFSRNet] Progressive Face Super-Resolution via Attention to Facial Landmark, D. Kim et al., BMVC2019. [[PDF](https://arxiv.org/pdf/1908.08239.pdf)][[Code](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution)]

-   [JASRNet] Joint Super-Resolution and Alignment of Tiny Faces, Y. Yin et al. AAAI 2019.

-   Component Attention Guided Face Super-Resolution Network: CAGFace, R. Kalarot et al., WACV 2020.

-   SAAN: Semantic Attention Adaptation Network for Face Super-Resolution, T. Zhao et al., ICME 2020.

-   [PMGMSAN] Parsing Map Guided Multi-Scale Attention Network For Face Hallucination, C. Wang et al., ICASSP 2020.

-   [ATSENet] Learning Face Image Super-Resolution through Facial Semantic Attribute Transformation and Self-Attentive Structure Enhancement, M. Li et al., TMM 2020.

-   [DIC] Deep Face Super-Resolution With Iterative Collaboration Between Attentive Recovery and Landmark Estimation, Cheng Ma et al., CVPR 2020.

-   MSFSR: A Multi-Stage Face Super-Resolution with Accurate Facial Representation via Enhanced Facial Boundaries, Y. Zhang et al., CVPRW 2020.

-   Semantic-driven Face Hallucination Based on Residual Network, X. Yu et al., TBIOM 2021

-   Progressive Semantic-Aware Style Transformation for Blind Face Restoration, C. Chen et al., CVPR 2021

-   [HapFSR] Heatmap-Aware Pyramid Face Hallucination, C. Wang et al. ICME 2021.

-   [OBC-FSR] Organ-Branched CNN for Robust Face Super-Resolution, J. Li et al., ICME 2021.

-   [HCRF] Features Guided Face Super-Resolution via Hybrid Model of Deep Learning and Random Forests, Z. S. Liu et al., TIP 2021.

-   DCLNet: Dual Closed-loop Networks for face super-resolution, H. Wang et al., KBS 2021.

-   Progressive face super-resolution with cascaded recurrent convolutional network, S. Liu et al., Neurocomputing 2021.

-   Face Super-Resolution Network with Incremental Enhancement of Facial Parsing Information, S. Liu et al., ICPR 2021.

-   Unsupervised face super-resolution via gradient enhancement and semantic guidance, L. Li et al., VC 2021.






    
###### Attribute-constrained FSR Methods

-   [FaceAttr] Super-resolving very low-resolution face images with
    supplementary attributes, CVPR2018, Xin Yu et al.
    [[PDF](https://basurafernando.github.io/papers/XinYuCVPR18.pdf)][[Web](https://github.com/XinYuANU)]
    
-   Attribute-Guided Face Generation Using Conditional CycleGAN, ECCV2018,
    Yongyi Lu et al.
    [[PDF](https://arxiv.org/pdf/1705.09966.pdf)][[Web](http://www.cse.ust.hk/~yluaw/)]
    
-   Attribute Augmented Convolutional Neural Network for Face Hallucination, CVPRW2018,
    Cheng-Han Lee et al.
    [[PDF](http://openaccess.thecvf.com/content_cvpr_2018_workshops/supplemental/Lee_Attribute_Augmented_Convolutional_CVPR_2018_supplemental.pdf)][[Web](https://steven413d.github.io/)] 
    
-   Residual Attribute Attention Network for Face Image Super-Resolution, Jingwei Xin et al. AAAI2019. [[PDF](https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4937)]

-   [ATNet] Deep Learning Face Hallucination via Attributes Transfer and Enhancement, M. Li et al., ICME 2019.

-   [FACN] Facial Attribute Capsules for Noise Face Super Resolution, J. Xin et al., AAAI 2020.

-   [ATSENet] Learning Face Image Super-Resolution through Facial Semantic Attribute Transformation and Self-Attentive Structure Enhancement, M. Li et al., TMM 2020.
    
###### Idnetity-preserving FSR Methods

-   [SICNN] Super-Identity Convolutional Neural Network for Face Hallucination, K. Zhang et al., ECCV 2018. [[PDF]](https://arxiv.org/pdf/1811.02328.pdf)[[Web]](http://kpzhang93.github.io/)

-   [FH-GAN] FH-GAN: Face Hallucination and Recognition Using Generative Adversarial Network, B. Bayramli et al., NIP 2019.

-   [WaSRGAN] Wavelet domain generative adversarial network for multi-scale face hallucination, H. Huang et al., IJCV 2019. [[Code](https://github.com/hhb072/WaveletSRNet)]

-   Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination, S. Lai et al., ICIP 2019.

-   [IPFH] Identity-Preserving Face Hallucination via Deep Reinforcement Learning, X. Cheng et al., TCSVT 2019.

-   Optimizing Super Resolution for Face Recognition, A. A. Abello et al., SIBGRAPI 2019.

-   SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination, C.Hsu et al., TIP 2019. [[Code](https://github.com/jesse1029/SiGAN)]

-   [IADFH] Identity-Aware Deep Face Hallucination via Adversarial Face Verification, H. Kazemi et al., BTAS 2019. 

-   [C-SRIP] Face Hallucination Using Cascaded Super-Resolution and Identity Priors, K. Grm et al., TIP 2020.

-   [SPGAN] Supervised Pixel-Wise GAN for Face Super-Resolution, M. Zhang et al., TMM 2020.

-   Identity-Aware Face Super-Resolution for Low-Resolution Face Recognition, J. Chen et al., SPL 2020.

-   Face Super-Resolution Through Dual-Identity Constraint, F. Cheng et al., ICME 2021.

-   Edge and identity preserving network for face super-resolution, J. Kim et al., Neurocomputing 2021.



###### Reference FSR Methods
-   [GFRNet] Learning Warped Guidance for Blind Face Restoration, X. Li et al., ECCV 2019.
    
-   [GWAInet] Exemplar Guided Face Image Super-Resolution without Facial Landmarks, CVPRW 2019.

-   [JSRFC] Recovering Extremely Degraded Faces by Joint Super-Resolution and Facial Composite, X. Li et al., ICTAI 2019.

-   [ASFFNet] Enhanced Blind Face Restoration With Multi-Exemplar Images and Adaptive Spatial Feature Fusion, X. Li et al., CVPR 2020.[[Web](https://github.com/csxmli2016/ASFFNet)]

-   [MEFSR] Multiple Exemplars-based Hallucination for Face Super-resolution and Editing, K. Wang et al., ACCV 2020.

-   [DFDNet] Blind Face Restoration via Deep Multi-scale Component Dictionaries, X. Li et al. ECCV 2020. [[Web](https://github.com/csxmli2016/DFDNet)]

###### Discriminative face hallucination
-   Verification of Very Low-Resolution Faces Using An Identity-Preserving Deep Face Super-resolution Network,
    TR2018-116, Esra Ataer-Cansizoglu et al. [[PDF]](http://www.merl.com/publications/docs/TR2018-116.pdf)
    

#### Image Quality Measurement

-   RMSE, PSNR, SSIM, LPIPS, NIQE, FID

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
