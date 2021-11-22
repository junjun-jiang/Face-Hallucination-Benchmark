# Face-Hallucination-Benchmark
A list of face face super-resolution/hallucination resources collected by [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun). If you find these resources are useful, please cite our following sruvey paper.

## Survey paper
J. Jiang, C. Wang, X. Liu, and J. Ma, “Deep Learning-based Face Super-resolution: A Survey,” ACM Computing Surveys, In Press. [arXiv](https://arxiv.org/abs/2101.03749).

```
@article{jiang2021survey
  title={Deep Learning-based Face Super-resolution: A Survey},
  author={Jiang, Junjun and Wang, Chenyang and Liu, Xianming and Ma, Jiayi},
  journal={ACM Computing Surveys},
  volume={},
  number={},
  pages={},
  year={2021}
}
```

*Some classical algorithms (including NE, LSR, SR, LcR, LINE, TLcR-RL, and EigTran) implemented by myself can be found [here](https://github.com/junjun-jiang/TLcR-RL).

*As for deep learning-based methods, we provide the training sets, and the experimental results of several state-of-the-art methods in  [[Baidu Drive](https://pan.baidu.com/s/1ox742-xGn6q3_YyttwbB9w)](va2i).  Note that the partition of the dataset follows [[DIC](https://github.com/Maclory/Deep-Iterative-Collaboration)]. The eval_psnr_ssim.py and calc_lpips.py are built on [[DIC](https://github.com/Maclory/Deep-Iterative-Collaboration)] and [[LPIPS](https://github.com/richzhang/PerceptualSimilarity)]. We thank the authors for sharing their codes. 

## Classical Methods


### Classical Patch-based Methods

-   Hallucinating face, S. Baker and T. Kanade, FG 2000.
    [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=840616)]

-   [NE] Super-resolution through neighbor embedding, Chang et al. CVPR 2004.
    [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [LSR] Hallucinating face by position-patch, Ma et al., PR 2010.
    [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [SR] Position-patch based face hallucination using convex optimization, C. Jung et al., SPL 2010. [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   [LcR] Noise robust face hallucination via locality-constrained representation, J. Jiang et al., TMM 2014.[[Web](https://github.com/junjun-jiang)]

-   [LINE] Multilayer Locality-Constrained Iterative Neighbor Embedding, J. Jiang et al., TIP 2014. [[Web](https://github.com/junjun-jiang)]

-   Face Hallucination Using Linear Models of Coupled Sparse Support, R. A. Farrugia et al., TIP 2017.[[PDF](https://ieeexplore.ieee.org/document/7953547/)][[Web](https://www.um.edu.mt/staff/reuben.farrugia)]

-   Hallucinating Face Image by Regularization Models in High-Resolution Feature Space, J. Shi et al., TIP 2018.
    [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8310603)]
    
-   [TLcR-RL] Context-Patch based Face Hallucination via Thresholding Locality-Constrained Representation and Reproducing Learning, J. Jiang et al., TCYB 2018. [[PDF]](https://arxiv.org/abs/1809.00665)[[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   Face Hallucination via Coarse-to-Fine Recursive Kernel Regression Structure, J. Shi et al. TMM 2019.

-   Robust Face Image Super-Resolution via Joint Learning of Subdivided Contextual Model, L. Chen et al. TIP 2019. [[PDF](https://ieeexplore.ieee.org/abstract/document/8733990)]

-   SSR2: Sparse signal recovery for single-image super-resolution on faces with extreme low resolutions, R. Abiantun et al. PR 2019. [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320319300597)]

-   Robust face hallucination via locality-constrained multiscale coding, L. Liu et al., INS 2020.

-   Face hallucination via multiple feature learning with hierarchical structure, L. Liu et al., INS 2020.

-   Hallucinating Color Face Image by Learning Graph Representation in Quaternion Space, L. Liu et al., TCYB 2021.



### Classical Global Face Methods

-   [EigTran] Hallucinating face by eigentransformation, X. Wang et al., TSMC-C 2005 [[Web](https://github.com/junjun-jiang/TLcR-RL)]

-   Super-resolution of face images using kernel PCA-based prior, A. Chakrabarti et al., TMM 2007. [[PDF](http://ieeexplore.ieee.org/document/4202583/)]

-   A Bayesian Approach to Alignment-Based Image Hallucination, C. Liu et al., ECCV 2012.[[PDF](https://people.csail.mit.edu/celiu/pdfs/ECCV12-ImageHallucination.pdf)]

-   A convex approach for image hallucination, P. Innerhofer et al., AAPRW 2013.[[Code](http://www.escience.cn/system/file?fileId=88901)]

-   Structured face hallucination, Y. Yang et al., CVPR 2013.[[Web](https://eng.ucmerced.edu/people/cyang35/CVPR13/CVPR13.html)]

### Classical Two-Step Methods

-   A two-step approach to hallucinating faces: global parametric model and local nonparametric model, C. Liu et al., CVPR 2001.[[Web](https://people.csail.mit.edu/celiu/FaceHallucination/fh.html)]

-   Hallucinating faces: LPH super-resolution and neighbor reconstruction for residue compensation, Y. Zhuang et al., PR 2007.[[PDF](https://www.sciencedirect.com/science/article/pii/S0031320307001355)]

-   [CCA] Super-resolution of human face image using canonical correlation analysis, H. Huang et al., PR 2010.[[PDF](https://www.sciencedirect.com/science/article/pii/S0031320310000853)]

## Deep learning-based Methods

### General FSR Methods

-   [BCCNN] Learning Face Hallucination in the Wild, E. Zhou et al., AAAI 2015.

-   [URDGN] Ultra-resolving face images by discriminative generative networks, X. Yu et al., CVPR 2016. [[Web](https://github.com/XinYuANU)]

-   [SRCNN-IBP] Face Hallucination Using Convolutional Neural Network with Iterative Back Projection, D. Huang et al., CCBR 2016.

-   [GLN] Global-Local Face Upsampling Network, O. Tuzel et al., ArXiv 2016.

-   [GLFSR] Global-local fusion network for face super-resolution, Tao Lu et al., Neurocomputing 2020.

-   Patch-based face hallucination with multitask deep neural network, W. Ko et al., ICME 2016.

-   Face hallucination by deep traversal network, Z. Feng et at., ICPR 2016.

-   Face hallucination using region-based deep convolutional networks, T. Lu et al., ICIP 2017.

-   Face Super-Resolution Through Wasserstein GANs. Z. Chen et al., ArXiv 2017.

-   High-Quality Face Image SR Using Conditional Generative Adversarial Networks, B. Huang et al., ArXiv 2017.

-   [WaSRNet] Wavelet-SRNet: A Wavelet-Based CNN for Multi-Scale Face Super Resolution, H. Huang et al., ICCV 2017. 

-   [Attention-FH] Attention-Aware Face Hallucination via Deep Reinforcement Learning, Q. Cao et al., CVPR 2017. [[PDF](https://arxiv.org/abs/1708.03132)][[Web](https://github.com/ykshi/facehallucination)]

-   Super-resolution Reconstruction of Face Image Based on Convolution Network, W. Huang et al., AISC 2018.

-   [LRGAN] To learn image super-resolution, use a GAN to learn how to do image
    degradation first, A.Bulat et al., ECCV 2018.
    [[PDF](https://arxiv.org/abs/1807.11458)][[Web](https://github.com/jingyang2017/Face-and-Image-super-resolution)]
    
-   A Noise Robust Face Hallucination Framework Via Cascaded Model of Deep Convolutional Networks and Manifold Learning, L. Han et al., ICME 2018

-   Face Hallucination via Convolution Neural Network, H. Nie et al., ICTAI 2018.
    
-   Face Hallucination by Attentive Sequence Optimization with Reinforcement Learning, Yukai Shi et al. TPAMI 2019.

-   Joint Face Hallucination and Deblurring via Structure Generation and Detail Enhancement, Yibing Song et al. IJCV 2019. [[Web](https://ybsong00.github.io/ijcv19_fhd/)]

-   Sequential Gating Ensemble Network for Noise Robust Multiscale Face Restoration, Z. chen et al., TCYB 2019.

-   Face Image Super-Resolution Using Inception Residual Network and GAN Framework, S. D. Indradi et al., ICOICT 2019.

-   Guided Cyclegan Via Semi-Dual Optimal Transport for Photo-Realistic Face Super-Resolution, W. Zheng et al., ICIP 2019.

-   ATMFN: Adaptive-threshold-based Multi-model Fusion Network for Compressed Face Hallucination, K. Jiang et al., TMM 2019.

-   [SRDSI] Face hallucination from low quality images using definition-scalable inference, X. Hu et al. PR 2019.

-   RBPNET: An asymptotic Residual Back-Projection Network for super-resolution of very low-resolution face image, X. Wang et al., Neurocomputing 2020.

-   Efficient Face Super-Resolution Based on Separable Convolution Projection Networks, X. Chen et al., CRC 2020.

-   A Densely Connected Face Super-Resolution Network Based on Attention Mechanism, Y. Liu et al., ICIEA 2020.

-   [HiFaceGAN] Implicit Subspace Prior Learning for Dual-Blind Face Restoration, L. Yang et al., ArXiv 2020.

-   Super-resolving Tiny Faces with Face Feature Vectors, Y. Lu et al., ICIST 2020.

-   [SPARNet]Learning Spatial Attention for Face Super-Resolution, C. Chen et al., TIP 2020. [[Web](https://github.com/chaofengc/Face-SPARNet)]

-   PCA-SRGAN: Incremental Orthogonal Projection Discrimination for Face Super-resolution, H. Du et al., ACM MM 2020.

-   [SPGAN] Supervised Pixel-Wise GAN for Face Super-Resolution, M. Zhang et al., TMM 2020.

-   Robust Super-Resolution of Real Faces using Smooth Features, S. Goswami et al., ECCVW 2020.

-   Learning wavelet coefficients for face super-resolution, Y. Liu et al., VC 2020.

-   PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models, S. Memon et al,. CVPR 2020.

-   Characteristic Regularisation for Super-Resolving Face Images, Z. Cheng et al., WACV 2020.

-   [DPDFN] Dual-path deep fusion network for face image hallucination, K. Jiang, TMM 2020.

-   Real-World Super-Resolution of Face-Images from Surveillance Cameras, A. Aakerberg et al., ArXiv 2021.

-   GLEAN: Generative Latent Bank for Large-Factor Image Super-Resolution, K. C. K. Chan et al., CVPR 2021.

-   [GFP-GAN] Towards Real-World Blind Face Restoration with Generative Facial Prior, X. Wang et al., CVPR 2021.

-   [GPEN] GAN Prior Embedded Network for Blind Face Restoration in the Wild, T. Yang et al., CVPR 2021.

-   E-ComSupResNet: Enhanced Face Super-Resolution Through Compact Network, E. Chudasama et al., TBIOM 2021.

-   [MLGE] Multi-Laplacian GAN with Edge Enhancement for Face Super Resolution, S. Ko et al., ICPR 2021. 

    
### Prior-guided FSR Methods

-   [CBN] Deep cascaded bi-network for face hallucination, S. Zhu et
    al., ECCV 2016. [[PDF](https://arxiv.org/abs/1607.05046)][[Web](https://github.com/Liusifei/ECCV16-CBN)]
    
-   [KPEFH] Face Hallucination Based on Key Parts Enhancement, K. Li et al., ICASSP 2018.
    
-   [LCGE] Learning to hallucinate face images via component generation and
    enhancement, Y. Song et al., IJCAI 2017
    [[PDF](https://arxiv.org/abs/1708.00223)][[Web](http://www.cs.cityu.edu.hk/~yibisong/)]
    
-   [MNCEFH] Deep CNN Denoiser and Multi-layer Neighbor Component Embedding for Face Hallucination, J. Jiang et al., IJCAI 2018. 
    [[PDF](https://arxiv.org/abs/1806.10726)][[Web](https://github.com/junjun-jiang/IJCAI-18)] 

-   [FSRNet] FSRNet: End-to-End learning face super-resolution with facial
    priors, Y. Chen et al., CVPR 2018. [[PDF](https://arxiv.org/abs/1711.10703)][[Web](https://github.com/tyshiwo/FSRNet)]

-   Super-FAN: integrated facial landmark localization and super-resolution of
    real-world low resolution faces in arbitrary poses with GANs
    A. Bulat et al., CVPR 2018. [[PDF](https://arxiv.org/abs/1712.02765)][[Web](https://github.com/1adrianb)]

-   [FSRGFCH] Face super-resolution guided by facial component heatmaps, ECCV 2018, X. Yu
    et al.
    [[PDF](https://ivul.kaust.edu.sa/Documents/Publications/2018/Face%20Super%20resolution%20Guided%20by%20Facial%20Component%20Heatmaps.pdf)]
    [[Web](https://github.com/XinYuANU)]
    
-   A coarse-to-fine face hallucination method by exploiting facial prior knowledge, ICIP 2018,
    Mengyan Li et al. [[PDF](https://github.com/lemoner20/ICIP2018/blob/master/Li.pdf)][[Web](https://github.com/lemoner20/ICIP2018)]
    
-   [PFSRNet] Progressive Face Super-Resolution via Attention to Facial Landmark, D. Kim et al., BMVC 2019. [[PDF](https://arxiv.org/pdf/1908.08239.pdf)][[Code](https://github.com/DeokyunKim/Progressive-Face-Super-Resolution)]

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

    
### Attribute-constrained FSR Methods

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
    
### Idnetity-preserving FSR Methods

-   [SICNN] Super-Identity Convolutional Neural Network for Face Hallucination, K. Zhang et al., ECCV 2018. [[PDF]](https://arxiv.org/pdf/1811.02328.pdf)[[Web]](http://kpzhang93.github.io/)

-   [FH-GAN] FH-GAN: Face Hallucination and Recognition Using Generative Adversarial Network, B. Bayramli et al., NIP 2019.

-   [WaSRGAN] Wavelet domain generative adversarial network for multi-scale face hallucination, H. Huang et al., IJCV 2019. [[Code](https://github.com/hhb072/WaveletSRNet)]

-   Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination, S. Lai et al., ICIP 2019.

-   [IPFH] Identity-Preserving Face Hallucination via Deep Reinforcement Learning, X. Cheng et al., TCSVT 2019.

-   Verification of Very Low-Resolution Faces Using An Identity-Preserving Deep Face Super-resolution Network, E. Ataer-Cansizoglu et al., ArXiv 2019.

-   Optimizing Super Resolution for Face Recognition, A. A. Abello et al., SIBGRAPI 2019.

-   SiGAN: Siamese Generative Adversarial Network for Identity-Preserving Face Hallucination, C.Hsu et al., TIP 2019. [[Code](https://github.com/jesse1029/SiGAN)]

-   [IADFH] Identity-Aware Deep Face Hallucination via Adversarial Face Verification, H. Kazemi et al., BTAS 2019. 

-   [C-SRIP] Face Hallucination Using Cascaded Super-Resolution and Identity Priors, K. Grm et al., TIP 2020.

-   [SPGAN] Supervised Pixel-Wise GAN for Face Super-Resolution, M. Zhang et al., TMM 2020.

-   Identity-Aware Face Super-Resolution for Low-Resolution Face Recognition, J. Chen et al., SPL 2020.

-   Face Super-Resolution Through Dual-Identity Constraint, F. Cheng et al., ICME 2021.

-   Edge and identity preserving network for face super-resolution, J. Kim et al., Neurocomputing 2021.


### Reference FSR Methods

-   [GFRNet] Learning Warped Guidance for Blind Face Restoration, X. Li et al., ECCV 2019.
    
-   [GWAInet] Exemplar Guided Face Image Super-Resolution without Facial Landmarks, CVPRW 2019.

-   [JSRFC] Recovering Extremely Degraded Faces by Joint Super-Resolution and Facial Composite, X. Li et al., ICTAI 2019.

-   [ASFFNet] Enhanced Blind Face Restoration With Multi-Exemplar Images and Adaptive Spatial Feature Fusion, X. Li et al., CVPR 2020.[[Web](https://github.com/csxmli2016/ASFFNet)]

-   [MEFSR] Multiple Exemplars-based Hallucination for Face Super-resolution and Editing, K. Wang et al., ACCV 2020.

-   [DFDNet] Blind Face Restoration via Deep Multi-scale Component Dictionaries, X. Li et al. ECCV 2020. [[Web](https://github.com/csxmli2016/DFDNet)]

### Joint Tasks

#### Joint Face Completion and Super-resolution

-   Hallucinating very low-resolution and obscured face images, L. Yang et al., ArXiv 2018.

-   FCSR-GAN: End-to-end Learning for Joint Face Completion and Super-resolution, J. Cai et al., FG 2019.

-   FCSR-GAN: Joint Face Completion and Super-Resolution via Multi-Task Learning, J. Cai et al., TBIOM 2020.

-   [MFG-GAN] Joint Face Completion and Super-resolution using Multi-scale Feature Relation Learning, Z. Liu et al., ArXiv 2020.

-   Pro-UIGAN: Progressive Face Hallucination from Occluded Thumbnails, Y. Zhang et al., ArXiv 2021.


#### Joint Face Deblurring and Super-resolution 

-   Learning to Super-Resolve Blurry Face and Text Images, X. Yu et al., ICCV 2017.
-   Joint face hallucination and deblurring via structure generation and detail enhancement, Y. Song et al., IJCV 2019.
-   [DGFAN] Deblurring And Super-Resolution Using Deep Gated Fusion Attention Networks For Face Images, C. H. Yang et al., ICASSP 2020. 
-   Super-resolving blurry face images with identity preservation, Y. Xu et al., PRL 2021.

#### Joint Face Alignment and Super-resolution

-   [TDAE] Hallucinating very low-resolution unaligned and noisy face images, X. Yu et al., CVPR 2017. [[Web](https://github.com/XinYuANU)]    
-   [TDN] Hallucinating very low-resolution unaligned and noisy face images by
    transformative discriminative autoencoders, X. Yu et al., AAAI 2017.[[Web](https://github.com/XinYuANU)]  
-   [MTDN] Hallucinating Unaligned Face Images by Multiscale Transformative Discriminative Networks, X. Yu et al., IJCV 2021.


#### Joint Illumination Compensation and Super-resolution

-   [SeLENet] SeLENet: A Semi-Supervised Low Light Face Enhancement Method for Mobile Face Unlock, H. A. Le et al., ICB 2019.
-  Learning To See Faces In The Dark，X. Ding et al., ICME 2020.
-   [CPGAN] Copy and paste GAN: Face hallucination from shaded thumbnails, Y. Zhang et al., CVPR 2020.
-   Recursive Copy and Paste GAN: Face Hallucination from Shaded Thumbnails, Y. Zhang et al., TPAMI 2021.
-   Network Architecture Search for Face Enhancement, R. Yasarla et al., ArXiv 2021.


#### Joint Face Fronlization and Super-resolution

-   Can We See More? Joint Frontalization and Hallucination of Unaligned Tiny Faces, X. Yu et al. TPAMI 2019.
-   Face Hallucination With Finishing Touches, Y. Zhang et al., TIP 2021.
-   Joint Face Image Restoration and Frontalization for Recognition, X. Tu et al., TCSVT 2021.


### Related Applications

#### Face Video Super-resolution

-   Face video super-resolution with identity guided generative adversarial networks, D. Li et al., CCCV 2017.
-   Super-resolution of Very Low-Resolution Faces from Videos, E. Ataer-Cansizoglu et al., BMVC 2018.
-   Video Face Super-Resolution with Motion-Adaptive Feedback Cell, J. Xin et al., AAAI 2020.
-   Self-Enhanced Convolutional Network for Facial Video Hallucination, C. Fang et al., TIP 2020.
-   VidFace: A Full-Transformer Solver for Video FaceHallucination with Unaligned Tiny Snapshots, Y. GAN et al., ArXiv 2021.
-   [MDVDNet] Multi-modality Deep Restoration of Extremely Compressed Face Videos, X. Zhang et al., ArXiv 2021.

#### Old Photo Restoration

-   [BOPBL] Bringing Old Photos Back to Life, Z. Wan et al., CVPR 2020.

#### Audio-guided FSR

-   Learning to Have an Ear for Face Super-Resolution, G. Meishvili et al., CVPR 2020.

#### 3D FSR

-   Super-resolution of 3D face, G. Fan et al., ECCV 2006.
-   3D face hallucination from a single depth frame, L. Shu et al., 3DV 2014.
-   Robust 3D patch-based face hallucination, C. Qu et al., WACV 2017.
-   3D Face Point Cloud Super-Resolution Network, J. Li et al., IJCB 2021.

#### Hyperspectral FSR

-   [SSANet]Spectral Splitting and Aggregation Network for Hyperspectral Face Super-Resolution, J. Jiang et al., arXiv 2021. [[PDF](https://arxiv.org/abs/2108.13584)]

#### Cross-Domain Face Miniatures 
-   [DAR-FSR]Super-Resolving Cross-Domain Face Miniatures by Peeking at One-Shot Exemplar, P, Li et al., ICCV 2021. [[PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Super-Resolving_Cross-Domain_Face_Miniatures_by_Peeking_at_One-Shot_Exemplar_ICCV_2021_paper.pdf)]

## Image Quality Measurement

-   RMSE, PSNR, SSIM, LPIPS, NIQE, FID

-   Face recognition rate

-   Mean Opinion Score (MOS)

## Databases

### Classical databases

-   [FERET](http://www.nist.gov/itl/iad/ig/colorferet.cfm)

-   [CMU-PIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html)

-   [CAS-PEAL-R1](http://www.jdl.ac.cn/peal/JDL-PEAL-Release.htm)

-   [FEI](https://fei.edu.br/~cet/facedatabase.html)

-   [CMU+MIT](https://github.com/junjun-jiang/Face-Hallucination-Benchmark)*

-   [WHU-SCF](https://github.com/junjun-jiang/Face-Hallucination-Benchmark)* 
The last two databases are collected by myself.


### Largescale databases

-   [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

-   [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/)

-   [Menpo](https://www.menpo.org/)

-   [Widerface](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

-   [AFLW](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/aflw/)

-   [VGGFace2](https://arxiv.org/abs/1710.08092)

-   [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)
