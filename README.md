# AI for Time Series (AI4TS) Papers, Tutorials, and Surveys

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/qingsongedu/awesome-AI-for-time-series-papers)
[![Visits Badge](https://badges.pufler.dev/visits/qingsongedu/awesome-AI-for-time-series-papers)](https://badges.pufler.dev/visits/qingsongedu/awesome-AI-for-time-series-papers)
<!-- ![Forks](https://img.shields.io/github/forks/qingsongedu/awesome-AI-for-time-series-papers) -->

A professionally curated list of papers (with available code), tutorials, and surveys on recent **AI for Time Series Analysis (AI4TS)**, including Time Series, Spatio-Temporal Data, Event Data, Sequence Data, Temporal Point Processes, etc., at the **Top AI Conferences and Journals**, which is **updated ASAP (the earliest time)** once the accepted papers are announced in the corresponding top AI conferences/journals. Hope this list would be helpful for researchers and engineers who are interested in AI for Time Series Analysis.

The top conferences including:
- Machine Learning: NeurIPS, ICML, ICLR
- Data Mining: KDD, WWW
- Artificial Intelligence: AAAI, IJCAI
- Data Management: SIGMOD, VLDB, ICDE
- Misc (selected): AISTAT, CIKM, ICDM, WSDM, SIGIR, ICASSP, CVPR, ICCV, etc.

The top journals including (mainly for survey papers):
CACM, PIEEE, TPAMI, TKDE, TNNLS, TITS, TIST, SPM, JMLR, JAIR, CSUR, DMKD, KAIS, IJF, arXiv(selected), etc.

If you find any missed resources (paper/code) or errors, please feel free to open an issue or make a pull request. 

For general **Recent AI Advances: Tutorials and Surveys in various areas (DL, ML, DM, CV, NLP, Speech, etc.)** at the **Top AI Conferences and Journals**, please check [This Repo](https://github.com/qingsongedu/awesome-AI-tutorials-surveys). 

## Main Recent Update Note
- [Mar. 04, 2024] Add papers accepted by ICLR'24, AAAI'24, WWW'24!
- [Jul. 05, 2023] Add papers accepted by KDD'23!
- [Jun. 20, 2023] Add papers accepted by ICML'23!
- [Feb. 07, 2023] Add papers accepted by ICLR'23 and AAAI'23!
- [Sep. 18, 2022] Add papers accepted by NeurIPS'22!
- [Jul. 14, 2022] Add papers accepted by KDD'22!
- [Jun. 02, 2022] Add papers accepted by ICML'22, ICLR'22, AAAI'22, IJCAI'22!

## Table of Contents
- [AI4TS Tutorials and Surveys](#AI4TS-Tutorials-and-Surveys)
  * [AI4TS Tutorials](#AI4TS-Tutorials)
  * [AI4TS Surveys](#AI4TS-Surveys)

- [AI4TS Papers 2024](#AI4TS-Papers-2024)
  * [NeurIPS 2024](#NeurIPS-2024), [ICML 2024](#ICML-2024), [ICLR 2024](#ICLR-2024)
  * [KDD 2024](#KDD-2024), [WWW 2024](#WWW-2024), [AAAI 2024](#AAAI-2024), [IJCAI 2024](#IJCAI-2024)
  * [SIGMOD VLDB ICDE 2024](#SIGMOD-VLDB-ICDE-2024)
  * [Misc 2024](#Misc-2024)
 
- [AI4TS Papers 2023](#AI4TS-Papers-2023)
  * [NeurIPS 2023](#NeurIPS-2023), [ICML 2023](#ICML-2023), [ICLR 2023](#ICLR-2023)
  * [KDD 2023](#KDD-2023), [AAAI 2023](#AAAI-2023), [IJCAI 2023](#IJCAI-2023)
  * [SIGMOD VLDB ICDE 2023](#SIGMOD-VLDB-ICDE-2023)
  * [Misc 2023](#Misc-2023)

- [AI4TS Papers 2022](#AI4TS-Papers-2022)
  * [NeurIPS 2022](#NeurIPS-2022), [ICML 2022](#ICML-2022), [ICLR 2022](#ICLR-2022)
  * [KDD 2022](#KDD-2022), [AAAI 2022](#AAAI-2022), [IJCAI 2022](#IJCAI-2022)
  * [SIGMOD VLDB ICDE 2022](#SIGMOD-VLDB-ICDE-2022)
  * [Misc 2022](#Misc-2022)
 
- [AI4TS Papers 2021](#AI4TS-Papers-2021)
  * [NeurIPS 2021](#NeurIPS-2021), [ICML 2021](#ICML-2021), [ICLR 2021](#ICLR-2021)
  * [KDD 2021](#KDD-2021), [AAAI 2021](#AAAI-2021), [IJCAI 2021](#IJCAI-2021)
  * [SIGMOD VLDB ICDE 2021](#SIGMOD-VLDB-ICDE-2021)
  * [Misc 2021](#Misc-2021)

- [AI4TS Papers 201X-2020 Selected](#AI4TS-Papers-201X-2020-Selected)
  * [NeurIPS 201X-2020](#NeurIPS-201X-2020), [ICML 201X-2020](#ICML-201X-2020), [ICLR 201X-2020](#ICLR-201X-2020)
  * [KDD 201X-2020](#KDD-201X-2020), [AAAI 201X-2020](#AAAI-201X-2020), [IJCAI 201X-2020](#IJCAI-201X-2020)
  * [SIGMOD VLDB ICDE 201X-2020](#SIGMOD-VLDB-ICDE-201X-2020)
  * [Misc 201X-2020](#Misc-201X-2020)


## AI4TS Tutorials and Surveys
### AI4TS Tutorials
* Out-of-Distribution Generalization in Time Series, in *AAAI* 2024. [\[Link\]](https://ood-timeseries.github.io/)
* Robust Time Series Analysis and Applications: An Interdisciplinary Approach, in *ICDM* 2023. [\[Link\]](https://sites.google.com/view/timeseries-tutorial-icdm2023)
* Robust Time Series Analysis and Applications: An Industrial Perspective, in *KDD* 2022. [\[Link\]](https://qingsongedu.github.io/timeseries-tutorial-kdd-2022/)
* Time Series in Healthcare: Challenges and Solutions, in *AAAI* 2022. [\[Link\]](https://www.vanderschaar-lab.com/time-series-in-healthcare/)
* Time Series Anomaly Detection: Tools, Techniques and Tricks, in *DASFAA* 2022. [\[Link\]](https://www.dasfaa2022.org//tutorials/Time%20Series%20Anomaly%20Result%20Master%20File_Dasfaa_2022.pdf)
* Modern Aspects of Big Time Series Forecasting, in *IJCAI* 2021. [\[Link\]](https://lovvge.github.io/Forecasting-Tutorial-IJCAI-2021/)
* Explainable AI for Societal Event Predictions: Foundations, Methods, and Applications, in *AAAI* 2021. [\[Link\]](https://yue-ning.github.io/aaai-21-tutorial.html)
* Physics-Guided AI for Large-Scale Spatiotemporal Data, in *KDD* 2021. [\[Link\]](https://sites.google.com/view/kdd2021tutorial/home)
* Deep Learning for Anomaly Detection, in *KDD & WSDM* 2020. [\[Link1\]](https://sites.google.com/view/kdd2020deepeye/home) [\[Link2\]](https://sites.google.com/view/wsdm2020dlad) [\[Link3\]](https://www.youtube.com/watch?v=Fn0qDbKL3UI)
* Building Forecasting Solutions Using Open-Source and Azure Machine Learning, in *KDD* 2020. [\[Link\]](https://chenhuims.github.io/forecasting/)
* Interpreting and Explaining Deep Neural Networks: A Perspective on Time Series Data, *KDD* 2020. [\[Link\]](https://xai.kaist.ac.kr/Tutorial/2020/)
* Forecasting Big Time Series: Theory and Practice, *KDD* 2019. [\[Link\]](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* Spatio-Temporal Event Forecasting and Precursor Identification, *KDD* 2019. [\[Link\]](http://mason.gmu.edu/~lzhao9/projects/event_forecasting_tutorial_KDD)
* Modeling and Applications for Temporal Point Processes, *KDD* 2019. [\[Link1\]](https://dl.acm.org/doi/10.1145/3292500.3332298) [\[Link2\]](https://thinklab.sjtu.edu.cn/TPP_Tutor_KDD19.html)


### AI4TS Surveys
#### General Time Series Survey
* What Can Large Language Models Tell Us about Time Series Analysis, in *arXiv* 2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2310.10196) [\[Website\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* Deep Learning for Multivariate Time Series Imputation: A Survey, in *arXiv* 2024. [\[paper\]](https://arxiv.org/abs/2402.04059) [\[Website\]](https://github.com/wenjiedu/awesome_imputation)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)
* Transformers in Time Series: A Survey, in *IJCAI* 2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* Time series data augmentation for deep learning: a survey, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2002.12478)
* Neural temporal point processes: a review, in *IJCAI* 2021. [\[paper\]](https://arxiv.org/abs/2104.03528)
* Causal inference for time series analysis: problems, methods and evaluation, in *KAIS* 2022. [\[paper\]](https://scholar.google.com/scholar?cluster=15831734748668637115&hl=en&as_sdt=5,48&sciodt=0,48)
* Survey and Evaluation of Causal Discovery Methods for Time Series, in *JAIR* 2022. [\[paper\]](https://www.jair.org/index.php/jair/article/view/13428/26775)
* Deep learning for spatio-temporal data mining: A survey, in *TKDE* 2020. [\[paper\]](https://arxiv.org/abs/1906.04928)
* Generative Adversarial Networks for Spatio-temporal Data: A Survey, in *TIST* 2022. [\[paper\]](https://arxiv.org/abs/2008.08903)
* Spatio-Temporal Data Mining: A Survey of Problems and Methods, in *CSUR* 2018. [\[paper\]](https://dl.acm.org/doi/10.1145/3161602) 
* A Survey on Principles, Models and Methods for Learning from Irregularly Sampled Time Series, in *NeurIPS Workshop* 2020. [\[paper\]](https://arxiv.org/abs/2012.00168)
* Count Time-Series Analysis: A signal processing perspective, in *SPM* 2019. [\[paper\]](https://ieeexplore.ieee.org/document/8700675)
* Wavelet transform application for/in non-stationary time-series analysis: a review, in *Applied Sciences* 2019. [\[paper\]](https://www.mdpi.com/2076-3417/9/7/1345)
* Granger Causality: A Review and Recent Advances, in *Annual Review of Statistics and Its Application* 2014. [\[paper\]](https://www.annualreviews.org/doi/epdf/10.1146/annurev-statistics-040120-010930)
* A Review of Deep Learning Methods for Irregularly Sampled Medical Time Series Data, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2010.12493)
* Beyond Just Vision: A Review on Self-Supervised Representation Learning on Multimodal and Temporal Data, in *arXiv* 2022. [\[paper\]](https://arxiv.org/abs/2206.02353)
* A Survey on Time-Series Pre-Trained Models, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2305.10716) [\[link\]](https://github.com/qianlima-lab/time-series-ptms)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *arXiv* 2023. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)


#### Time Series Forecasting Survey
* Forecasting: theory and practice, in *IJF* 2022. [\[paper\]](https://www.sciencedirect.com/science/article/pii/S0169207021001758)
* Time-series forecasting with deep learning: a survey, in *Philosophical Transactions of the Royal Society A* 2021. [\[paper\]](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2020.0209)
* Deep Learning on Traffic Prediction: Methods, Analysis, and Future Directions, in *TITS* 2022. [\[paper\]](https://arxiv.org/abs/2004.08555)
* Event prediction in the big data era: A systematic survey, in *CSUR* 2022. [\[paper\]](https://dl.acm.org/doi/10.1145/3450287)
* A brief history of forecasting competitions, in *IJF* 2020. [\[paper\]](https://www.monash.edu/business/ebs/our-research/publications/ebs/wp03-2019.pdf)
* Neural forecasting: Introduction and literature overview, in *arXiv* 2020. [\[paper\]](https://arxiv.org/abs/2004.10240) 
* Probabilistic forecasting, in *Annual Review of Statistics and Its Application* 2014. [\[paper\]](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-062713-085831)

#### Time Series Anomaly Detection Survey
* A review on outlier/anomaly detection in time series data, in *CSUR* 2021. [\[paper\]](https://arxiv.org/abs/2002.04236)
* Anomaly detection for IoT time-series data: A survey, in *IEEE Internet of Things Journal* 2019. [\[paper\]](https://eprints.keele.ac.uk/7576/1/08926446.pdf)
* A Survey of AIOps Methods for Failure Management, in *TIST* 2021. [\[paper\]](https://jorge-cardoso.github.io/publications/Papers/JA-2021-025-Survey_AIOps_Methods_for_Failure_Management.pdf)
* Sequential (quickest) change detection: Classical results and new directions, in *IEEE Journal on Selected Areas in Information Theory* 2021. [\[paper\]](https://arxiv.org/abs/2104.04186)
* Outlier detection for temporal data: A survey, TKDE'13. [\[paper\]](https://romisatriawahono.net/lecture/rm/survey/machine%20learning/Gupta%20-%20Outlier%20Detection%20for%20Temporal%20Data%20-%202014.pdf)
* Anomaly detection for discrete sequences: A survey, TKDE'12. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/5645624)
* Anomaly detection: A survey, CSUR'09. [\[paper\]](https://arindam.cs.illinois.edu/papers/09/anomaly.pdf)
 
#### Time Series Classification Survey
* Deep learning for time series classification: a review, in *Data Mining and Knowledge Discovery* 2019. [\[paper\]](https://link.springer.com/article/10.1007/s10618-019-00619-1?sap-outbound-id=11FC28E054C1A9EB6F54F987D4B526A6EE3495FD&mkt-key=005056A5C6311EE999A3A1E864CDA986)
* Approaches and Applications of Early Classification of Time Series: A Review, in *IEEE Transactions on Artificial Intelligence* 2020. [\[paper\]](https://arxiv.org/abs/2005.02595)

[\[paper\]]()
## AI4TS Papers 2024
### NeurIPS 2024

### ICML 2024

### ICLR 2024
#### Time Series Forecasting
* Time-LLM: Time Series Forecasting by Reprogramming Large Language Models [\[paper\]](https://openreview.net/forum?id=Unb5CVPtae) [\[official code\]](https://github.com/kimmeen/time-llm/)
* TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series [\[paper\]](https://openreview.net/forum?id=Tuh4nZVb0g) [\[official code\]](https://github.com/scxsunchenxi/test)
* TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=YH5w12OUuU)
* DAM: A Foundation Model for Forecasting [\[paper\]](https://openreview.net/forum?id=4NhMhElWqP)
* CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=MJksrOhurE)
* Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=lJkOCMP2aW)
* GAFormer: Enhancing Timeseries Transformers Through Group-Aware Embeddings [\[paper\]](https://openreview.net/forum?id=c56TWtYp0W) 
* Transformer-Modulated Diffusion Models for Probabilistic Multivariate Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=qae04YACHs)
* iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=JePfAI8fah)
* RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies [\[paper\]](https://openreview.net/forum?id=ltZ9ianMth)
* ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis [\[paper\]](https://openreview.net/forum?id=vpJMJerXHU)
* TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=7oLshfEIC2)
* FITS: Modeling Time Series with 10k Parameters [\[paper\]](https://openreview.net/forum?id=bWcnvZ3qMb)
* Multi-Resolution Diffusion Models for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=mmjnr0G8ZY)
* MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process [\[paper\]](https://openreview.net/forum?id=CZiY6OLktd)
* Interpretable Sparse System Identification: Beyond Recent Deep Learning Techniques on Time-Series Prediction [\[paper\]](https://openreview.net/forum?id=aFWUY3E7ws)
* TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series [\[paper\]](https://openreview.net/forum?id=xtOydkE1Ku)
* Towards Transparent Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=TYXtXLYHpR)
* Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values [\[paper\]](https://openreview.net/forum?id=O9nZCwdGcG)
* Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators [\[paper\]](https://openreview.net/forum?id=JiTVtCUOpS)
* VQ-TR: Vector Quantized Attention for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=IxpTsFS7mh)
* Copula Conformal prediction for multi-step time series prediction [\[paper\]](https://openreview.net/forum?id=ojIJZDNIBj)
* ClimODE: Climate Forecasting With Physics-informed Neural ODEs [\[paper\]](https://openreview.net/forum?id=xuY33XhEGR)
* STanHop: Sparse Tandem Hopfield Model for Memory-Enhanced Time Series Prediction [\[paper\]](https://openreview.net/forum?id=6iwg437CZs)
* T-Rep: Representation Learning for Time Series using Time-Embeddings [\[paper\]](https://openreview.net/forum?id=3y2TfP966N)
* Periodicity Decoupling Framework for Long-term Series Forecasting [\[paper\]](https://openreview.net/forum?id=dp27P5HBBt)
* Self-Supervised Contrastive Forecasting [\[paper\]](https://openreview.net/forum?id=nBCuRzjqK7)

#### Others
* Explaining Time Series via Contrastive and Locally Sparse Perturbations [\[paper\]](https://openreview.net/forum?id=qDdSRaOiyb)
* SocioDojo: Building Lifelong Analytical Agents with Real-world Text and Time Series [\[paper\]](https://openreview.net/forum?id=s9z0HzWJJp)
* Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns [\[paper\]](https://openreview.net/forum?id=CdjnzWsQax)
* Stable Neural Stochastic Differential Equations in Analyzing Irregular Time Series Data [\[paper\]](https://openreview.net/forum?id=4VIgNuQ1pY)
* Soft Contrastive Learning for Time Series [\[paper\]](https://openreview.net/forum?id=pAsQSWlDUf)
* Retrieval-Based Reconstruction For Time-series Contrastive Learning [\[paper\]](https://openreview.net/forum?id=3zQo5oUvia)
* Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach [\[paper\]](https://openreview.net/forum?id=K2c04ulKXn)
* Diffusion-TS: Interpretable Diffusion for General Time Series Generation [\[paper\]](https://openreview.net/forum?id=4h1apFjO99)
* Disentangling Time Series Representations via Contrastive based l-Variational Inference [\[paper\]](https://openreview.net/forum?id=iI7hZSczxE)
* Leveraging Generative Models for Unsupervised Alignment of Neural Time Series Data [\[paper\]](https://openreview.net/forum?id=9zhHVyLY4K)
* Conditional Information Bottleneck Approach for Time Series Imputation [\[paper\]](https://openreview.net/forum?id=K1mcPiDdOJ)
* Generative Modeling of Regular and Irregular Time Series Data via Koopman VAEs [\[paper\]](https://openreview.net/forum?id=eY7sLb0dVF)
* Learning to Embed Time Series Patches Independently [\[paper\]](https://openreview.net/forum?id=WS7GuBDFa2)
* CausalTime: Realistically Generated Time-series for Benchmarking of Causal Discovery [\[paper\]](https://openreview.net/forum?id=iad1yyyGme)
* Parametric Augmentation for Time Series Contrastive Learning [\[paper\]](https://openreview.net/forum?id=EIPLdFy3vp)
* Inherently Interpretable Time Series Classification via Multiple Instance Learning [\[paper\]](https://openreview.net/forum?id=xriGRsoAza)


### KDD 2024

### WWW 2024

#### Time Series Forecasting
* UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2310.09751)
* Unveiling Delay Effects in Traffic Forecasting: A Perspective from Spatial-Temporal Delay Differential Equations [\[paper\]](https://arxiv.org/abs/2402.01231)

#### Time Series Anomaly Detection
* LARA: A Light and Anti-overfitting Retraining Approach for Unsupervised Time Series Anomaly Detection [\[paper\]](https://arxiv.org/abs/2310.05668)
* Revisiting VAE for Unsupervised Time Series Anomaly Detection: A Frequency Perspective [\[paper\]](https://arxiv.org/abs/2402.02820)
* Breaking the Time-Frequency Granularity Discrepancy in Time-Series Anomaly Detection [\[paper\]]()

#### Others
* Dynamic Multi-Network Mining of Tensor Time Series [\[paper\]]()
* E2USD: Efficient-yet-effective Unsupervised State Detection for Multivariate Time Series [\[paper\]](https://arxiv.org/abs/2402.14041)

  
### AAAI 2024
#### Time Series Forecasting
* U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2401.02236)
* HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting [\[paper\]]()
* Considering Nonstationary within Multivariate Time Series with Variational Hierarchical Transformer for Forecasting [\[paper\]]()
* Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2312.08763)
* MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2401.00423)
* Latent Diffusion Transformer for Probabilistic Time Series Forecasting [\[paper\]]()
* Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting [\[paper\]]()

#### Time Series Classification, Clustering, Anomaly Detection
* Graph-Aware Contrasting for Multivariate Time-Series Classification [\[paper\]](https://arxiv.org/abs/2309.05202)
* Diffusion Language-Shapelets for Semisupervised Time-series Classification [\[paper\]]()
*  Energy-efficient Streaming Time Series Classification with Attentive Power Iteration [\[paper\]]()
* Cross-Domain Contrastive Learning for Time Series Clustering [\[paper\]]()
* When Model Meets New Normals: Test-time Adaptation for Unsupervised Time-series Anomaly Detection [\[paper\]](https://arxiv.org/abs/2312.11976)
  
#### Others
* TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning [\[paper\]](https://arxiv.org/abs/2312.15709)
* GraFITi: Graphs for Forecasting Irregularly Sampled Time Series [\[paper\]](https://arxiv.org/abs/2305.12932)
* IVP-VAE: Modeling EHR Time Series with Initial Value Problem Solvers [\[paper\]](https://arxiv.org/abs/2305.06741)
* SimPSI: A Simple Strategy to Preserve Spectral Information in Time Series Data Augmentation [\[paper\]](https://arxiv.org/abs/2312.05790)
* CGS-Mask: Making Time Series Predictions Intuitive for All [\[paper\]](https://arxiv.org/abs/2312.09513)
* CUTS+: High-dimensional Causal Discovery from Irregular Time-series [\[paper\]](https://arxiv.org/abs/2305.05890)
* Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data [\[paper\]](https://arxiv.org/abs/2309.05305)

  
## AI4TS Papers 2023
### NeurIPS 2023
#### Time Series Forecasting
* OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling [\[paper\]](https://neurips.cc/virtual/2023/poster/71725)
* One Fits All: Power General Time Series Analysis by Pretrained LM [\[paper\]](https://neurips.cc/virtual/2023/poster/70856)
* Large Language Models Are Zero Shot Time Series Forecasters [\[paper\]](https://neurips.cc/virtual/2023/poster/70543)
* BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis [\[paper\]](https://neurips.cc/virtual/2023/poster/69976)
* ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling [\[paper\]](https://neurips.cc/virtual/2023/poster/71304)
* FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective [\[paper\]](https://neurips.cc/virtual/2023/poster/71159)
* Frequency-domain MLPs are More Effective Learners in Time Series Forecasting [\[paper\]](https://neurips.cc/virtual/2023/poster/70726)
* Adaptive Normalization for Non-stationary Time Series Forecasting: A Temporal Slice Perspective [\[paper\]](https://neurips.cc/virtual/2023/poster/72816)
* WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting [\[paper\]](https://neurips.cc/virtual/2023/poster/69972)
* Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting [\[paper\]](https://neurips.cc/virtual/2023/poster/70377)
* Conformal PID Control for Time Series Prediction [\[paper\]](https://neurips.cc/virtual/2023/poster/69896)
* SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling [\[paper\]](https://neurips.cc/virtual/2023/poster/70829)
* Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors [\[paper\]](https://neurips.cc/virtual/2023/poster/72562)

#### Time Series Anomaly Detection, Classification
* Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection [\[paper\]](https://neurips.cc/virtual/2023/poster/71195)
* Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction [\[paper\]](https://neurips.cc/virtual/2023/poster/70582)
* MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection [\[paper\]](https://neurips.cc/virtual/2023/poster/71519)
* Time Series as Images: Vision Transformer for Irregularly Sampled Time Series [\[paper\]](https://neurips.cc/virtual/2023/poster/71219)
* Scale-teaching: Robust Multi-scale Training for Time Series Classification with Noisy Labels [\[paper\]](https://neurips.cc/virtual/2023/poster/72608)
  
#### Others
* Causal Discovery from Subsampled Time Series with Proxy Variables [\[paper\]](https://neurips.cc/virtual/2023/poster/70936)
* Causal Discovery in Semi-Stationary Time Series [\[paper\]](https://neurips.cc/virtual/2023/poster/71016)
* Encoding Time-Series Explanations through Self-Supervised Model Behavior Consistency [\[paper\]](https://neurips.cc/virtual/2023/poster/69958)
* Sparse Deep Learning for Time Series Data: Theory and Applications [\[paper\]](https://neurips.cc/virtual/2023/poster/72629)
* CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement [\[paper\]](https://neurips.cc/virtual/2023/poster/70010)
* WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction [\[paper\]](https://neurips.cc/virtual/2023/poster/73593)
* Conformal Prediction for Time Series with Modern Hopfield Networks [\[paper\]](https://neurips.cc/virtual/2023/poster/72007)
* Time Series Kernels based on Nonlinear Vector AutoRegressive Delay Embeddings [\[paper\]](https://neurips.cc/virtual/2023/poster/71521)
* On the Constrained Time-Series Generation Problem [\[paper\]](https://neurips.cc/virtual/2023/poster/72006)
* Contrast Everything: Multi-Granularity Representation Learning for Medical Time-Series [\[paper\]](https://neurips.cc/virtual/2023/poster/70272)
* Finding Order in Chaos: A Novel Data Augmentation Method for Time Series in Contrastive Learning [\[paper\]](https://neurips.cc/virtual/2023/poster/71014)
* FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space [\[paper\]](https://neurips.cc/virtual/2023/poster/70617)
* BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series [\[paper\]](https://neurips.cc/virtual/2023/poster/73499

### ICML 2023 
#### Time Series Forecasting
* Learning Deep Time-index Models for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=pgcfCCNQXO)
* Regions of Reliability in the Evaluation of Multivariate Probabilistic Forecasts [\[paper\]](https://openreview.net/forum?id=gTGFxnBymb) 
* Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=YbYMRZbO1Y) 
* Feature Programming for Multivariate Time Series Prediction [\[paper\]](https://openreview.net/forum?id=LVARH5wXM9) 
* Non-autoregressive Conditional Diffusion Models for Time Series Prediction [\[paper\]](https://openreview.net/forum?id=wZsnZkviro)
  
#### Time Series Anomaly Detection, Classification, Imputation, and XAI
* Prototype-oriented unsupervised anomaly detection for multivariate time series [\[paper\]](https://openreview.net/forum?id=3vO4lS6PuF) 
* Probabilistic Imputation for Time-series Classification with Missing Data [\[paper\]](https://openreview.net/forum?id=7pcZLgulIV) 
* Provably Convergent Schrödinger Bridge with Applications to Probabilistic Time Series Imputation [\[paper\]](https://openreview.net/forum?id=HRmSGZZ1FY) 
* Self-Interpretable Time Series Prediction with Counterfactual Explanations [\[paper\]](https://openreview.net/forum?id=JPMT9kjeJi) 
* Learning Perturbations to Explain Time Series Predictions [\[paper\]](https://openreview.net/forum?id=WpeZu6WzTB)
  
#### Other Time Series Analysis
* Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion [\[paper\]](https://openreview.net/forum?id=OUWckW2g3j) 
* Neural Stochastic Differential Games for Time-series Analysis [\[paper\]]() 
* Sequential Monte Carlo Learning for Time Series Structure Discovery [\[paper\]]() 
* Context Consistency Regularization for Label Sparsity in Time Series [\[paper\]]() 
* Sequential Predictive Conformal Inference for Time Series [\[paper\]]() 
* Improved Online Conformal Prediction via Strongly Adaptive Online Learning [\[paper\]]() 
* Sequential Multi-Dimensional Self-Supervised Learning for Clinical Time Series [\[paper\]]() 
* SOM-CPC: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series [\[paper\]]() 
* Domain Adaptation for Time Series Under Feature and Label Shifts [\[paper\]]() 
* Deep Latent State Space Models for Time-Series Generation [\[paper\]]() 
* Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series [\[paper\]]() 
* Generative Causal Representation Learning for Out-of-Distribution Motion Forecasting [\[paper\]]() 
* Generalized Teacher Forcing for Learning Chaotic Dynamics [\[paper\]]() 
* Learning the Dynamics of Sparsely Observed Interacting Systems [\[paper\]]() 
* Markovian Gaussian Process Variational Autoencoders [\[paper\]](https://openreview.net/forum?id=Z8QlQ207V6) 
* ClimaX: A foundation model for weather and climate [\[paper\]](https://openreview.net/forum?id=TowCaiz7Ui) 


### ICLR 2023
#### Time Series Forecasting
* A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [\[paper\]](https://openreview.net/forum?id=Jbdc0vTOcol) [\[official code\]](https://github.com/yuqinie98/PatchTST)
* Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=vSVLM2j9eie) [\[official code\]]()
* Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=sCrnllCtjoE) [\[official code\]]()
* MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting [\[paper\]](https://openreview.net/forum?id=zt53IDUR1U) [\[official code\]]()
* Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting [\[paper\]](https://openreview.net/forum?id=7C9aRX2nBf2) [\[official code\]]()
* Learning Fast and Slow for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=q-PbpHD3EOk) [\[official code\]]()
* Koopman Neural Operator Forecaster for Time-series with Temporal Distributional Shifts [\[paper\]](https://openreview.net/forum?id=kUmdmHxK5N) [\[official code\]]()
* Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms [\[paper\]](https://openreview.net/forum?id=ctmLBs8lITa) [\[official code\]]()

#### Time Series Anomaly Detection and Classification
* Unsupervised Model Selection for Time Series Anomaly Detection [\[paper\]](https://openreview.net/forum?id=gOZ_pKANaPW) [\[official code\]]()
* Out-of-distribution Representation Learning for Time Series Classification [\[paper\]](https://openreview.net/forum?id=gUZWOE42l6Q) [\[official code\]]()

#### Other Time Series Analysis
* Effectively Modeling Time Series with Simple Discrete State Spaces [\[paper\]](https://openreview.net/forum?id=2EpjkjzdCAa) [\[official code\]]()
* TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis  [\[paper\]](https://openreview.net/forum?id=ju_Uqw384Oq) [\[official code\]](https://github.com/thuml/Time-Series-Library)
* Contrastive Learning for Unsupervised Domain Adaptation of Time Series [\[paper\]](https://openreview.net/forum?id=xPkJYRsQGM) [\[official code\]]()
* Recursive Time Series Data Augmentation [\[paper\]](https://openreview.net/forum?id=5lgD4vU-l24s) [\[official code\]]()
* Multivariate Time-series Imputation with Disentangled Temporal Representations [\[paper\]](https://openreview.net/forum?id=rdjeCNUS6TG) [\[official code\]]()
* Deep Declarative Dynamic Time Warping for End-to-End Learning of Alignment Paths [\[paper\]](https://openreview.net/forum?id=UClBPxIZqnY) [\[official code\]]()
* Rhino: Deep Causal Temporal Relationship Learning with History-dependent Noise [\[paper\]](https://openreview.net/forum?id=i_1rbq8yFWC) [\[official code\]]()
* CUTS: Neural Causal Discovery from Unstructured Time-Series Data [\[paper\]](https://openreview.net/forum?id=UG8bQcD3Emv) [\[official code\]]()
* Temporal Dependencies in Feature Importance for Time Series Prediction [\[paper\]](https://openreview.net/forum?id=C0q9oBc3n4) [\[official code\]]()

### KDD 2023
#### Time Series Anomaly Detection
* DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection [\[paper\]](https://arxiv.org/abs/2306.10347) [\[official code\]](https://github.com/DAMO-DI-ML/KDD2023-DCdetector)
* Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models [\[paper\]](https://github.com/ChunjingXiao/DiffAD/blob/main/KDD_23_DiffAD.pdf) [\[official code\]](https://github.com/ChunjingXiao/DiffAD)
* Precursor-of-Anomaly Detection for Irregular Time Series [\[paper\]](https://arxiv.org/abs/2306.15489)  
#### Time Series Forecasting
* When Rigidity Hurts: Soft Consistency Regularization for Probabilistic Hierarchical Time Series Forecasting
* TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2306.09364)
* Hierarchical Proxy Modeling for Improved HPO in Time Series Forecasting
* Sparse Binary Transformers for Multivariate Time Series Modeling [\[paper\]]() [\[official code\]]()
* Interactive Generalized Additive Model and Its Applications in Electric Load Forecasting
#### Time Series Forecasting (Traffic)
* Frigate: Frugal Spatio-temporal Forecasting on Road Networks [\[paper\]]() [\[official code\]]()
* Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities
* Robust Spatiotemporal Traffic Forecasting with Reinforced Dynamic Adversarial Training
* Pattern Expansion and Consolidation on Evolving Graphs for Continual Traffic Prediction
#### Time Series Imputation
* Source-Free Domain Adaptation with Temporal Imputation for Time Series Data [\[paper\]]() [\[official code\]]()
* Networked Time Series Imputation via Position-aware Graph Enhanced Variational Autoencoders
* An Observed Value Consistent Diffusion Model for Imputing Missing Values in Multivariate Time Series
#### Others
* Online Few-Shot Time Series Classification for Aftershock Detection [\[paper\]]() [\[official code\]]()
* Self-supervised Classification of Clinical Multivariate Time Series using Time Series Dynamics
* Warpformer: A Multi-scale Modeling Approach for Irregular Clinical Time Series
* Parameter-free Spikelet: Discovering Different Length and Warped Time Series Motifs using an Adaptive Time Series Representation
* FLAMES2Graph: An Interpretable Federated Multivariate Time Series Classification Framework
* WHEN: A Wavelet-DTW Hybrid Attention Network for Heterogeneous Time Series Analysis

### AAAI 2023
#### Time Series Forecasting
* AirFormer: Predicting Nationwide Air Quality in China with Transformers [\[paper\]](https://arxiv.org/abs/2211.15979) [\[official code\]](https://github.com/yoshall/AirFormer)
* Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting [\[paper\]]() [\[official code\]]()
* WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting of Multivariate Time Series [\[paper\]]() [\[official code\]]() 
* Are Transformers Effective for Time Series Forecasting [\[paper\]]() [\[official code\]]()
* Forecasting with Sparse but Informative Variables: A Case Study in Predicting Blood Glucose [\[paper\]]() [\[official code\]]()
* An Extreme-Adaptive Time Series Prediction Model Based on Probability-Enhanced LSTM Neural Networks [\[paper\]](https://arxiv.org/abs/2211.15891) [\[official code\]]()
* Spatio-Temporal Meta-Graph Learning for Traffic Forecasting [\[paper\]]() [\[official code\]]()

#### Other Time Series Analysis
* Temporal-Frequency Co-Training for Time Series Semi-Supervised Learning [\[paper\]]() [\[official code\]]()
* SEnsor Alignment for Multivariate Time-Series Unsupervised Domain Adaptation [\[paper\]]() [\[official code\]]()
* Causal Recurrent Variational Autoencoder for Medical Time Series Generation [\[paper\]]() [\[official code\]]()
* AEC-GAN: Adversarial Error Correction GANs for Auto-Regressive Long Time-series Generation [\[paper\]]() [\[official code\]]()
* SVP-T: A Shape-Level Variable-Position Transformer for Multivariate Time Series Classification [\[paper\]]() [\[official code\]]()


## AI4TS Papers 2022
### NeurIPS 2022
#### Time Series Forecasting
* FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2205.08897) [\[official code\]](https://github.com/DAMO-DI-ML/NeurIPS2022-FiLM)
* SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction [\[paper\]](https://arxiv.org/abs/2106.09305) [\[official code\]](https://github.com/cure-lab/SCINet)

* Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2205.14415)
* Earthformer: Exploring Space-Time Transformers for Earth System Forecasting [\[paper\]](https://arxiv.org/abs/2207.05833)
* Generative Time Series Forecasting with Diffusion, Denoise and Disentanglement
* Learning Latent Seasonal-Trend Representations for Time Series Forecasting
* WaveBound: Dynamically Bounding Error for Stable Time Series Forecasting
 
* Time Dimension Dances with Simplicial Complexes: Zigzag Filtration Curve based Supra-Hodge Convolution Networks for Time-series Forecasting
 
* Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks
 
* C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting
 
* Meta-Learning Dynamics Forecasting Using Task Inference [\[paper\]](https://arxiv.org/abs/2102.10271)
 
* Conformal Prediction with Temporal Quantile Adjustments
 



#### Other Time Series Analysis
* Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency Consistency, [\[paper\]](https://arxiv.org/abs/2206.08496) [\[official code\]](https://github.com/mims-harvard/TFC-pretraining)
* Causal Disentanglement for Time Series
* BILCO: An Efficient Algorithm for Joint Alignment of Time Series
* Dynamic Sparse Network for Time Series Classification: Learning What to “See”
* AutoST: Towards the Universal Modeling of Spatio-temporal Sequences
 
* GT-GAN: General Purpose Time Series Synthesis with Generative Adversarial Networks
 
* Efficient learning of nonlinear prediction models with time-series privileged information
 
* Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models
 




### ICML 2022
#### Time Series Forecasting
* FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting [\[paper\]](https://arxiv.org/abs/2201.12740) [\[official code\]](https://github.com/DAMO-DI-ML/ICML2022-FEDformer)
* TACTiS: Transformer-Attentional Copulas for Time Series [\[paper\]](https://arxiv.org/abs/2202.03528) [\[official code\]](https://github.com/ServiceNow/tactis)
* Volatility Based Kernels and Moving Average Means for Accurate Forecasting with Gaussian Processes [\[paper\]](https://arxiv.org/abs/2207.06544) [\[official code\]](https://github.com/g-benton/volt)
* Domain Adaptation for Time Series Forecasting via Attention Sharing [\[paper\]](https://arxiv.org/abs/2102.06828) 
* DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting [\[paper\]](https://proceedings.mlr.press/v162/lan22a.html) [\[official code\]](https://github.com/SYLan2019/DSTAGNN)

#### Time Series Anomaly Detection
* Deep Variational Graph Convolutional Recurrent Network for Multivariate Time Series Anomaly Detection [\[paper\]](https://proceedings.mlr.press/v162/chen22x.html)

#### Other Time Series Analysis
* Adaptive Conformal Predictions for Time Series [\[paper\]](https://arxiv.org/abs/2202.07282) [\[official code\]](https://github.com/mzaffran/adaptiveconformalpredictionstimeseries)
* Modeling Irregular Time Series with Continuous Recurrent Units [\[paper\]](https://arxiv.org/abs/2111.11344) [\[official code\]](https://github.com/boschresearch/continuous-recurrent-units)
* Unsupervised Time-Series Representation Learning with Iterative Bilinear Temporal-Spectral Fusion [\[paper\]](https://arxiv.org/abs/2202.04770) 
* Reconstructing nonlinear dynamical systems from multi-modal time series [\[paper\]](https://arxiv.org/abs/2111.02922) [\[official code\]](https://github.com/durstewitzlab/mmplrnn)
* Utilizing Expert Features for Contrastive Learning of Time-Series Representations [\[paper\]](https://arxiv.org/abs/2206.11517) [\[official code\]](https://github.com/boschresearch/expclr)
* Learning of Cluster-based Feature Importance for Electronic Health Record Time-series [\[paper\]](https://proceedings.mlr.press/v162/aguiar22a.html)

### ICLR 2022
#### Time Series Forecasting
* Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting [\[paper\]](https://openreview.net/forum?id=0EXmFzUn5I) [\[official code\]](https://github.com/alipay/Pyraformer)
* DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=AJAR-JgNw__) [\[official code\]](https://github.com/weifantt/depts)
* CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting [\[paper\]](https://openreview.net/forum?id=PilZY3omXV2) [\[official code\]](https://github.com/salesforce/CoST)
* Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift [\[paper\]](https://openreview.net/forum?id=cGDAkQo1C0p) [\[official code\]](https://github.com/ts-kim/RevIN)
* TAMP-S2GCNets: Coupling Time-Aware Multipersistence Knowledge Representation with Spatio-Supra Graph Convolutional Networks for Time-Series Forecasting [\[paper\]](https://openreview.net/forum?id=wv6g8fWLX2q) [\[official code\]](https://www.dropbox.com/sh/n0ajd5l0tdeyb80/AABGn-ejfV1YtRwjf_L0AOsNa?dl=0)
* Back2Future: Leveraging Backfill Dynamics for Improving Real-time Predictions in Future [\[paper\]](https://openreview.net/forum?id=L01Nn_VJ9i) [\[official code\]](https://github.com/AdityaLab/Back2Future)
* On the benefits of maximum likelihood estimation for Regression and Forecasting [\[paper\]](https://openreview.net/forum?id=zrW-LVXj2k1)
* Learning to Remember Patterns: Pattern Matching Memory Networks for Traffic Forecasting [\[paper\]](https://openreview.net/forum?id=wwDg3bbYBIq) [\[official code\]](https://github.com/hyunwookl/pm-memnet)


#### Time Series Anomaly Detection
* Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy [\[paper\]](https://openreview.net/forum?id=LzQQ89U1qm_) [\[official code\]](https://github.com/thuml/Anomaly-Transformer)
* Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series [\[paper\]](https://openreview.net/forum?id=45L_dgP48Vd) [\[official code\]](https://github.com/enyandai/ganf)

#### Time Series Classification
* T-WaveNet: A Tree-Structured Wavelet Neural Network for Time Series Signal Analysis [\[paper\]](https://openreview.net/forum?id=U4uFaLyg7PV)
* Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification [\[paper\]](https://openreview.net/forum?id=PDYs7Z2XFGv)

#### Other Time Series Analysis
* Graph-Guided Network for Irregularly Sampled Multivariate Time Series [\[paper\]](https://openreview.net/forum?id=Kwm8I7dU-l5)
* Heteroscedastic Temporal Variational Autoencoder For Irregularly Sampled Time Series [\[paper\]](https://openreview.net/forum?id=Az7opqbQE-3)
* Transformer Embeddings of Irregularly Spaced Events and Their Participants [\[paper\]](https://openreview.net/forum?id=Rty5g9imm7H)
* Filling the G_ap_s: Multivariate Time Series Imputation by Graph Neural Networks [\[paper\]](https://openreview.net/forum?id=kOu3-S3wJ7)
* PSA-GAN: Progressive Self Attention GANs for Synthetic Time Series [\[paper\]](https://openreview.net/forum?id=Ix_mh42xq5w)
* Huber Additive Models for Non-stationary Time Series Analysis [\[paper\]](https://openreview.net/forum?id=9kpuB2bgnim)
* LORD: Lower-Dimensional Embedding of Log-Signature in Neural Rough Differential Equations [\[paper\]](https://openreview.net/forum?id=fCG75wd39ze)
* Imbedding Deep Neural Networks [\[paper\]](https://openreview.net/forum?id=yKIAXjkJc2F)
* Coherence-based Label Propagation over Time Series for Accelerated Active Learning [\[paper\]](https://openreview.net/forum?id=gjNcH0hj0LM)
* Long Expressive Memory for Sequence Modeling [\[paper\]](https://openreview.net/forum?id=vwj6aUeocyf)
* Autoregressive Quantile Flows for Predictive Uncertainty Estimation [\[paper\]](https://openreview.net/forum?id=z1-I6rOKv1S)
* Learning the Dynamics of Physical Systems from Sparse Observations with Finite Element Networks [\[paper\]](https://openreview.net/forum?id=HFmAukZ-k-2)
* Temporal Alignment Prediction for Supervised Representation Learning and Few-Shot Sequence Classification [\[paper\]](https://openreview.net/forum?id=p3DKPQ7uaAi)
* Explaining Point Processes by Learning Interpretable Temporal Logic Rules [\[paper\]](https://openreview.net/forum?id=P07dq7iSAGr)


### KDD 2022
 
#### Time Series Forecasting
* Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting [\[code\]](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)
* Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting
* Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting
* Multi-Variate Time Series Forecasting on Variable Subset
* Greykite: Deploying Flexible Forecasting at Scale in LinkedIn

#### Time Series Anomaly Detection
* Local Evaluation of Time Series Anomaly Detection Algorithms
* Scaling Time Series Anomaly Detection to Trillions of Datapoints and Ultra-fast Arriving Data Streams

#### Other Time-Series/Spatio-Temporal Analysis
* Task-Aware Reconstruction for Time-Series Transformer
* Towards Learning Disentangled Representations for Time Series
* ProActive: Self-Attentive Temporal Point Process Flows for Activity Sequences
* Non-stationary Time-aware Kernelized Attention for Temporal Event Prediction
* MSDR: Multi-Step Dependency Relation Networks for Spatial Temporal Forecasting
* Graph2Route: A Dynamic Spatial-Temporal Graph Neural Network for Pick-up and Delivery Route Prediction
* Beyond Point Prediction: Capturing Zero-Inflated & Heavy-Tailed Spatiotemporal Data with Deep Extreme Mixture Models
* Robust Event Forecasting with Spatiotemporal Confounder Learning
* Mining Spatio-Temporal Relations via Self-Paced Graph Contrastive Learning
* Spatio-Temporal Graph Few-Shot Learning with Cross-City Knowledge Transfer
* Characterizing Covid waves via spatio-temporal decomposition


### AAAI 2022
#### Time Series Forecasting
* CATN: Cross Attentive Tree-Aware Network for Multivariate Time Series Forecasting [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai7403) 
* Reinforcement Learning based Dynamic Model Combination for Time Series Forecasting [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai8424)
* DDG-DA: Data Distribution Generation for Predictable Concept Drift Adaptation [\[paper\]](https://arxiv.org/abs/2201.04038) [official code\]](https://github.com/microsoft/qlib/tree/main/examples/benchmarks_dynamic/DDG-DA)
* PrEF: Probabilistic Electricity Forecasting via Copula-Augmented State Space Model [\[paper\]](https://aaai-2022.virtualchair.net/poster_aisi7128)
* LIMREF: Local Interpretable Model Agnostic Rule-Based Explanations for Forecasting, with an Application to
Electricity Smart Meter Data [\[paper\]](https://aaai-2022.virtualchair.net/poster_aisi8802)  
* Learning and Dynamical Models for Sub-Seasonal Climate Forecasting: Comparison and Collaboration [\[paper\]](https://arxiv.org/abs/2110.05196) [\[official code\]](https://github.com/Sijie-umn/SSF-MIP)
* CausalGNN: Causal-Based Graph Neural Networks for Spatio-Temporal Epidemic Forecasting [\[paper\]](https://aaai-2022.virtualchair.net/poster_aisi6475)
* Conditional Local Convolution for Spatio-Temporal Meteorological Forecasting [\[paper\]](https://arxiv.org/abs/2101.01000) [\[official code\]](https://github.com/bird-tao/clcrn)
* Graph Neural Controlled Differential Equations for Traffic Forecasting [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai6502) [\[official code\]](https://github.com/jeongwhanchoi/STG-NCDE)
* STDEN: Towards Physics-Guided Neural Networks for Traffic Flow Prediction [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai211) [\[official code\]](https://github.com/Echo-Ji/STDEN)

#### Time Series Anomaly Detection
* Towards a Rigorous Evaluation of Time-Series Anomaly Detection [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai2239)  
* AnomalyKiTS-Anomaly Detection Toolkit for Time Series [\[Demo paper\]](https://aaai-2022.virtualchair.net/poster_dm318) 

#### Other Time Series Analysis
* TS2Vec: Towards Universal Representation of Time Series [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai8809) [\[official code\]](https://github.com/yuezhihan/ts2vec)
* I-SEA: Importance Sampling and Expected Alignment-Based Deep Distance Metric Learning for Time Series Analysis and Embedding [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai10930)  
* Training Robust Deep Models for Time-Series Domain: Novel Algorithms and Theoretical Analysis [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai4151)  
* Conditional Loss and Deep Euler Scheme for Time Series Generation [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai12878)  
* Clustering Interval-Censored Time-Series for Disease Phenotyping [\[paper\]](https://aaai-2022.virtualchair.net/poster_aaai12517)  


### IJCAI 2022
#### Time Series Forecasting
* Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2204.13767)
* Coherent Probabilistic Aggregate Queries on Long-horizon Forecasts [\[paper\]](https://arxiv.org/abs/2111.03394) [\[official code\]](https://github.com/pratham16cse/aggforecaster)
* Regularized Graph Structure Learning with Semantic Knowledge for Multi-variates Time-Series Forecasting
* DeepExtrema: A Deep Learning Approach for Forecasting Block Maxima in Time Series Data [\[paper\]](https://arxiv.org/abs/2205.02441) [\[official code\]](https://github.com/galib19/deepextrema-ijcai22-)
* Memory Augmented State Space Model for Time Series Forecasting  
* Physics-Informed Long-Sequence Forecasting From Multi-Resolution Spatiotemporal Data 
* Long-term Spatio-Temporal Forecasting via Dynamic Multiple-Graph Attention [\[paper\]](https://arxiv.org/abs/2204.11008) [\[official code\]](https://arxiv.org/abs/2204.11008)
* FOGS: First-Order Gradient Supervision with Learning-based Graph for Traffic Flow Forecasting

#### Time Series Anomaly Detection
* Neural Contextual Anomaly Detection for Time Series [\[paper\]](https://arxiv.org/abs/2107.07702)  
* GRELEN: Multivariate Time Series Anomaly Detection from the Perspective of Graph Relational Learning  

#### Time Series Classification
* A Reinforcement Learning-Informed Pattern Mining Framework for Multivariate Time Series Classification [\[paper\]](https://cpsl.pratt.duke.edu/sites/cpsl.pratt.duke.edu/files/docs/gao_ijcai22.pdf)
* T-SMOTE: Temporal-oriented Synthetic Minority Oversampling Technique for Imbalanced Time Series Classification


### SIGMOD VLDB ICDE 2022
#### Time Series Forecasting
* METRO: A Generic Graph Neural Network Framework for Multivariate Time Series Forecasting, VLDB'22. [\[paper\]](http://vldb.org/pvldb/vol15/p224-cui.pdf) [\[official code\]](https://zheng-kai.com/code/metro_single_s.zip) 
* AutoCTS: Automated Correlated Time Series Forecasting, VLDB'22. [\[paper\]](http://vldb.org/pvldb/vol15/p971-wu.pdf)
* Towards Spatio-Temporal Aware Traffic Time Series Forecasting, ICDE'22. [\[paper\]](https://arxiv.org/abs/2203.15737) [\[official code\]](https://github.com/razvanc92/st-wa) 


#### Time Series Anomaly Detection
* Sintel: A Machine Learning Framework to Extract Insights from Signals, SIGMOD'22. [\[paper\]](https://arxiv.org/abs/2204.09108) [\[official code\]](https://github.com/sarahmish/sintel-paper) 
* TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection, VLDB'22. [\[paper\]](https://helios2.mi.parisdescartes.fr/~themisp/publications/pvldb22-tsbuad.pdf) [\[official code\]](https://github.com/johnpaparrizos/TSB-UAD)
* TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data, VLDB'22. [\[paper\]](https://arxiv.org/abs/2201.07284) [\[official code\]](https://github.com/imperial-qore/tranad)
* Unsupervised Time Series Outlier Detection with Diversity-Driven Convolutional Ensembles, VLDB'22. [\[paper\]](http://vldb.org/pvldb/vol15/p611-chaves.pdf)
* Robust and Explainable Autoencoders for Time Series Outlier Detection, ICDE'22. [\[paper\]](https://arxiv.org/abs/2204.03341)
* Anomaly Detection in Time Series with Robust Variational Quasi-Recurrent Autoencoders, ICDE'22.  

#### Time Series Classification
* IPS: Instance Profile for Shapelet Discovery for Time Series Classification, ICDE'22. [\[paper\]](https://personal.ntu.edu.sg/assourav/papers/ICDE-22-IPS.pdf)
* Towards Backdoor Attack on Deep Learning based Time Series Classification, ICDE'22. [\[paper\]]()

#### Other Time Series Analysis
* OnlineSTL: Scaling Time Series Decomposition by 100x, VLDB'22. [\[paper\]](http://vldb.org/pvldb/vol15/p1417-mishra.pdf) 
* Efficient temporal pattern mining in big time series using mutual information, VLDB'22. [\[paper\]](https://arxiv.org/abs/2010.03653)
* Learning Evolvable Time-series Shapelets, ICDE'22.  


<!--  [\[paper\]]() [\[official code\]]()  -->  
### Misc 2022
#### Time Series Forecasting
* CAMul: Calibrated and Accurate Multi-view Time-Series Forecasting, WWW'22. [\[paper\]](https://arxiv.org/abs/2109.07438) [\[official code\]](https://github.com/adityalab/camul)
* Multi-Granularity Residual Learning with Confidence Estimation for Time Series Prediction, WWW'22. [\[paper\]](https://web.archive.org/web/20220426115606id_/https://dl.acm.org/doi/pdf/10.1145/3485447.3512056)  
* RETE: Retrieval-Enhanced Temporal Event Forecasting on Unified Query Product Evolutionary Graph, WWW'22. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3485447.3511974) 
* Robust Probabilistic Time Series Forecasting, AISTATS'22. [\[paper\]](https://arxiv.org/abs/2202.11910) [\[official code\]](https://github.com/tetrzim/robust-probabilistic-forecasting) 
* Learning Quantile Functions without Quantile Crossing for Distribution-free Time Series Forecasting, AISTATS'22. [\[paper\]](https://arxiv.org/abs/2111.06581)


#### Time Series Anomaly Detection
* TFAD: A Decomposition Time Series Anomaly Detection Architecture with Time-Frequency Analysis, CIKM'22. [\[paper\]](https://arxiv.org/abs/2210.09693) [\[official code\]](https://github.com/DAMO-DI-ML/CIKM22-TFAD)
* Deep Generative model with Hierarchical Latent Factors for Time Series Anomaly Detection, AISTATS'22. [\[paper\]](https://arxiv.org/abs/2202.07586) [\[official code\]](https://github.com/cchallu/dghl)
* A Semi-Supervised VAE Based Active Anomaly Detection Framework in Multivariate Time Series for Online Systems, WWW'22. [\[paper\]](https://dl.acm.org/doi/10.1145/3485447.3511984) 


#### Other Time Series Analysis
* Decoupling Local and Global Representations of Time Series, AISTATS'22. [\[paper\]](https://arxiv.org/abs/2202.02262) [\[official code\]](https://github.com/googleinterns/local_global_ts_representation)
* LIMESegment: Meaningful, Realistic Time Series Explanations, AISTATS'22. [\[paper\]](https://proceedings.mlr.press/v151/sivill22a.html)
* Using time-series privileged information for provably efficient learning of prediction models, AISTATS'22. [\[paper\]](https://arxiv.org/abs/2110.14993) [\[official code\]](https://github.com/RickardKarl/LearningUsingPrivilegedTimeSeries)
* Amortised Likelihood-free Inference for Expensive Time-series Simulators with Signatured Ratio Estimation, AISTATS'22. [\[paper\]]() [\[official code\]](https://arxiv.org/abs/2202.11585)
* EXIT: Extrapolation and Interpolation-based Neural Controlled Differential Equations for Time-series Classification and Forecasting, WWW'22. [\[paper\]](https://arxiv.org/abs/2204.08771) 





## AI4TS Papers 2021 

### NeurIPS 2021
#### Time Series Forecasting
* Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting [\[paper\]](https://arxiv.org/abs/2106.13008) [\[official code\]](https://github.com/thuml/autoformer)
* MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data [\[paper\]](https://arxiv.org/abs/2110.14354) 
* Conformal Time-Series Forecasting [\[paper\]](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html) [\[official code\]](https://github.com/kamilest/conformal-rnn)
* Probabilistic Forecasting: A Level-Set Approach [\[paper\]](https://proceedings.neurips.cc/paper/2021/hash/32b127307a606effdcc8e51f60a45922-Abstract.html) 
* Topological Attention for Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2107.09031) 
* When in Doubt: Neural Non-Parametric Uncertainty Quantification for Epidemic Forecasting [\[paper\]](https://arxiv.org/abs/2106.03904) [\[official code\]](https://github.com/AdityaLab/EpiFNP)
* Monash Time Series Forecasting Archive [\[paper\]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/eddea82ad2755b24c4e168c5fc2ebd40-Abstract-round2.html) [\[official code\]](https://forecastingdata.org/)  

#### Time Series Anomaly Detection
* Revisiting Time Series Outlier Detection: Definitions and Benchmarks [\[paper\]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/ec5decca5ed3d6b8079e2e7e7bacc9f2-Abstract-round1.html) [\[official code\]](https://github.com/datamllab/tods/tree/benchmark)   
* Online false discovery rate control for anomaly detection in time series [\[paper\]](https://arxiv.org/abs/2112.03196)  
* Detecting Anomalous Event Sequences with Temporal Point Processes [\[paper\]](https://arxiv.org/abs/2106.04465) 

#### Other Time Series Analysis
* Probabilistic Transformer For Time Series Analysis [\[paper\]](https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html) 
* Shifted Chunk Transformer for Spatio-Temporal Representational Learning [\[paper\]](https://arxiv.org/abs/2108.11575) 
* Deep Explicit Duration Switching Models for Time Series [\[paper\]](https://openreview.net/forum?id=LaM6G4yrMy0) [\[official code\]](https://github.com/abdulfatir/REDSDS)
* Time-series Generation by Contrastive Imitation [\[paper\]](https://openreview.net/forum?id=RHZs3GqLBwg)  
* CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation [\[paper\]](https://arxiv.org/abs/2107.03502) [\[official code\]](https://github.com/ermongroup/csdi)
* Adjusting for Autocorrelated Errors in Neural Networks for Time Series [\[paper\]](https://arxiv.org/abs/2101.12578) [\[official code\]](https://github.com/Daikon-Sun/AdjustAutocorrelation)
* SSMF: Shifting Seasonal Matrix Factorization [\[paper\]](https://arxiv.org/abs/2110.12763) [\[official code\]](https://github.com/kokikwbt/ssmf)
* Coresets for Time Series Clustering [\[paper\]](https://arxiv.org/abs/2110.15263)  
* Neural Flows: Efficient Alternative to Neural ODEs [\[paper\]](https://arxiv.org/abs/2110.13040) [\[official code\]](https://github.com/mbilos/neural-flows-experiments)
* Spatio-Temporal Variational Gaussian Processes [\[paper\]](https://arxiv.org/pdf/2111.01732.pdf) [\[official code\]](https://github.com/aaltoml/spatio-temporal-gps)
* Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers [\[paper\]](https://openreview.net/forum?id=A_Aeb-XLozL) [\[official code\]](https://github.com/SamsungLabs/Drop-DTW) 



### ICML 2021
#### Time Series Forecasting
* Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2101.12072) [\[official code\]](https://github.com/zalandoresearch/pytorch-ts)
* End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series [\[paper\]](https://proceedings.mlr.press/v139/rangapuram21a.html) [\[official code\]](https://github.com/rshyamsundar/gluonts-hierarchical-ICML-2021)
* RNN with particle flow for probabilistic spatio-temporal forecasting [\[paper\]](https://arxiv.org/abs/2106.06064) [\[official code\]](https://github.com/networkslab/rnn_flow)
* Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2105.04100) [\[official code\]](https://github.com/Z-GCNETs/Z-GCNETs)
* Variance Reduction in Training Forecasting Models with Subgroup Sampling [\[paper\]](https://arxiv.org/abs/2103.02062)  
* Explaining Time Series Predictions With Dynamic Masks [\[paper\]](https://arxiv.org/abs/2106.05303) [\[official code\]](https://github.com/JonathanCrabbe/Dynamask)
* Conformal prediction interval for dynamic time-series [\[paper\]](https://arxiv.org/abs/2010.09107) [\[official code\]](https://github.com/hamrel-cxu/EnbPI)

#### Time Series Anomaly Detection
* Neural Transformation Learning for Deep Anomaly Detection Beyond Images [\[paper\]](https://arxiv.org/abs/2103.16440) [\[official code\]](https://github.com/boschresearch/NeuTraL-AD)
* Event Outlier Detection in Continuous Time [\[paper\]](https://arxiv.org/abs/1912.09522) [\[official code\]](https://github.com/siqil/CPPOD)

#### Other Time Series Analysis
* Voice2Series: Reprogramming Acoustic Models for Time Series Classification [\[paper\]](https://arxiv.org/abs/2106.09296) [\[official code\]](https://github.com/huckiyang/Voice2Series-Reprogramming)
* Neural Rough Differential Equations for Long Time Series [\[paper\]](https://arxiv.org/abs/2009.08295) [\[official code\]](https://github.com/jambo6/neuralRDEs)
* Neural Spatio-Temporal Point Processes [\[paper\]](https://arxiv.org/abs/2011.04583) [\[official code\]](https://github.com/facebookresearch/neural_stpp)
* Learning Neural Event Functions for Ordinary Differential Equations [\[paper\]](https://arxiv.org/abs/2011.03902) [\[official code\]](https://github.com/rtqichen/torchdiffeq)
* Approximation Theory of Convolutional Architectures for Time Series Modelling [\[paper\]](https://arxiv.org/abs/2107.09355) 
* Whittle Networks: A Deep Likelihood Model for Time Series [\[paper\]](https://proceedings.mlr.press/v139/yu21c.html) [\[official code\]](https://github.com/ml-research/WhittleNetworks)
* Necessary and sufficient conditions for causal feature selection in time series with latent common causes [\[paper\]](http://proceedings.mlr.press/v139/mastakouri21a.html)  


### ICLR 2021
#### Time Series Forecasting
* Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows [\[paper\]](https://openreview.net/forum?id=WiGQBFuVRv) [\[official code\]](https://github.com/zalandoresearch/pytorch-ts) 
* Discrete Graph Structure Learning for Forecasting Multiple Time Series [\[paper\]](https://openreview.net/forum?id=WEHSlH5mOk) [\[official code\]](https://github.com/chaoshangcs/GTS)

#### Other Time Series Analysis
* Clairvoyance: A Pipeline Toolkit for Medical Time Series [\[paper\]](https://openreview.net/forum?id=xnC8YwKUE3k) [\[official code\]](https://github.com/vanderschaarlab/clairvoyance)
* Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding [\[paper\]](https://openreview.net/forum?id=8qDwejCuCN) [\[official code\]](https://github.com/sanatonek/TNC_representation_learning)
* Multi-Time Attention Networks for Irregularly Sampled Time Series [\[paper\]](https://openreview.net/forum?id=4c0J6lwQ4_) [\[official code\]](https://github.com/reml-lab/mTAN)
* Generative Time-series Modeling with Fourier Flows [\[paper\]](https://openreview.net/forum?id=PpshD0AXfA) [\[official code\]](https://github.com/ahmedmalaa/Fourier-flows)
* Differentiable Segmentation of Sequences [\[paper\]](https://openreview.net/forum?id=4T489T4yav) [\[slides\]](https://iclr.cc/media/Slides/iclr/2021/virtual(05-08-00)-05-08-00UTC-2993-differentiable_.pdf)  [\[official code\]](https://github.com/diozaka/diffseg) 
* Neural ODE Processes [\[paper\]](https://openreview.net/forum?id=27acGyyI1BY) [\[official code\]](https://github.com/crisbodnar/ndp) 
* Learning Continuous-Time Dynamics by Stochastic Differential Networks [\[paper\]](https://openreview.net/forum?id=U850oxFSKmN) [\[official code\]]() 

 
### KDD 2021
#### Time Series Forecasting
* ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting [\[paper\]](https://dl.acm.org/doi/10.1145/3447548.3467330) [\[official code\]](https://github.com/JLDeng/ST-Norm)
* Graph Deep Factors for Forecasting with Applications to Cloud Resource Allocation [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3447548.3467357)  
* Quantifying Uncertainty in Deep Spatiotemporal Forecasting [\[paper\]](https://arxiv.org/abs/2105.11982) 
* Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting [\[paper\]](https://arxiv.org/abs/2106.12931) [\[official code\]](https://github.com/square-coder/STGODE)
* TrajNet: A Trajectory-Based Deep Learning Model for Traffic Prediction [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3447548.3467236)  
* Dynamic and Multi-faceted Spatio-temporal Deep Learning for Traffic Speed Forecasting [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3447548.3467275) 


#### Time Series Anomaly Detection
* Multivariate Time Series Anomaly Detection and Interpretation using Hierarchical Inter-Metric and Temporal Embedding [\[paper\]](https://netman.aiops.org/wp-content/uploads/2021/08/KDD21_InterFusion_Li.pdf) [\[official code\]](https://github.com/zhhlee/InterFusion)
* Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization [\[paper\]](https://dl.acm.org/doi/10.1145/3447548.3467174) [\[official code\]](https://github.com/eBay/RANSynCoders)
* Time Series Anomaly Detection for Cyber-physical Systems via Neural System Identification and Bayesian Filtering [\[paper\]](https://arxiv.org/abs/2106.07992) [\[official code\]](https://arxiv.org/abs/2106.07992)
* Multi-Scale One-Class Recurrent Neural Networks for Discrete Event Sequence Anomaly Detection [\[paper\]](https://arxiv.org/abs/2008.13361) [\[official code\]](https://github.com/wzwtrevor/Multi-Scale-One-Class-Recurrent-Neural-Networks)

#### Other Time Series Analysis
* Representation Learning of Multivariate Time Series using a Transformer Framework [\[paper\]](https://arxiv.org/abs/2010.02803) [\[official code\]](https://github.com/gzerveas/mvts_transformer)
* Causal and Interpretable Rules for Time Series Analysis [\[paper\]](https://josselin-garnier.org/wp-content/uploads/2021/10/kdd21.pdf)  
* MiniRocket: A Fast (Almost) Deterministic Transform for Time Series Classification [\[paper\]](https://arxiv.org/abs/2012.08791) [\[official code\]](https://github.com/angus924/minirocket)
* Statistical models coupling allows for complex localmultivariate time series analysis [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3447548.3467362)
* Fast and Accurate Partial Fourier Transform for Time Series Data [\[paper\]](https://jungijang.github.io/resources/2021/KDD/pft.pdf) [\[official code\]](https://github.com/snudatalab/PFT)
* Deep Learning Embeddings for Data Series Similarity Search [\[paper\]](https://qtwang.github.io/assets/pdf/kdd21-seanet.pdf) [\[link\]](https://qtwang.github.io/kdd21-seanet)


### AAAI 2021
#### Time Series Forecasting 
* Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [\[paper\]](https://arxiv.org/abs/2012.07436) [\[official code\]](https://github.com/zhouhaoyi/Informer2020) 
* Deep Switching Auto-Regressive Factorization: Application to Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2009.05135) [\[official code\]](https://github.com/ostadabbas/DSARF)
* Dynamic Gaussian Mixture Based Deep Generative Model for Robust Forecasting on Sparse Multivariate Time Series [\[paper\]](https://arxiv.org/abs/2103.02164) [\[official code\]](https://github.com/thuwuyinjun/DGM2)
* Temporal Latent Autoencoder: A Method for Probabilistic Multivariate Time Series Forecasting [\[paper\]](https://arxiv.org/abs/2101.10460)  
* Synergetic Learning of Heterogeneous Temporal Sequences for Multi-Horizon Probabilistic Forecasting [\[paper\]](https://arxiv.org/abs/2102.00431)  
* Meta-Learning Framework with Applications to Zero-Shot Time-Series Forecasting [\[paper\]](https://arxiv.org/abs/2002.02887) 
* Attentive Neural Point Processes for Event Forecasting [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16929) [\[official code\]](https://github.com/guyulongcs/AAAI2021_ANPP) 
* Forecasting Reservoir Inflow via Recurrent Neural ODEs [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17763)  
* Hierarchical Graph Convolution Network for Traffic Forecasting [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16088) 
* Traffic Flow Forecasting with Spatial-Temporal Graph Diffusion Network [\[paper\]](https://arxiv.org/abs/2110.04038) [\[official code\]](https://github.com/jillbetty001/ST-GDN) 
* Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting [\[paper\]](https://arxiv.org/abs/2012.09641) [\[official code\]](https://github.com/MengzhangLI/STFGNN) 
* FC-GAGA: Fully Connected Gated Graph Architecture for Spatio-Temporal Traffic Forecasting [\[paper\]](https://arxiv.org/abs/2007.15531) [\[official code\]](https://github.com/boreshkinai/fc-gaga) 
* Fairness in Forecasting and Learning Linear Dynamical Systems [\[paper\]](https://arxiv.org/abs/2006.07315) 
* A Multi-Step-Ahead Markov Conditional Forward Model with Cube Perturbations for Extreme Weather Forecasting [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16856)  
* Sub-Seasonal Climate Forecasting via Machine Learning: Challenges, Analysis, and Advances [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16090)  

#### Time Series Anomaly Detection
* Graph Neural Network-Based Anomaly Detection in Multivariate Time Series [\[paper\]](https://arxiv.org/abs/2106.06947) [\[official code\]](https://github.com/d-ailin/GDN) 
* Time Series Anomaly Detection with Multiresolution Ensemble Decoding [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17152)  
* Outlier Impact Characterization for Time Series Data [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17379) 

#### Time Series Classification
* Correlative Channel-Aware Fusion for Multi-View Time Series Classification [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/16830/16637)
* Learnable Dynamic Temporal Pooling for Time Series Classification [\[paper\]](https://arxiv.org/abs/2104.02577) [\[official code\]](https://github.com/donalee/DTW-Pool)
* ShapeNet: A Shapelet-Neural Network Approach for Multivariate Time Series Classification [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17018) 
* Joint-Label Learning by Dual Augmentation for Time Series Classification [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17071)  

#### Other Time Series Analysis
*  Time Series Domain Adaptation via Sparse Associative Structure Alignment [\[paper\]](https://arxiv.org/abs/2012.11797) [\[official code\]](https://github.com/DMIRLAB-Group/SASA)
*  Learning Representations for Incomplete Time Series Clustering [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17070)  
*  Generative Semi-Supervised Learning for Multivariate Time Series Imputation [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17086) [\[official code\]](https://github.com/zjuwuyy-DL/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation) 
*  Second Order Techniques for Learning Time-Series with Structural Breaks [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/17117)  



### IJCAI 2021
#### Time Series Forecasting
* Two Birds with One Stone: Series Saliency for Accurate and Interpretable Multivariate Time Series Forecasting [\[paper\]](https://www.ijcai.org/proceedings/2021/397)  
* Residential Electric Load Forecasting via Attentive Transfer of Graph Neural Networks [\[paper\]](https://www.ijcai.org/proceedings/2021/374)  
* Hierarchical Adaptive Temporal-Relational Modeling for Stock Trend Prediction [\[paper\]](https://www.ijcai.org/proceedings/2021/0508.pdf) 
* TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning [\[paper\]](https://arxiv.org/abs/2106.06273) [\[official code\]](https://arxiv.org/abs/2106.06273) 

#### Other Time Series Analysis
* Time Series Data Augmentation for Deep Learning: A Survey [\[paper\]](https://arxiv.org/abs/2002.12478) 
* Time-Series Representation Learning via Temporal and Contextual Contrasting [\[paper\]](https://arxiv.org/abs/2106.14112) [\[official code\]](https://arxiv.org/abs/2106.14112) 
* Adversarial Spectral Kernel Matching for Unsupervised Time Series Domain Adaptation [\[paper\]](https://www.ijcai.org/proceedings/2021/378) [\[official code\]](https://github.com/jarheadjoe/Adv-spec-ker-matching) 
* Time-Aware Multi-Scale RNNs for Time Series Modeling [\[paper\]](https://www.ijcai.org/proceedings/2021/315)  
* TE-ESN: Time Encoding Echo State Network for Prediction Based on Irregularly Sampled Time Series Data [\[paper\]](https://arxiv.org/abs/2105.00412)   


<!--  [\[paper\]]() [\[official code\]]()  --> 
### SIGMOD VLDB ICDE 2021
#### Time Series Forecasting
* AutoAI-TS:AutoAI for Time Series Forecasting, SIGMOD'21. [\[paper\]](https://arxiv.org/abs/2102.12347)  
* FlashP: An Analytical Pipeline for Real-time Forecasting of Time-Series Relational Data, VLDB'21. [\[paper\]](http://vldb.org/pvldb/vol14/p721-ding.pdf)
* MDTP: a multi-source deep traffic prediction framework over spatio-temporal trajectory data, VLDB'21. [\[paper\]]()
* EnhanceNet: Plugin Neural Networks for Enhancing Correlated Time Series Forecasting, ICDE'21. [\[paper\]](https://ieeexplore.ieee.org/document/9458855) [\[slides\]](https://pdfs.semanticscholar.org/3cb0/6f67fbfcf3c2dac32c02248a03eb84cc246d.pdf)  
* An Effective Joint Prediction Model for Travel Demands and Traffic Flows, ICDE'21. [\[paper\]](https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/icde21-traffic.pdf)  
 
#### Time Series Anomaly Detection
* Exathlon: A Benchmark for Explainable Anomaly Detection over Time Series, VLDB'21. [\[paper\]](https://arxiv.org/abs/2010.05073) [\[official code\]](https://github.com/exathlonbenchmark/exathlon)
* SAND: Streaming Subsequence Anomaly Detection, VLDB'21. [\[paper\]](http://vldb.org/pvldb/vol14/p1717-boniol.pdf)  

#### Other Time Series Analysis
* RobustPeriod: Robust Time-Frequency Mining for Multiple Periodicity Detection, SIGMOD'21. [\[paper\]](https://arxiv.org/abs/2002.09535) [\[code\]](https://github.com/ariaghora/robust-period)
* ORBITS: Online Recovery of Missing Values in Multiple Time Series Streams, VLDB'21. [\[paper\]](http://vldb.org/pvldb/vol14/p294-khayati.pdf) [\[official code\]](https://github.com/eXascaleInfolab/orbits)
* Missing Value Imputation on Multidimensional Time Series, VLDB'21. [\[paper\]](http://vldb.org/pvldb/vol14/p2533-bansal.pdf) 

<!--    , WSDM'21. [\[paper\]]() [\[official code\]]()   --> 
### Misc 2021
#### Time Series Forecasting
* DeepFEC: Energy Consumption Prediction under Real-World Driving Conditions for Smart Cities, WWW'21. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3442381.3449983) [\[official code\]](https://github.com/ElmiSay/DeepFEC)
* AutoSTG: Neural Architecture Search for Predictions of Spatio-Temporal Graph, WWW'21. [\[paper\]](http://panzheyi.cc/publication/pan2021autostg/paper.pdf) [\[official code\]](https://github.com/panzheyi/AutoSTG)
* REST: Reciprocal Framework for Spatiotemporal-coupled Predictions, WWW'21. [\[paper\]](https://s2.smu.edu/~jiazhang/Papers/JiaZhang-WWW2021-REST.pdf)
* Simultaneously Reconciled Quantile Forecasting of Hierarchically Related Time Series, AISTATS'21. [\[paper\]](http://proceedings.mlr.press/v130/han21a/han21a.pdf)  
* SSDNet: State Space Decomposition Neural Network for Time Series Forecasting, ICDM'21. [\[paper\]](https://arxiv.org/abs/2112.10251)  
* AdaRNN: Adaptive Learning and Forecasting of Time Series, CIKM'21. [\[paper\]](https://arxiv.org/abs/2108.04443) [\[official code\]](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn)
* Learning to Learn the Future: Modeling Concept Drifts in Time Series Prediction, CIKM'21. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3459637.3482271)  
* Stock Trend Prediction with Multi-Granularity Data: A Contrastive Learning Approach with Adaptive Fusion, CIKM'21. [\[paper\]](http://staff.ustc.edu.cn/~cheneh/paper_pdf/2021/Min-Hou-CIKM.pdf)  
* DL-Traff: Survey and Benchmark of Deep Learning Models for Urban Traffic Prediction, CIKM'21. [\[paper\]](https://arxiv.org/abs/2108.09091) [\[official code1\]](https://github.com/deepkashiwa20/dl-traff-graph) [\[official code2\]](https://github.com/deepkashiwa20/dl-traff-grid)
* Long Horizon Forecasting With Temporal Point Processes, WSDM'21. [\[paper\]](https://arxiv.org/abs/2101.02815) [\[official code\]](https://github.com/pratham16cse/DualTPP)
* Time-Series Event Prediction with Evolutionary State Graph, WSDM'21. [\[paper\]](https://arxiv.org/abs/1905.05006) [\[official code\]](https://github.com/VachelHU/EvoNet).

#### Time Series Anomaly Detection
* SDFVAE: Static and Dynamic Factorized VAE for Anomaly Detection of Multivariate CDN KPIs, WWW'21. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3442381.3450013) 
* Time Series Change Point Detection with Self-Supervised Contrastive Predictive Coding, WWW'21. [\[paper\]](https://arxiv.org/abs/2011.14097) [\[official code\]](https://github.com/cruiseresearchgroup/TSCP2)
* FluxEV: A Fast and Effective Unsupervised Framework for Time-Series Anomaly Detection, WSDM'21. [\[paper\]](https://dl.acm.org/doi/10.1145/3437963.3441823) 
* Weakly Supervised Temporal Anomaly Segmentation with Dynamic Time Warping, ICCV'21. [\[paper\]](https://dl.acm.org/doi/10.1145/3437963.3441823) [\[official code\]](https://github.com/donalee/wetas)
* Jump-Starting Multivariate Time Series Anomaly Detection for Online Service Systems, ATC'21. [\[paper\]](https://www.usenix.org/conference/atc21/presentation/ma)



#### Other Time Series Analysis
* Network of Tensor Time Series, WWW'21. [\[paper\]](https://arxiv.org/abs/2102.07736) [\[official code\]](https://github.com/baoyujing/NET3)
* Radflow: A Recurrent, Aggregated, and Decomposable Model for Networks of Time Series, WWW'21. [\[paper\]](https://arxiv.org/abs/2102.07289) [\[official code\]](https://github.com/alasdairtran/radflow)
* SrVARM: State Regularized Vector Autoregressive Model for Joint Learning of Hidden State Transitions and State-Dependent Inter-Variable Dependencies from Multi-variate Time Series, WWW'21. [\[paper\]](https://faculty.ist.psu.edu/vhonavar/Papers/SRVARM.pdf)  
* Deep Fourier Kernel for Self-Attentive Point Processes, AISTATS'21. [\[paper\]](https://proceedings.mlr.press/v130/zhu21b.html)
* Differentiable Divergences Between Time Series, AISTATS'21. [\[paper\]](https://arxiv.org/abs/2010.08354) [\[official code\]](https://github.com/google-research/soft-dtw-divergences) 
* Aligning Time Series on Incomparable Spaces, AISTATS'21. [\[paper\]](https://arxiv.org/abs/2006.12648) [\[official code\]](https://github.com/samcohen16/Aligning-Time-Series) 
* Continual Learning for Multivariate Time Series Tasks with Variable Input Dimensions, ICDM'21. [\[paper\]](https://arxiv.org/abs/2203.06852)  
* Towards Generating Real-World Time Series Data, ICDM'21. [\[paper\]](https://arxiv.org/abs/2111.08386) [\[official code\]](https://github.com/acphile/RTSGAN)
* Learning Saliency Maps to Explain Deep Time Series Classifiers, CIKM'21. [\[paper\]](https://kingspp.github.io/publications/) [\[official code\]](https://github.com/kingspp/timeseries-explain)
* Actionable Insights in Urban Multivariate Time-series, CIKM'21. [\[paper\]](https://people.cs.vt.edu/anikat1/publications/ratss-cikm2021.pdf) 
* Explainable Multivariate Time Series Classification: A Deep Neural Network Which Learns To Attend To Important Variables As Well As Informative Time Intervals, WSDM'21. [\[paper\]](https://arxiv.org/abs/2011.11631)  


## AI4TS Papers 201X-2020 Selected

### NeurIPS 201X-2020

#### Time Series Forecasting
* Adversarial Sparse Transformer for Time Series Forecasting, NeurIPS'20. [\[paper\]](https://proceedings.neurips.cc//paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf) [\[official code\]](https://github.com/hihihihiwsf/AST) 
* Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2103.07719) [\[official code\]](https://github.com/microsoft/StemGNN) 
* Deep Rao-Blackwellised Particle Filters for Time Series Forecasting, NeurIPS'20. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/afb0b97df87090596ae7c503f60bb23f-Abstract.html) 
* Probabilistic Time Series Forecasting with Shape and Temporal Diversity, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2010.07349) [\[official code\]](https://github.com/vincent-leguen/STRIPE) 
* Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2007.02842) [\[official code\]](https://github.com/LeiBAI/AGCRN) 
* Interpretable Sequence Learning for Covid-19 Forecasting, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2008.00646) 
* Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1907.00235) [\[code\]](https://github.com/mlpotter/Transformer_Time_Series) 
* Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1905.03806) [\[official code\]](https://github.com/rajatsen91/deepglo) 
* High-dimensional multivariate forecasting with low-rank Gaussian Copula Processes, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1910.03002) [\[official code\]](https://github.com/mbohlkeschneider/gluon-ts) 
* Deep State Space Models for Time Series Forecasting, NeurIPS'18. [\[paper\]](https://proceedings.neurips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html)  
* Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction, NeurIPS'16. [\[paper\]](https://papers.nips.cc/paper/2016/hash/85422afb467e9456013a2a51d4dff702-Abstract.html)  

#### Time Series Anomaly Detection
* Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network, NeurIPS'20. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/97e401a02082021fd24957f852e0e475-Abstract.html)  
* PIDForest: Anomaly Detection via Partial Identification, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1912.03582) [\[official code\]](https://github.com/vatsalsharan/pidforest) 
* Precision and Recall for Time Series, NeurIPS'18. [\[paper\]](https://arxiv.org/abs/1803.03639) [\[official code\]](https://github.com/IntelLabs/TSAD-Evaluator) 

#### Time Series Classification
* Shallow RNN: Accurate Time-series Classification on Resource Constrained Devices, NeurIPS'19. [\[paper\]](https://proceedings.neurips.cc/paper/2019/hash/76d7c0780ceb8fbf964c102ebc16d75f-Abstract.html)  
#### Time Series Clustering
* Learning Representations for Time Series Clustering, NeurIPS'19. [\[paper\]](https://papers.nips.cc/paper/2019/hash/1359aa933b48b754a2f54adb688bfa77-Abstract.html) [\[official code\]](https://github.com/qianlima-lab/DTCR) 
* Learning low-dimensional state embeddings and metastable clusters from time series data, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1906.00302)

#### Time Series Imputation
* NAOMI: Non-autoregressive multiresolution sequence imputation, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1901.10946) [\[official code\]](https://github.com/felixykliu/NAOMI) 
* BRITS: Bidirectional Recurrent Imputation for Time Series, NeurIPS'18. [\[paper\]](https://arxiv.org/abs/1805.10572) [\[official code\]](https://github.com/caow13/BRITS) 
* Multivariate Time Series Imputation with Generative Adversarial Networks, NeurIPS'18. [\[paper\]](https://papers.nips.cc/paper/2018/hash/96b9bff013acedfb1d140579e2fbeb63-Abstract.html) [\[official code\]](https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks) 

#### Time Series Neural xDE
* Neural Controlled Differential Equations for Irregular Time Series, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2005.08926) [\[official code\]](https://github.com/patrick-kidger/NeuralCDE)  
* GRU-ODE-Bayes: Continuous Modeling of Sporadically-Observed Time Series, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1905.12374) [\[official code\]](https://github.com/edebrouwer/gru_ode_bayes)  
* Latent Ordinary Differential Equations for Irregularly-Sampled Time Series, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1907.03907) [\[official code\]](https://github.com/YuliaRubanova/latent_ode)  
* Neural Ordinary Differential Equations, NeurIPS'18. [\[paper\]](https://arxiv.org/abs/1806.07366) [\[official code\]](https://github.com/rtqichen/torchdiffeq)  

#### General Time Series Analysis 
* High-recall causal discovery for autocorrelated time series with latent confounders, NeurIPS'20. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/94e70705efae423efda1088614128d0b-Abstract.html) [\[paper2\]](https://arxiv.org/abs/2007.01884) [\[official code\]](https://github.com/jakobrunge/tigramite) 
* Benchmarking Deep Learning Interpretability in Time Series Predictions, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2010.13924) [\[official code\]](https://github.com/ayaabdelsalam91/TS-Interpretability-Benchmark)
* What went wrong and when? Instance-wise feature importance for time-series black-box models, NeurIPS'20. [\[paper\]](https://arxiv.org/abs/2003.02821) [\[official code\]]()
* Normalizing Kalman Filters for Multivariate Time Series Analysis, NeurIPS'20. [\[paper\]](https://proceedings.neurips.cc/paper/2020/hash/1f47cef5e38c952f94c5d61726027439-Abstract.html)
* Unsupervised Scalable Representation Learning for Multivariate Time Series, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1901.10738) [\[official code\]](https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries)
* Time-series Generative Adversarial Networks, NeurIPS'19. [\[paper\]](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) [\[official code\]](https://github.com/jsyoon0823/TimeGAN) 
* U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging, NeurIPS'19. [\[paper\]](https://arxiv.org/abs/1910.11162) [\[official code\]](https://github.com/perslev/U-Time) 
* Autowarp: Learning a Warping Distance from Unlabeled Time Series Using Sequence Autoencoders, NeurIPS'18. [\[paper\]](https://arxiv.org/abs/1810.10107)
* Safe Active Learning for Time-Series Modeling with Gaussian Processes, NeurIPS'18. [\[paper\]](https://proceedings.neurips.cc/paper/2018/hash/b197ffdef2ddc3308584dce7afa3661b-Abstract.html)  
 
### ICML 201X-2020

#### General Time Series Analysis
* Learning from Irregularly-Sampled Time Series: A Missing Data Perspective, ICML'20. [\[paper\]](https://arxiv.org/abs/2008.07599) [\[official code\]](https://github.com/steveli/partial-encoder-decoder)
* Set Functions for Time Series, ICML'20. [\[paper\]](https://arxiv.org/abs/1909.12064) [\[official code\]](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series)
* Time Series Deconfounder: Estimating Treatment Effects over Time in the Presence of Hidden Confounders, ICML'20. [\[paper\]](https://arxiv.org/abs/1902.00450) [\[official code\]](https://github.com/ioanabica/Time-Series-Deconfounder)
* Spectral Subsampling MCMC for Stationary Time Series, ICML'20. [\[paper\]](https://proceedings.mlr.press/v119/salomone20a.html)  
* Learnable Group Transform For Time-Series, ICML'20. [\[paper\]](https://proceedings.mlr.press/v119/cosentino20a.html) 
* Causal Discovery and Forecasting in Nonstationary Environments with State-Space Models, ICML'19. [\[paper\]](https://arxiv.org/abs/1905.10857) [\[official code\]](https://github.com/Biwei-Huang/Causal-discovery-and-forecasting-in-nonstationary-environments)
* Discovering Latent Covariance Structures for Multiple Time Series, ICML'19. [\[paper\]](https://arxiv.org/abs/1703.09528) 
* Autoregressive convolutional neural networks for asynchronous time series, ICML'18. [\[paper\]](https://arxiv.org/abs/1703.04122) [\[official code\]](https://github.com/mbinkowski/nntimeseries)
* Hierarchical Deep Generative Models for Multi-Rate Multivariate Time Series, ICML'18. [\[paper\]](https://proceedings.mlr.press/v80/che18a.html)  
* Soft-DTW: a Differentiable Loss Function for Time-Series, ICML'17. [\[paper\]](https://arxiv.org/abs/1703.01541) [\[official code\]](https://github.com/mblondel/soft-dtw)


#### Time Series Forecasting
* Forecasting Sequential Data Using Consistent Koopman Autoencoders, ICML'20. [\[paper\]](https://arxiv.org/abs/2003.02236) [\[official code\]](https://github.com/erichson/koopmanAE)
* Adversarial Attacks on Probabilistic Autoregressive Forecasting Models, ICML'20. [\[paper\]](https://arxiv.org/abs/2003.03778) [\[official code\]](https://github.com/eth-sri/probabilistic-forecasts-attacks)
* Influenza Forecasting Framework based on Gaussian Processes, ICML'20. [\[paper\]](http://proceedings.mlr.press/v119/zimmer20a.html) 
* Deep Factors for Forecasting, ICML'19. [\[paper\]](https://arxiv.org/abs/1905.12417)  
* Coherent Probabilistic Forecasts for Hierarchical Time Series, ICML'17. [\[paper\]](https://proceedings.mlr.press/v70/taieb17a.html) 

### ICLR 201X-2020
#### General Time Series Analysis
* Interpolation-Prediction Networks for Irregularly Sampled Time Series, ICLR'19. [\[paper\]](https://openreview.net/forum?id=r1efr3C9Ym) [\[official code\]](https://github.com/mlds-lab/interp-net)
* SOM-VAE: Interpretable Discrete Representation Learning on Time Series, ICLR'19. [\[paper\]](https://openreview.net/forum?id=rygjcsR9Y7) [\[official code\]](https://github.com/ratschlab/SOM-VAE)

#### Time Series Forecasting
* N-BEATS: Neural basis expansion analysis for interpretable time series forecasting, ICLR'20. [\[paper\]](https://openreview.net/forum?id=r1ecqn4YwB) [\[official code\]](https://github.com/ElementAI/N-BEATS)
* Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting, ICLR'18. [\[paper\]](https://openreview.net/forum?id=SJiHXGWAZ) [\[official code\]](https://github.com/liyaguang/DCRNN) 
* Automatically Inferring Data Quality for Spatiotemporal Forecasting, ICLR'18. [\[paper\]](https://openreview.net/forum?id=ByJIWUnpW) 

 
### KDD 201X-2020

#### General Time Series Analysis
* Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns, KDD'20. [\[paper\]](https://www.researchgate.net/profile/Qingsong-Wen/publication/343780200_Fast_RobustSTL_Efficient_and_Robust_Seasonal-Trend_Decomposition_for_Time_Series_with_Complex_Patterns/links/614b9828a3df59440ba498b3/Fast-RobustSTL-Efficient-and-Robust-Seasonal-Trend-Decomposition-for-Time-Series-with-Complex-Patterns.pdf) [\[code\]](https://github.com/ariaghora/fast-robust-stl)
* Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data, KDD'20. [\[paper\]](https://arxiv.org/abs/2005.10996) [\[official code\]](https://github.com/floft/codats)
* Online Amnestic DTW to allow Real-Time Golden Batch Monitoring, KDD'19. [\[paper\]](https://dl.acm.org/doi/10.1145/3292500.3330650) 
* Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis, KDD'18. [\[paper\]](https://arxiv.org/abs/1806.08946)  
* Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data, KDD'17. [\[paper\]](https://arxiv.org/abs/1706.03161) 


#### Time Series Forecasting
* Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks, KDD'20. [\[paper\]](https://arxiv.org/abs/2005.11650) [\[official code\]](https://github.com/nnzhan/MTGNN)
* Attention based Multi-Modal New Product Sales Time-series Forecasting, KDD'20. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3394486.3403362)
* Forecasting the Evolution of Hydropower Generation, KDD'20. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3394486.3403337)
* Modeling Extreme Events in Time Series Prediction, KDD'19. [\[paper\]](http://staff.ustc.edu.cn/~hexn/papers/kdd19-timeseries.pdf)  
* Multi-Horizon Time Series Forecasting with Temporal Attention Learning, KDD'19. [\[paper\]](https://dl.acm.org/doi/10.1145/3292500.3330662)
* Regularized Regression for Hierarchical Forecasting Without Unbiasedness Conditions, KDD'19. [\[paper\]](https://souhaib-bentaieb.com/papers/2019_kdd_hts_reg.pdf)
* Streaming Adaptation of Deep Forecasting Models using Adaptive Recurrent Units, KDD'19. [\[paper\]](https://arxiv.org/abs/1906.09926) [\[official code\]](https://github.com/pratham16/ARU)
* Dynamic Modeling and Forecasting of Time-evolving Data Streams, KDD'19. [\[paper\]](https://www.dm.sanken.osaka-u.ac.jp/~yasuko/PUBLICATIONS/kdd19-orbitmap.pdf) [\[official code\]](https://github.com/yasuko-matsubara/orbitmap)
* DeepUrbanEvent: A System for Predicting Citywide Crowd Dynamics at Big Events, KDD'19. [\[paper\]](https://www.researchgate.net/profile/Renhe-Jiang/publication/334714928_DeepUrbanEvent_A_System_for_Predicting_Citywide_Crowd_Dynamics_at_Big_Events/links/5d417167299bf1995b597f28/DeepUrbanEvent-A-System-for-Predicting-Citywide-Crowd-Dynamics-at-Big-Events.pdf) [\[official code\]](https://github.com/deepkashiwa20/DeepUrbanEvent)
* Stock Price Prediction via Discovering Multi-Frequency Trading Patterns, KDD'17. [\[paper\]](https://www.eecs.ucf.edu/~gqi/publications/kdd2017_stock.pdf) [\[official code\]](https://github.com/z331565360/State-Frequency-Memory-stock-prediction)

#### Time Series Anomaly Detection
* USAD: UnSupervised Anomaly Detection on Multivariate Time Series, KDD'20. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392) [\[official code\]](https://github.com/manigalati/usad)
* RobustTAD: Robust Time Series Anomaly Detection via Decomposition and Convolutional Neural Networks, KDD'20 MiLeTS. [\[paper\]](https://arxiv.org/abs/2002.09545)
* Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network, KDD'19. [\[paper\]](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf) [\[official code\]](https://github.com/NetManAIOps/OmniAnomaly)
* Time-Series Anomaly Detection Service at Microsoft, KDD'19. [\[paper\]](https://arxiv.org/abs/1906.03821) 
* Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding, KDD'18. [\[paper\]](https://arxiv.org/abs/1802.04431) [\[official code\]](https://github.com/khundman/telemanom)
* Anomaly Detection in Streams with Extreme Value Theory, KDD'17. [\[paper\]](https://hal.archives-ouvertes.fr/hal-01640325/document)


 
### AAAI 201X-2020

#### General Time Series Analysis
* Time2Graph: Revisiting Time Series Modeling with Dynamic Shapelets, AAAI'20. [\[paper\]](https://arxiv.org/abs/1911.04143) [\[official code\]](https://github.com/petecheng/Time2Graph) 
* DATA-GRU: Dual-Attention Time-Aware Gated Recurrent Unit for Irregular Multivariate Time Series, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5440)
* Tensorized LSTM with Adaptive Shared Memory for Learning Trends in Multivariate Time Series, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5496) [\[official code\]](https://github.com/DerronXu/DeepTrends) 
* Factorized Inference in Deep Markov Models for Incomplete Multimodal Time Series, AAAI'20. [\[paper\]](https://arxiv.org/abs/1905.13570) [\[official code\]](https://github.com/ztangent/multimodal-dmm) 
* Relation Inference among Sensor Time Series in Smart Buildings with Metric Learning, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5900) 
* TapNet: Multivariate Time Series Classification with Attentional Prototype Network, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/6165) [\[official code\]](https://github.com/xuczhang/tapnet) 
* RobustSTL: A Robust Seasonal-Trend Decomposition Procedure for Long Time Series, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/4480) [\[code\]](https://github.com/LeeDoYup/RobustSTL)
* Estimating the Causal Effect from Partially Observed Time Series, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/4281)
* Adversarial Unsupervised Representation Learning for Activity Time-Series, AAAI'19. [\[paper\]](https://arxiv.org/abs/1811.06847)
* Fourier Feature Approximations for Periodic Kernels in Time-Series Modelling, AAAI'18. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/11696)

#### Time Series Forecasting
* Joint Modeling of Local and Global Temporal Dynamics for Multivariate Time Series Forecasting with Missing Values, AAAI'20. [\[paper\]](https://arxiv.org/abs/1911.10273)  
* Block Hankel Tensor ARIMA for Multiple Short Time Series Forecasting, AAAI'20. [\[paper\]](https://arxiv.org/abs/2002.12135) [\[official code\]](https://github.com/yokotatsuya/BHT-ARIMA) 
* Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework for Spatial-Temporal Network Data Forecasting, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5438) [\[official code\]](https://github.com/Davidham3/STSGCN) 
* Self-Attention ConvLSTM for Spatiotemporal Prediction, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/6819) 
* Multi-Range Attentive Bicomponent Graph Convolutional Network for Traffic Forecasting, AAAI'20. [\[paper\]](https://arxiv.org/abs/1911.12093)  
* Spatio-Temporal Graph Structure Learning for Traffic Forecasting, AAAI'20. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/5470) 
* GMAN: A Graph Multi-Attention Network for Traffic Prediction, AAAI'20. [\[paper\]](https://arxiv.org/abs/1911.08415) [\[official code\]](https://github.com/zhengchuanpan/GMAN) 
* Cogra: Concept-drift-aware Stochastic Gradient Descent for Time-series Forecasting, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/4383)  
* Dynamic Spatial-Temporal Graph Convolutional Neural Networks for Traffic Forecasting, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/3877) 
* Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting, AAAI'19. [\[paper\]](https://ojs.aaai.org//index.php/AAAI/article/view/3881) [\[official code\]](https://github.com/guoshnBJTU/ASTGCN-r-pytorch)
* MRes-RGNN: A Novel Deep Learning based Framework for Traffic Prediction, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/3821)
* DeepSTN+: Context-aware Spatial Temporal Neural Network for Crowd Flow Prediction in Metropolis, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/3892) [\[official code\]](https://github.com/FIBLAB/DeepSTN)
* Incomplete Label Multi-task Deep Learning for Spatio-temporal Event Subtype Forecasting, AAAI'19. [\[paper\]](http://cs.emory.edu/~lzhao41/materials/papers/main_AAAI2019.pdf) 
* Learning Heterogeneous Spatial-Temporal Representation for Bike-sharing Demand Prediction, AAAI'19. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/3890)  
* Spatiotemporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting, AAAI'19. [\[paper\]](https://ojs.aaai.org//index.php/AAAI/article/view/4247) 

#### Time Series Anomaly Detection
* A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data, AAAI'19. [\[paper\]](https://arxiv.org/abs/1811.08055)
* Non-parametric Outliers Detection in Multiple Time Series A Case Study: Power Grid Data Analysis, AAAI'18. [\[paper\]](https://ojs.aaai.org/index.php/AAAI/article/view/11632)

 
### IJCAI 201X-2020

#### General Time Series Analysis
* RobustTrend: A Huber Loss with a Combined First and Second Order Difference Regularization for Time Series Trend Filtering, IJCAI'19. [\[paper\]](https://arxiv.org/abs/1906.03751) 
* E2GAN: End-to-End Generative Adversarial Network for Multivariate Time Series Imputation, IJCAI'19. [\[paper\]](https://www.ijcai.org/Proceedings/2019/0429.pdf)
* Causal Inference in Time Series via Supervised Learning, IJCAI'18. [\[paper\]](https://www.ijcai.org/proceedings/2018/282)

#### Time Series Forecasting
* PewLSTM: Periodic LSTM with Weather-Aware Gating Mechanism for Parking Behavior Prediction, IJCAI'20. [\[paper\]](https://www.ijcai.org/proceedings/2020/610) [\[official code\]](https://github.com/NingxuanFeng/PewLSTM)
* LSGCN: Long Short-Term Traffic Prediction with Graph Convolutional Networks, IJCAI'20. [\[paper\]](https://www.ijcai.org/proceedings/2020/326)
* Cross-Interaction Hierarchical Attention Networks for Urban Anomaly Prediction, IJCAI'20. [\[paper\]](https://www.ijcai.org/proceedings/2020/601)
* Learning Interpretable Deep State Space Model for Probabilistic Time Series Forecasting, IJCAI'19. [\[paper\]](https://arxiv.org/abs/2102.00397)
* Explainable Deep Neural Networks for Multivariate Time Series Predictions, IJCAI'19. [\[paper\]](https://www.ijcai.org/proceedings/2019/932)
* Periodic-CRN: A Convolutional Recurrent Model for Crowd Density Prediction with Recurring Periodic Patterns. [\[paper\]](https://www.ijcai.org/proceedings/2018/519)
* Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting. [\[paper\]](https://arxiv.org/abs/1709.04875) [\[official code\]](https://github.com/VeritasYin/STGCN_IJCAI-18)
* LC-RNN: A Deep Learning Model for Traffic Speed Prediction. [\[paper\]](https://www.ijcai.org/proceedings/2018/482)
* GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction, IJCAI'18. [\[paper\]](https://www.ijcai.org/proceedings/2018/476) [\[official code\]](https://github.com/yoshall/GeoMAN)
* Hierarchical Electricity Time Series Forecasting for Integrating Consumption Patterns Analysis and Aggregation Consistency, IJCAI'18. [\[paper\]](https://www.ijcai.org/proceedings/2018/487)
* NeuCast: Seasonal Neural Forecast of Power Grid Time Series, IJCAI'18. [\[paper\]](https://www.ijcai.org/Proceedings/2018/460) [\[official code\]](https://github.com/chenpudigege/NeuCast)
* A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction, IJCAI'17. [\[paper\]](https://arxiv.org/abs/1704.02971) [\[code\]](https://paperswithcode.com/paper/a-dual-stage-attention-based-recurrent-neural)
* Hybrid Neural Networks for Learning the Trend in Time Series, IJCAI'17. [\[paper\]](https://www.ijcai.org/proceedings/2017/316)

#### Time Series Anomaly Detection
* BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series, IJCAI'19. [\[paper\]](https://www.ijcai.org/proceedings/2019/616) [\[official code\]](https://github.com/hi-bingo/BeatGAN) 
* Outlier Detection for Time Series with Recurrent Autoencoder Ensembles, IJCAI'19. [\[paper\]](https://www.ijcai.org/proceedings/2019/378) [\[official code\]](https://github.com/tungk/OED) 
* Stochastic Online Anomaly Analysis for Streaming Time Series, IJCAI'17. [\[paper\]](https://www.ijcai.org/proceedings/2017/0445.pdf)


#### Time Series Clustering
* Linear Time Complexity Time Series Clustering with Symbolic Pattern Forest, IJCAI'19. [\[paper\]](https://www.ijcai.org/proceedings/2019/406)
* Similarity Preserving Representation Learning for Time Series Clustering, IJCAI'19. [\[paper\]](https://arxiv.org/abs/1702.03584)


#### Time Series Classification
* A new attention mechanism to classify multivariate time series, IJCAI'20. [\[paper\]](https://www.ijcai.org/proceedings/2020/277)

 
### SIGMOD VLDB ICDE 201X-2020
#### General Time Series Analysis
* Debunking Four Long-Standing Misconceptions of Time-Series Distance Measures, SIGMOD'20. [\[paper\]](http://people.cs.uchicago.edu/~jopa/Papers/PaparrizosSIGMOD2020.pdf) [\[official code\]](https://github.com/johnpaparrizos/TSDistEval) 
* Database Workload Capacity Planning using Time Series Analysis and Machine Learning, SIGMOD'20. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3318464.3386140) 
* Mind the gap: an experimental evaluation of imputation of missing values techniques in time series, VLDB'20. [\[paper\]](http://www.vldb.org/pvldb/vol13/p768-khayati.pdf) [\[official code\]](https://github.com/eXascaleInfolab/bench-vldb20) 
* Active Model Selection for Positive Unlabeled Time Series Classification, ICDE'20. [\[paper\]](https://ieeexplore.ieee.org/document/9101367) [\[official code\]](https://github.com/sliang11/Active-Model-Selection-for-PUTSC) 
* ExplainIt! -- A declarative root-cause analysis engine for time series data, SIGMOD'19. [\[paper\]](https://arxiv.org/abs/1903.08132) 
* Cleanits: A Data Cleaning System for Industrial Time Series, VLDB'19. [\[paper\]](https://vldb.org/pvldb/vol12/p1786-ding.pdf) 
* Matrix Profile X: VALMOD - Scalable Discovery of Variable-Length Motifs in Data Series, SIGMOD'18. [\[paper\]](https://helios2.mi.parisdescartes.fr/~themisp/publications/sigmod18-valmod.pdf) 
* Effective Temporal Dependence Discovery in Time Series Data, VLDB'18. [\[paper\]](https://vldb.org/pvldb/vol11/p893-cai.pdf) 

#### Time Series Anomaly Detection
* Series2Graph: graph-based subsequence anomaly detection for time series, VLDB'20. [\[paper\]](http://www.vldb.org/pvldb/vol13/p1821-boniol.pdf) [\[official code\]](https://helios2.mi.parisdescartes.fr/~themisp/series2graph/) 
* Neighbor Profile: Bagging Nearest Neighbors for Unsupervised Time Series Mining, ICDE'20. [\[paper\]](https://www.researchgate.net/profile/Yuanduo-He/publication/340663191_Neighbor_Profile_Bagging_Nearest_Neighbors_for_Unsupervised_Time_Series_Mining/links/5e97d607a6fdcca7891c2a0b/Neighbor-Profile-Bagging-Nearest-Neighbors-for-Unsupervised-Time-Series-Mining.pdf)  
* Automated Anomaly Detection in Large Sequences, ICDE'20. [\[paper\]](https://helios2.mi.parisdescartes.fr/~themisp/publications/icde20-norm.pdf) [\[official code\]](https://helios2.mi.parisdescartes.fr/~themisp/norm/) 
* User-driven error detection for time series with events, ICDE'20. [\[paper\]](https://www.eurecom.fr/en/publication/6192/download/data-publi-6192.pdf)


<!--    , Misc'20. [\[paper\]]() [\[official code\]]()   WWW, AISTAT, CIKM, ICDM, WSDM, SIGIR, ATC, etc. --> 
### Misc 201X-2020
#### General Time Series Analysis
* STFNets: Learning Sensing Signals from the Time-Frequency Perspective with Short-Time Fourier Neural Networks, WWW'19. [\[paper\]](https://arxiv.org/abs/1902.07849) [\[official code\]](https://github.com/yscacaca/STFNets)
* GP-VAE: Deep probabilistic time series imputation, AISTATS'20. [\[paper\]](https://arxiv.org/abs/1907.04155) [\[official code\]](https://github.com/ratschlab/GP-VAE)
* DYNOTEARS: Structure Learning from Time-Series Data, AISTATS'20. [\[paper\]](https://arxiv.org/abs/2002.00498)
* Personalized Imputation on Wearable-Sensory Time Series via Knowledge Transfer, CIKM'20. [\[paper\]](https://dl.acm.org/doi/pdf/10.1145/3340531.3411879)
* Order-Preserving Metric Learning for Mining Multivariate Time Series, ICDM'20. [\[paper\]](https://par.nsf.gov/servlets/purl/10233687)
* Learning Periods from Incomplete Multivariate Time Series, ICDM'20. [\[paper\]](http://www.cs.albany.edu/~petko/lab/papers/zgzb2020icdm.pdf)
* Foundations of Sequence-to-Sequence Modeling for Time Series, AISTATS'19. [\[paper\]](https://arxiv.org/abs/1805.03714)

#### Time Series Forecasting
* Hierarchically Structured Transformer Networks for Fine-Grained Spatial Event Forecasting, WWW'20. [\[paper\]](https://dl.acm.org/doi/10.1145/3366423.3380296)
* HTML: Hierarchical Transformer-based Multi-task Learning for Volatility Prediction, WWW'20. [\[paper\]](https://www.researchgate.net/publication/340385140_HTML_Hierarchical_Transformer-based_Multi-task_Learning_for_Volatility_Prediction) [\[official code\]](https://github.com/YangLinyi/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction)
* Traffic Flow Prediction via Spatial Temporal Graph Neural Network, WWW'20. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3366423.3380186)
* Towards Fine-grained Flow Forecasting: A Graph Attention Approach for Bike Sharing Systems, WWW'20. [\[paper\]](https://uconnuclab.github.io/publications/2020/Conference/he-www-2020-a.pdf) 
* Domain Adaptive Multi-Modality Neural Attention Network for Financial Forecasting, WWW'20. [\[paper\]](https://par.nsf.gov/servlets/purl/10161328)
* Spatiotemporal Hypergraph Convolution Network for Stock Movement Forecasting, ICDM'20. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/9338303)
* Probabilistic Forecasting with Spline Quantile Function RNNs, AISTATS'19. [\[paper\]](http://proceedings.mlr.press/v89/gasthaus19a.html)
* DSANet: Dual self-attention network for multivariate time series forecasting, CIKM'19. [\[paper\]](https://dl.acm.org/doi/abs/10.1145/3357384.3358132)
* RESTFul: Resolution-Aware Forecasting of Behavioral Time Series Data, CIKM'18. [\[paper\]](https://www3.nd.edu/~dial/publications/xian2018restful.pdf)
* Forecasting Wavelet Transformed Time Series with Attentive Neural Networks, ICDM'18. [\[paper\]](https://ieeexplore.ieee.org/abstract/document/8595010)
* A Flexible Forecasting Framework for Hierarchical Time Series with Seasonal Patterns: A Case Study of Web Traffic, SIGIR'18. [\[paper\]](https://people.cs.pitt.edu/~milos/research/2018/SIGIR_18_Liu_Hierarchical_Seasonal_TS.pdf)
* Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks, SIGIR'18. [\[paper\]](https://arxiv.org/abs/1703.07015) [\[official code\]](https://github.com/laiguokun/LSTNet) 

#### Time Series Anomaly Detection
* Multivariate Time-series Anomaly Detection via Graph Attention Network, ICDM'20. [\[paper\]](https://arxiv.org/abs/2009.02040) [\[code\]](https://github.com/ML4ITS/mtad-gat-pytorch)
* MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive Time Series Archives, ICDM'20. [\[paper\]](https://www.cs.ucr.edu/~eamonn/MERLIN_Long_version_for_website.pdf) [\[official code\]](https://sites.google.com/view/merlin-find-anomalies/MERLIN) 
* Cross-dataset Time Series Anomaly Detection for Cloud Systems, ATC'19. [\[paper\]](https://www.usenix.org/conference/atc19/presentation/zhang-xu) 
* Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications, WWW'18. [\[paper\]](https://arxiv.org/abs/1802.03903) [\[official code\]](https://github.com/NetManAIOps/donut)

 

