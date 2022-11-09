# SWEIPENET-CMA
This is the reproduced version (run on TF2) of the following published paper.
Chen Long, Zhou Feixiang, Wang, Shengke, Dong, Junyu,..., and Zhou, Huiyu. "SWIPENET: Object detection in noisy underwater scenes", Pattern Recognition Volume 132, 2022.

# Abstract
Deep learning based object detection methods have achieved promising performance in controlled envi-
ronments. However, these methods lack sufficient capabilities to handle underwater object detection due to these challenges: (1) images in the underwater datasets and real applications are blurry whilst accompanying severe noise that confuses the detectors and (2) objects in real applications are usually small. In this paper, we propose a Sample-WeIghted hyPEr Network (SWIPENET), and a novel training paradigm named Curriculum Multi-Class Adaboost (CMA), to address these two problems at the same time. Firstly, the backbone of SWIPENET produces multiple high resolution and semantic-rich Hyper Feature Maps, which significantly improve small object detection. Secondly, inspired by the human education process that drives the learning from easy to hard concepts, we propose the noise-robust CMA training paradigm that learns the clean data first and then move on to learns the diverse noisy data. Experiments on four underwater object detection datasets show that the proposed SWIPENET+CMA framework achieves better or competitive accuracy in object detection against several state-of-the-art approaches

# Dependencies
* python >= 3.5
* keras
* numpy >= 1.18.0
* scipy
# DataSets
The underwater robot picking contest datasets is organized by National Natural Science Foundation of China and Dalian
Municipal People’s Government. The Chinese website is http://www.cnurpc.org/index.html and the English website is http:
//en.cnurpc.org/. The contest holds annually from 2017, consisting of online and offline object detection contests. In this
paper, we use URPC2017 and URPC2018 datasets from the online object detection contest. To use the datasets, participants need to communicate with zhuming@dlut.edu.cn and sign a commitment letter for data usage: http://www.cnurpc.org/a/js/2018/0914/102.html
# Usage
**Training stage:**
1. Train the first detection model:
```python ssd512_training.py```
2. Update the weights using CMA algorithm:

2.1 Train the deep model in the NECMA stage (on URPC2018, train 2 deep models in NECMA stage, then turn to train deep models in NLCMA stage)
(1) Set the variable stage='NECMA' in ssd512_updateweight.py, then run 
```python ssd512_updateweight.py```

2.2 Train the deep model in the NLCMA stage
Set the variable stage='NLCMA' in ssd512_updateweight.py, then run 
```python ssd512_updateweight.py```

**Testing stage:**  
3.1 Compute the detections for each class.
```python ssd512_evaluation.py```

3.2 Ouput the average precison (AP) of each class and the mean AP 
```evaluationcode/main.m```

# Citation
If you use these models in your research, please cite:
```
@ article{LongChenCV,  
	author = {Chen Long, Zhou Feixiang, Wang, Shengke, Dong, Junyu,..., and Zhou, Huiyu},  
	title = {SWIPENET: Object detection in noisy underwater scenes},  
	journal = {Pattern Recognition Volume 132},  
	year = {2022}  
} 
```

