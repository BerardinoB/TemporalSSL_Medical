# Self-Supervised Model for Temporal Prediction in Medical Imaging
**Note:** this is a preliminary work that will be completed if data become available for scientific publication

This project is based on the idea proposed in [Misra et al. (2016)](https://arxiv.org/pdf/1603.08561.pdf). In a nutshell, a longitudinal dataset on Multiple Sclerosis (MS) patients was used to predict the correct ordering of the evolution of the disease based on 3D MRI data. Five different randomized control trial (RCT) studies were used where at t<sub>0</sub> no patient received treatment and for all subsequent time points a treatment $s \in S = \{1,2,..k\}$ is administered to each subject. The aim of the project is to detect if the temporal ordering of the 3D MRI images is realistic!!! 
A schematic representation of the model is proposed below:

![alt text](https://github.com/BerardinoB/TemporalSSL_Medical/blob/main/Images/Diagram_Model.png)

