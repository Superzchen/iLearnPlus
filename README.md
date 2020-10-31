# *iLearnPlus*: a comprehensive and automated machine-learning platform for nucleic acid and protein sequence analysis, prediction and visualization. 
*iLearnPlus* is the first machine-learning platform with both graphical- and web-based user interface that enables the construction of automated machine-learning pipelines for computational analysis and predictions using nucleic acid and protein sequences. *iLearnPlus* integrates 21 machine-learning algorithms (including 12 conventional classification algorithms, two ensemble-learning frameworks and seven deep-learning approaches) and 19 major sequence encoding schemes (in total 152 feature descriptors), outnumbering all the current web servers and stand-alone tools for biological sequence analysis, to the best of our knowledge. In addition, the friendly GUI (Graphical User Interface) of *iLearnPlus* is available to biologists to conduct their analyses smoothly, significantly increasing the effectiveness and user experience compared to the existing pipelines. iLearnPlus is an open-source platform for academic purposes and is available at https://github.com/Superzchen/iLearnPlus/. The iLearnPlus-Basic module is online accessible at http://ilearnplus.erc.monash.edu/.

# Methods
Four major modules, including *iLearnPlus-Basic*, *iLearnPlus-Estimator*, *iLearnPlus-AutoML*, and *iLearnPlus-LoadModel*, are provided in *iLearnPlus* for biologists and bioinformaticians to conduct customizable sequence-based feature engineering and analysis, machine-learning algorithm construction, performance assessment, statistical analysis, and data visualization, without additional programming.![iLearnPlus](https://github.com/Superzchen/iLearnPlus/blob/main/images/Architecture.png )

# Installation

  - Download *iLearnPlus* by 
  ```sh
  git clone https://github.com/Superzchen/iLearnPlus
  ```
  *iLearnPlus* is an open-source Python-based toolkit, which operates in the Python environment (Python version 3.6 or above) and can run on multiple operating systems (such as Windows, Mac and Linux). Prior to installing and running *iLearnPlus*, all the dependencies should be installed in the Python environment, including sys, os, re, PyQt5, qdarkstyle, numpy (1.18.5), pandas (1.0.5), threading, sip, datetime, platform, pickle, copy, scikit-learn (0.23.1), math, scipy (1.5.0), collections, itertools, torch (≥1.3.1), lightgbm (2.3.1), xgboost (1.0.2), matplotlib (3.1.1), seaborn,  joblib, warnings, random, multiprocessing and time. **For convenience, we strongly recommended users to install the Anaconda Python environment in your local computer. The software can be freely downloaded from https://www.anaconda.com/.**

  - Step 1. Download and install the anaconda platform.
  ```sh  
  Download from: https://www.anaconda.com/products/individual
  ```
  
  - Step 2. Install PyTorch:
  ```sh  
  Please refer to https://pytorch.org/get-started/locally/ for PyTorch installation.
  ```
  
  - Step 3. Install other necessary packages, by the following commands:
  ```sh
  pip3 install lightgbm
  pip3 install xgboost
  pip3 install qdarkstyle
  ...
  ```



# Guidance to Use :
cd to the *iLearnPlus* folder which contains iLearnPlus.py and run the ‘iLearnPlus.py’ script as follows:
```sh
python iLearnPlus.py
```
  
Please refer to [*iLearnPlus* manual](https://github.com/Superzchen/iLearnPlus/blob/main/docs/iLearnPlus_manual.pdf ) for detailed usage.
  
## *iLearnPlus* interfaces:

*iLearnPlus* main interface:
  
![iLearnPlus](https://github.com/Superzchen/iLearnPlus/blob/main/images/iLearnPlus.png )

*iLearnPlus-Basic* module interface:
  
The *iLearnPlus-Basic* module facilities analysis and prediction using a selected feature-based representation of the input protein/RNA/DNA sequences (sequence descriptors) and a selected machine-learning classifier. This module is particularly instrumental when interrogating the impact of using different sequence feature descriptors and machine-learning algorithms on the predictive performance.
  
![Basic module](https://github.com/Superzchen/iLearnPlus/blob/main/images/Basic.png )

*iLearnPlus-Estimator* module interface:
  
The *iLearnPlus-Estimator* module provides a flexible way to perform feature extraction by allowing users to select multiple feature descriptors. 
  
![Estimator module](https://github.com/Superzchen/iLearnPlus/blob/main/images/Estimator.png )

*iLearnPlus-AutoML* module interface:
  
The *iLearnPlus-AutoML* module focuses on automated benchmarking and maximization of the predictive performance across different machine-learning classifiers that are applied on the same set or combined sets of feature descriptors.
  
![AutoML module](https://github.com/Superzchen/iLearnPlus/blob/main/images/AutoML.png )

*iLearnPlus-LoadModel* module interface:
  
The *iLearnPlus-LoadModel* module allows users to upload, deploy and test their models.
  
![LoadModel module](https://github.com/Superzchen/iLearnPlus/blob/main/images/LoadModel.png )

*iLearnPlus* Data visulaization:
  
![Data visualizaiton](https://github.com/Superzchen/iLearnPlus/blob/main/images/DataVisualization.png)

# Case study results for lysine crotonylation site prediction:
Here, the *iLearnPlus-Estimator* module was used to comparatively assess the performance of different feature sets. We used the *iLearnPlus-Estimator* module in the standalone GUI version to load the data, produced seven feature sets (AAC, EAAC, EGAAC, DDE, binary, ZScale, and BLOSUM) and selected a machine-learning algorithm, the random forest algorithm (with the default setting of 1000 trees) to construct the classifier via 10-fold cross-validation. This analysis reveals that the model built utilizing the EGAAC feature descriptors achieved the best performance. 
  
![Case study](https://github.com/Superzchen/iLearnPlus/blob/main/images/Case_1.png)
  
Then, the *iLearnPlus-AutoML* module to comparatively evaluate the predictive performance across seven machine-learning algorithms: SGD, LR, XGBoost, LightGBM, RF, MLP, and CNN. We used the bootstrap tests to assess statistical significance of the differences between the ROC curves produced by these algorithms.
  
![Case study](https://github.com/Superzchen/iLearnPlus/blob/main/images/Case_2.png)
  
The result shows that the deep-learning model, CNN, achieved the best predictive performance among all the seven machine-learning algorithms, with Acc=85.4% and AUC=0.823.
  
*iLearnPlus* makes it easy and straightforward to design and optimize machine-learning pipelines to achieve a competitive (if not the best) predictive performance. 
