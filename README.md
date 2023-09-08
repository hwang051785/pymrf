# pymrf
Stochastic geological modeling using Markov Random Field and Bayesian Machine Learning
## Contents


## Introduction
This package presents a novel stratigraphic stochastic simulation approach, which is developed by integrating a Markov random field (MRF) model and a discriminant adaptive nearest neighbor-based k-harmonic mean distance (DANN-KHMD) classifier into a Bayesian framework. The DANN-KHMD classifier is effective for extracting anisotropic patterns from sparse and heterogeneous spatial categorical data such as borehole logs. The MRF parameters can be initially estimated roughly or customized (if site-specific knowledge is available). Later these parameters can be updated and regularized in an unsupervised manner with constraints from site exploration results in a Bayesian manner. Throughout the learning process, both the soil profile and the MRF parameters are updated in a probabilistic manner. The advantages of the proposed approach can be summarized into four points: 1) inferring stratigraphic profile and associated uncertainty in an automatic and fully unsupervised manner; 2) reasonable initial stratigraphic configurations can be sampled and hence lower the computational cost; 3) both stratigraphic uncertainty and model uncertainty are taken into consideration throughout the inferential process; 4) relying on no training stratigraphy images. 

## Example case
You can try out this example by using an interactive Jupyter Notebook in your own web browser.

### Main code
The file "pyMRF.py" is the main code of the program.

### case dataset
The file "case_dataset.npy" is the data of the example case, which is generated using the file "generate_MRF_realizations.py". 
The file "generate_MRF_realizations.py" is the code for generating an MRF given a predetermined beta vector.

### The inference process
The file "Stratigraphic configuration acquisition process of synthetic case.ipynb" is the inference process.

## Reference
Wei, X., & Wang, H. (2022). Stochastic stratigraphic modeling using Bayesian machine learning. Engineering Geology, 307, 106789. doi: https://doi.org/10.1016/j.enggeo.2022.106789
