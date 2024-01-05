# 0 Introduction
This artifact includes source code, datasets and models of the paper "MAFT: Efficient Model-Agnostic Fairness Testing for Deep Neural 
Networks via Zero-Order Gradient Search".


# 1 Getting Started

1.1 Environment

Python3.6 + Tensorflow2.0 or later versions are recommended.

[//]: # (conda install -c asmeurer z3)
[//]: # (conda install -c newptcai z3-solver)
for SG:

    pip install z3-solver

for meps dataset:

    brew install r # install R language
    
    conda install -c conda-forge aif360 # install AIF360 package
    
    cd xx/anaconda3/envs/tf2/lib/python3.7/site-packages/aif360/data/raw/meps # find generat_data.R
    
    Rscript generate_data.R # generate meps dataset

The following packages are also required: numpy, scikit-learn, pandas, joblib.


1.2 Small Examples

We provide two small examples to validate the general functionality by generating some individual discriminatory instances 
with AEQUITAS, SG, ADF, EIDIG, MAFT respectively.

1. compare white-box fairness testing methods:

        python test.py
2. compare black-box fairness testing methods:

        python test_bb.py

You will get statistical results in the terminal and have discriminatory instances saved to 'logging_data/complete_comparison/complete_comparison_instances/' or 
'logging_data/complete_comparison/complete_comparison_instances_bb/'.