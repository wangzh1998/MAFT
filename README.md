# 0 Introduction
This artifact includes source code, datasets and models of the paper "MAFT: Efficient Model-Agnostic Fairness Testing for Deep Neural 
Networks via Zero-Order Gradient Search".


# 1 Getting Started

1.1 Environment

Python3.6 + Tensorflow2.0 or later versions are recommended.

You can do the following to create a conda environment named maft_env and activate it:

[//]: # (conda install -c conda-forge tensorflow)

[//]: # (conda install -c asmeurer z3)
[//]: # (conda install -c newptcai z3-solver)
[//]: # (for SG:)

[//]: # ()
[//]: # (    pip install z3-solver # ignore this if you have installed z3-solver by environment.yml)

[//]: # ()
[//]: # (for meps dataset:)

[//]: # ()
[//]: # (    brew install r # install R language)

[//]: # (    )
[//]: # (    conda install -c conda-forge aif360 # install AIF360 package # ignore this if you have installed aif360 by environment.yml)

[//]: # (    )
[//]: # (    cd xx/anaconda3/envs/maft_env/lib/python3.7/site-packages/aif360/data/raw/meps # to find generat_data.R, please replace xx with your own path where you install anaconda)

[//]: # (    )
[//]: # (    Rscript generate_data.R # generate meps dataset)

[//]: # ()
[//]: # (The following packages are also required: numpy, scikit-learn, pandas, joblib.)

~~~
conda env create -f environment.yml # create environment by environment.yml named maft_env default
~~~

~~~
conda activate maft_env # activate maft_env
~~~

To use MEPS15 dataset, you need to install R language and AIF360 package.
~~~
brew install r # install R language on MacOs
~~~

~~~
sudo apt-get update
sudo apt-get install r-base # install R language on Linux
~~~

~~~
cd xx/anaconda3/envs/maft_env/lib/python3.7/site-packages/aif360/data/raw/meps # to find generat_data.R, please replace xx with your own path where you install anaconda
~~~

~~~
Rscript generate_data.R # generate meps dataset
~~~


1.2 Small Examples

We provide two small examples to validate the general functionality by generating some individual discriminatory instances 
with AEQUITAS, SG, ADF, EIDIG, MAFT respectively.

1. compare white-box fairness testing methods:
   ~~~
   python test.py
   ~~~
   
   which is equivalent to:

   ~~~
   python test.py --round_id 1 --g_num 20 --l_num 20 --perturbation_size 1.0 --should_restore_progress True
   ~~~
   
   You will get statistical results in the terminal and have discriminatory instances saved to 'logging_data/complete_comparison/complete_comparison_instances/'.
2. compare black-box fairness testing methods:
   ~~~
   python test_bb.py
   ~~~
   which is equivalent to:
   ~~~ 
   python test_bb.py --round_id 1 --g_num 20 --l_num 20 --perturbation_size 1.0 --should_restore_progress True
   ~~~
   You will get statistical results in the terminal and have discriminatory instances saved to 'logging_data/complete_comparison/complete_comparison_instances_bb/'.


### 1.3 Detailed Usage
Run the following command in the terminal to execute the script:
~~~
python test.py --round_id <round_identifier> --g_num <global_seed_number> --l_num <local_seed_number> --perturbation_size <perturbation_parameter> --should_restore_progress <restore_progress_flag>
~~~
~~~
python test_bb.py --round_id <round_identifier> --g_num <global_seed_number> --l_num <local_seed_number> --perturbation_size <perturbation_parameter> --should_restore_progress <restore_progress_flag>
~~~

#### Parameter Descriptions

- round_id: Identifier for the round, can be freely set, typically starting from 1. New data will overwrite previous data with same round id.

- g_num: Global seed number, used to set the seed for the global random number generator.

- l_num: Local seed number, used to set the seed for the local random number generator.

- perturbation_size: Perturbation parameter for the MAFT method, with a default optimal setting of 1.0.

- should_restore_progress: Indicates whether to resume execution from a previous interruption. Possible values are yes or no.

#### Example
You can refer to 1.2 for small examples.

#### Notes
Ensure that the Python environment is installed before running the command, and the script's required libraries are installed.