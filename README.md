# COMP0087 group project

This is the repo for the project "Coherence Improvement on Pre-Training Based Cooking Recipe Generation", by group 17053487. 

## How to start?

The report contains all the detail of our project. After reading through that and watching the video, the main project file should be the first point to understand our code structure. Then if the evaluation results need to be reproduced, please use the evaluation files *evaluation_final.ipynb*


## What's in this github?
1. Main project notebook
  - The main file is an example of our experiment, modified from the experiment of BART1+GRU
2. Project report
  - The main file is an example of our experiment, modified from the experiment of BART1+GRU
3. Project presentation slides
  - The main file is an example of our experiment, modified from the experiment of BART1+GRU
4. Project presentation videos
  - The main file is an example of our experiment, modified from the experiment of BART1+GRU
5. All the experiments notebooks in './COMP0087_project/exp_nbs/'
  - **GPT2**: *GPT2_0510_v2.ipynb*
  - **BART1**: *BART1_vanilla_0521v3.ipynb*
  - **BART2**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+MLP**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+GRU**: This file
  - **BART2+GRU**: *BART2_GRU_0528_loss.ipynb*
6. helper functions python file in './COMP0087_project/exp_nbs/helper_functions'
  - Help functions for data preprocessing is in *data_preprocess.py*
  - Batch processing functions are saved in *batch_process.py*. This file mainly contains our processing during training procedure. 
  - The codes for differnt models structures can be found in *model_setup.py*. 
  - The codes of utility functions for various models (e.g. model initialization, model saving) can be found in *model_utils.py*
  - The training codes for different models are in *model_train.py*
  

7. The evaluation files The fine-tuned model files.
  - We also create a separate evaluation file for your reference. Please refer to the *evaluation_final.ipynb* file to reproduce our experiment results if necessary
  -  We upload all our trained model to huggingface (https://huggingface.co/jky594176). We also create a separate evaluation file for your reference. Please refer to the *.ipynb* file to evaluate our model. Using that, you can reproduce our experiment results.

Also, we uploaded everything to a github () in case something is missing. You can find the all necessary files there. 
