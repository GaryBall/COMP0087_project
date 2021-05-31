# COMP0087 group project

This is the repo for the project "Coherence Improvement on Pre-Training Based Cooking Recipe Generation", by group 17053487. 

## What's in this github?
1. Main project notebook
   - 
- Project report
- Project presentation slides
- Project presentation videos
- All the experiments notebooks
- helper functions python file


The main file is just an example of our experiment, which means the final results are not complete. Specifically, this file does not contain,

1. The dataset *full_dataset.csv*. We use the "RecipeNLG" Dataset, to download the full dataset, please access: https://recipenlg.cs.put.poznan.pl/ . (Citation is made in the report)

2. Some of the helper functions. Since our helper functions are designed with respect to various model structures, this notebook only contains helper functions for BART1+GRU model. Helper functions are saved in './COMP0087_project/exp_nbs/helper_functions'
  - Help functions for data preprocessing is in *data_preprocess.py*
  - Batch processing functions are saved in *batch_process.py*. This file mainly contains our processing during training procedure. 

3. Codes for all the models. In our report, we proposed a novel model. But to test how different modules in this model influence the overall performance, we conducted experiments based on various model structures. We have an example code for BART1 with GRU in this notebook, but you can find all other models' codes either by looking into the independent *.py* files or previous experiment notebooks in our Github. The python files are oraganized as follows,
  - The codes for differnt models structures can be found in *model_setup.py*. 
  - The codes of utility functions for various models (e.g. model initialization, model saving) can be found in *model_utils.py*
  - The training codes for different models are in *model_train.py*

4. All the experiment notebooks. To find those, you can look into the directory './COMP0087_project/exp_nbs/'. We list the experiment list and their corresponding notebook,
  - **GPT2**: 
  - **BART1**: *BART1_vanilla_0521v3.ipynb*
  - **BART2**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+MLP**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+GRU**: This file
  - **BART2+GRU**: *BART2_GRU_0528_loss.ipynb*
  
  

5. The fine-tuned model files. The models are complex thus contatin many parameters. So we decided not to upload it on Moodle. We upload all our trained model to huggingface (https://huggingface.co/jky594176). We also create a separate evaluation file for your reference. Please refer to the *.ipynb* file to evaluate our model. Using that, you can reproduce our experiment results.

Also, we uploaded everything to a github () in case something is missing. You can find the all necessary files there. 
