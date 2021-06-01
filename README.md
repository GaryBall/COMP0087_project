# COMP0087 group project

This is the repo for the project "Coherence Improvement on Pre-Training Based Cooking Recipe Generation", by group 17053487. 

## How to start?

The report contains all the detail of our project. After reading through that and watching the video, the main project file should be the first point to understand our code structure.

The **easy way** to reproduce our result is by running *evaluation_final.ipynb* in the directory *./COMP0087_project/exp_nbs/evaluations/*. For that, you need to import the datafile *350k_GPT-2_Score.csv*. After running the evaluation notebooks, you can use your own input to test our recipe generation in the evaluation notebook. An example code is here:

```
# your input
raw_text =  'potato, beef;'

model = BartForConditionalGeneration.from_pretrained("jky594176/BART2_GRU")
tokenizer = init_tokenizer("facebook/bart-base")
model = model.to('cuda')
md = generate_recipe(raw_text)
get_instr(md)
```


## What's in this repo?
1. Main project notebook
  - *prj_main_v6.ipynb*
  - The main file is an example of our experiment, modified from the experiment of BART1+GRU
2. Project report
  - *report_group_17053487.pdf*
  - The final report of our project
3. Project presentation slides
  - *G17053487_slides.pptx*
  - Our presentation slides. (The video's resolution is not high. So if you feel uncomfortable while watching, you can use this!)
4. Project presentation videos
  - *G17053487_pre.mp4*
  -  The video of our presentation
5. All the experiments notebooks in './COMP0087_project/exp_nbs/'
  - **GPT2**: *GPT2_0510_v2.ipynb*
  - **BART1**: *BART1_vanilla_0521v3.ipynb*
  - **BART2**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+MLP**: *BART2_vanilla_0522v2.ipynb*
  - **BART1+GRU**: *prj_main_v6.ipynb*
  - **BART2+GRU**: *BART2_GRU_0528_loss.ipynb*
6. helper functions python file in './COMP0087_project/exp_nbs/helper_functions'
  - Help functions for data preprocessing is in *data_preprocess.py*
  - Batch processing functions are saved in *batch_process.py*. This file mainly contains our processing during training procedure. 
  - The codes for differnt models structures can be found in *model_setup.py*. 
  - The codes of utility functions for various models (e.g. model initialization, model saving) can be found in *model_utils.py*
  - The training codes for different models are in *model_train.py*
  

7. The evaluation files.
  - We also create a separate evaluation file for your reference. This is the **easy way** to test our system. Please refer to the data directory *./COMP0087_project/exp_nbs/evaluations/* and find file *evaluation_final.ipynb* to reproduce our experiment results if necessary
  - All the evaluations are based on the dataset *350k_GPT-2_Score.csv*.
  -  We upload all our trained model to huggingface (https://huggingface.co/jky594176). We also create a separate evaluation file for your reference. Please refer to the *.ipynb* file to evaluate our model. Using that, you can reproduce our experiment results.
