# MultiFG
## File Description  
dataset: Folder containing datasets.  
result: Folder for saving analysis results.  
CV10.py: Code for 10-fold cross-validation, which will save 10 models and 10 valiloaders, along with a CSV file containing model evaluation metrics.  
Cold_CV.py: Code for evaluating model performance on new drugs using 10-fold cross-validation, which will save 10 model_cold and 10 valiloader_cold, along with a CSV file containing model evaluation metrics.  
G_test.py: Used to evaluate model performance as the number of drugs in the training set changes, resulting in a CSV file of model evaluation results.  
G_test_draw.ipynb: Used to visualize the results from G_test.py.  
descriptive_analysis.ipynb: Used to explore the distribution of the entire dataset.  
Four gin files: Pre-trained gin models used to obtain molecular graph embedding features, sourced from the *dgllife* module.  
vali.ipynb: Used to evaluate model performance on the validation set.  
### Dataset  
original data: The original source of data used in the text.  
data_process.ipynb: Processes files in the original data to generate files required for analysis (data6, PPS, NNs_all).  
restore_compressed_files.ipynb: Used to restore our processed files (data6, PPS, NNs_all), which can be directly used in codes like CV10 after restoration.   
## Links    
[Predicting the Frequencies of drug side effects](https://github.com/paccanarolab/Side-effect-Frequencies "Predicting the Frequencies of drug side effects")   
[A novel graph attention model for predicting frequencies of drug–side effects from multi-view data](https://github.com/zhc940702/MGPred "A novel graph attention model for predicting frequencies of drug–side effects from multi-view data")   
[DSGAT: predicting frequencies of drug side effects by graph attention networks](https://github.com/xxy45/DSGAT "DSGAT: predicting frequencies of drug side effects by graph attention networks")
