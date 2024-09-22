# MultiFG
We propose a new model called MultiFG to predict the associations and frequency information of drug side effects. MultiFG utilizes various fingerprint features and molecular graph embeddings to learn the complex relationships between drug-side effect pairs through a multi-head attention mechanism and convolutional neural networks.
## Requirements  
**python version: 3.7.1**   
**rdkit version: 2023.03.2**  
**torch version: 1.13.1+cpu**  
**dgl version: 2.0.0**  
**numpy version: 1.21.5**  
**pandas version: 1.3.5**  
**pubchempy version: 1.0.4**  
**dgllife version: 0.3.2**   
## File Description
1.dataset: Folder containing datasets.
 * original_data: Dataset compiled from previous research.
 * data_process.ipynb: Code for processing and merging datasets from previous research.
 * adr.csv.gz, drug.csv.gz: Our compiled datasets.
 * restore_compressed_files.ipynb: Code to restore compressed files to model data.
2.pre-trained graph model: Pre-trained graph models.
3.result: Folder for saving analysis results.
4.utils, utils_data: Various utility functions and tools.
5.run.py: Main code for model training, evaluation, etc.
6.Four gin files: Pre-trained GIN models used to obtain molecular graph embedding features, sourced from the dgllife module.
## Run
filename: Input file.
batch: Set batch size.
feature_size: Mapping dimension for features.
learning_rate: Learning rate.
weight: Weight for the loss of association and frequency prediction.
epoch: Number of epochs.

Example:
```bash
python run.py -f data.csv -b 32 -s 128 -l 0.001 -w 0.5 0.3 0.7 -e 10 20 30
```
## Links    
[Predicting the Frequencies of drug side effects](https://github.com/paccanarolab/Side-effect-Frequencies "Predicting the Frequencies of drug side effects")   
[A novel graph attention model for predicting frequencies of drug–side effects from multi-view data](https://github.com/zhc940702/MGPred "A novel graph attention model for predicting frequencies of drug–side effects from multi-view data")   
[DSGAT: predicting frequencies of drug side effects by graph attention networks](https://github.com/xxy45/DSGAT "DSGAT: predicting frequencies of drug side effects by graph attention networks")  
[An Efficient Implementation of Kolmogorov-Arnold Network](https://github.com/Blealtan/efficient-kan "An Efficient Implementation of Kolmogorov-Arnold Network")  
## Contact
If you have any questions or suggestions with this work, please let us know. Contact *Zuhai hu* at 2022120774@stu.cqmu.edu.cn
