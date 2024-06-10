# MultiFG
## 文件说明  
dataset: 数据集文件夹。  
result:用以保存分析结果的文件夹。  
CV10.py：十折交叉验证的代码，会保存10个model和10个valiloader，以及一个模型评价指标的csv文件。  
Cold_CV.py：10折交叉验证评价模型在新药物中的表现代码，会保存10个model_cold和10个valiloader_cold，以及一个模型评价指标的csv文件。  
G_test.py：用以评价模型性能随着训练集药物数量的变化情况，会产生一个模型评价结果csv文件。  
G_test_draw.ipynb：用以可视化G_test.py的结果。  
descriptive_analysis.ipynb：用以探索整个数据集的分布情况。  
四个gin文件：预训练的gin模型，用以获取分子图嵌入特征，来源*dgllife*模块  
vali.ipynb：用以计算模型在验证集上的表现。  
### Dataset  
original data：文本所用数据的原始来源  
data_process.ipynb：处理original data中的文件，产生分析所需要的文件(data6, PPS, NNs_all)  
restore_compressed_files.ipynb：用以恢复我们处理好的文件(data6, PPS, NNs_all)，恢复之后可以直接用以如CV10代码。   
## 连接    
[Predicting the Frequencies of drug side effects](https://github.com/paccanarolab/Side-effect-Frequencies "Predicting the Frequencies of drug side effects")   
[A novel graph attention model for predicting frequencies of drug–side effects from multi-view data](https://github.com/zhc940702/MGPred "A novel graph attention model for predicting frequencies of drug–side effects from multi-view data")   
[DSGAT: predicting frequencies of drug side effects by graph attention networks](https://github.com/xxy45/DSGAT "DSGAT: predicting frequencies of drug side effects by graph attention networks")
