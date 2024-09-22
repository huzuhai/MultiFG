import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gc
import argparse
import random

from scipy import stats
from math import sqrt
from utils.utils_fun import *
from utils.model import *
from sklearn.model_selection import KFold
from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_absolute_error,mean_squared_error
from torch.autograd import Variable



def opti(model, optimizer, criterion1=None, criterion2=None, target1=None, target2=None,a=None,inputs: tuple=()):
    model.train()
    optimizer.zero_grad()

    target1 = target1.reshape(-1,1)
    target2 = target2.reshape(-1,1)
    
    macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2 = inputs
    output1, output2 = model.forward(macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2)
    #output1:binary output2:freq
    if criterion1 is not None and criterion2 is None:
        loss1 = criterion1(output1, target1)
        loss = loss1
    if criterion1 is None and criterion2 is not None:
        try:
            loss2 = criterion2(output2[target1 == 1], target2[target1 == 1])
        except:
            print('The current batch has no known drug adverse reaction pairs.')
            loss = torch.zeros(1, requires_grad=True)
        else:
            loss = loss2
    if criterion1 is not None and criterion2 is not None:
        loss1 = criterion1(output1, target1)
        try:
            loss2 = criterion2(output2[target1 == 1], target2[target1 == 1])
        except:
            print('The current batch has no known drug adverse reaction pairs.')
            loss = loss1
        else:
            loss = a * loss1 + loss2

    loss.requires_grad_()
    loss.backward()
    optimizer.step()
    return loss, output1, output2, target1, target2


def train(model, loader, index, device, optimizer, loss_batch_train, loss_1, loss_2, b1, choose_mode:str = 'all'):
    trainy_pre_b = np.empty((0,1))
    trainy_true_b = np.empty((0,1))
    
    trainy_pre_f = np.empty((0,1))
    trainy_true_f = np.empty((0,1))
    
    batch_n = 0
    for drug, adr, macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2, label_1, label_2 in loader: #lable1_1:binary, lable1_2:freq
        
        batch_n +=1
        if batch_n % 100 == 0:
            print(f"In epoch {index}, batch {batch_n}")
        macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1 = macc_1.to(device), morgan_1.to(device), rtoplo_1.to(device), pharm_1.to(device), contextpred_1.to(device), edgepred_1.to(device), infomax_1.to(device), masking_1.to(device), similar_macc_1.to(device), similar_morgan_1.to(device), similar_rtoplo_1.to(device), similar_pharm_1.to(device), similar_contextpred_1.to(device), similar_edgepred_1.to(device), similar_infomax_1.to(device), similar_masking_1.to(device)

        adr_1, adr1_1, adr1_2, label_1, label_2 = adr_1.to(device), adr1_1.to(device), adr1_2.to(device), label_1.to(device), label_2.to(device)
        adr_1 = Variable(adr_1, requires_grad=False).float()
        adr1_1 = Variable(adr1_1, requires_grad=False).float()
        adr1_2 = Variable(adr1_2, requires_grad=False).float()
        label_1 = Variable(label_1, requires_grad=False).float()
        label_2 = Variable(label_2, requires_grad=False).float()
        
        macc_1 = Variable(macc_1, requires_grad=False).float()
        morgan_1 = Variable(morgan_1, requires_grad=False).float()
        rtoplo_1 = Variable(rtoplo_1, requires_grad=False).float()
        pharm_1 = Variable(pharm_1, requires_grad=False).float()
        
        contextpred_1 = Variable(contextpred_1, requires_grad=False).float()
        edgepred_1 = Variable(edgepred_1, requires_grad=False).float()
        infomax_1 = Variable(infomax_1, requires_grad=False).float()
        masking_1 = Variable(masking_1, requires_grad=False).float()
        
        similar_macc_1 = Variable(similar_macc_1, requires_grad=False).float()
        similar_morgan_1 = Variable(similar_morgan_1, requires_grad=False).float()
        similar_rtoplo_1 = Variable(similar_rtoplo_1, requires_grad=False).float()
        similar_pharm_1 = Variable(similar_pharm_1, requires_grad=False).float()
        similar_contextpred_1 = Variable(similar_contextpred_1, requires_grad=False).float()
        similar_edgepred_1 = Variable(similar_edgepred_1, requires_grad=False).float()
        similar_infomax_1 = Variable(similar_infomax_1, requires_grad=False).float()
        similar_masking_1 = Variable(similar_masking_1, requires_grad=False).float()
        
        inputs = (macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2)

        for name, value in model.named_parameters():
            value.requires_grad=True
        if choose_mode == 'all':
            loss, output_b, output_f, label_1,label_2 = opti(model, optimizer, criterion1=loss_1,criterion2=loss_2, target1=label_1, target2=label_2, a=b1, inputs=inputs)
        # Fine-tuning the binary classification mlpb
        if choose_mode == 'binary':
            # Ensure all parameters' requires_grad attributes are set correctly
            for name, value in model.named_parameters():
                value.requires_grad = False
            for name, value in model.named_parameters():
                if 'com_layerb.combine_layerf' in name:
                    value.requires_grad = True
            loss, output_b, output_f, label_1, label_2 = opti(model, optimizer, criterion1=loss_1, target1=label_1, target2=label_2, a=b1, inputs=inputs)

        # Fine-tuning the frequency prediction mlpf
        if choose_mode == "freq":
            # Ensure all parameters' requires_grad attributes are set correctly
            for name, value in model.named_parameters():
                value.requires_grad = False
            for name, value in model.named_parameters():
                if 'com_layerf.layers' in name:
                    value.requires_grad = True
            loss, output_b, output_f, label_1, label_2 = opti(model, optimizer, criterion2=loss_2, target1=label_1, target2=label_2, a=b1, inputs=inputs)

        # output = torch.softmax(output, dim=1)

        output_b_1 = torch.sigmoid(output_b)
        output_b = output_b_1.cpu().detach().numpy()
        output_f = output_f.cpu().detach().numpy()

        label_1 = label_1.cpu().detach().numpy()
        label_2 = label_2.cpu().detach().numpy()

        trainy_pre_b = np.concatenate((trainy_pre_b, output_b), axis=0)
        trainy_pre_f = np.concatenate((trainy_pre_f, output_f), axis=0)

        trainy_true_b = np.concatenate((trainy_true_b, label_1), axis=0)
        trainy_true_f = np.concatenate((trainy_true_f, label_2), axis=0)

        loss_score_1 = loss.cpu().detach()  # detach separates the data from the computation graph
        loss_batch_train.append(loss_score_1.item())

        del macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2, loss_score_1

        gc.collect()

        #  trainy_pre #np
        #  trainy_true #np
        auc_train = roc_auc_score(trainy_true_b, trainy_pre_b)
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(trainy_true_b, trainy_pre_b)

        aupr_train = auc(recall, precision)

        # Frequency evaluation metrics, only calculate the loss size for positives
        rmse_1 = sqrt(mean_squared_error(trainy_true_f[trainy_true_b == 1], trainy_pre_f[trainy_true_b == 1]))  # scalar
        mae_1 = mean_absolute_error(trainy_true_f[trainy_true_b == 1], trainy_pre_f[trainy_true_b == 1])  # scalar
        spearman_1 = stats.spearmanr(trainy_true_f[trainy_true_b == 1], trainy_pre_f[trainy_true_b == 1])

        print(f"Epoch {index} training set AUC: {auc_train}, AUPR: {aupr_train}, RMSE: {rmse_1}, MAE: {mae_1}, Spearman: {spearman_1}")
        return auc_train, aupr_train, rmse_1, mae_1, spearman_1


def evaluate(model, loader, index, device, loss_epoch_vali, loss_1, loss_2, b1):
    with torch.no_grad():
        model.eval()
        vali_true_f = np.empty((0,1))
        vali_pre_f = np.empty((0,1))
        
        vali_true_b = np.empty((0,1))
        vali_pre_b = np.empty((0,1))
        
        for drug, adr, macc_v, morgan_v, rtoplo_v, pharm_v, contextpred_v, edgepred_v, infomax_v, masking_v, similar_macc_v, similar_morgan_v, similar_rtoplo_v, similar_pharm_v, similar_contextpred_v, similar_edgepred_v, similar_infomax_v, similar_masking_v, adr_v, adr1_v, adr2_v, label_v, labe2_v in loader:     
            
            macc_v, morgan_v, rtoplo_v, pharm_v, contextpred_v, edgepred_v, infomax_v, masking_v, similar_macc_v, similar_morgan_v, similar_rtoplo_v, similar_pharm_v, similar_contextpred_v, similar_edgepred_v, similar_infomax_v, similar_masking_v = macc_v.to(device), morgan_v.to(device), rtoplo_v.to(device), pharm_v.to(device), contextpred_v.to(device), edgepred_v.to(device), infomax_v.to(device), masking_v.to(device), similar_macc_v.to(device), similar_morgan_v.to(device), similar_rtoplo_v.to(device), similar_pharm_v.to(device), similar_contextpred_v.to(device), similar_edgepred_v.to(device), similar_infomax_v.to(device), similar_masking_v.to(device)


            adr_v, adr1_v, adr2_v, label_v, labe2_v = adr_v.to(device), adr1_v.to(device), adr2_v.to(device), label_v.to(device), labe2_v.to(device)

            adr_v = Variable(adr_v, requires_grad=False).float()
            adr1_v = Variable(adr1_v, requires_grad=False).float()
            adr2_v = Variable(adr2_v, requires_grad=False).float()
            label_v = Variable(label_v, requires_grad=False).float()
            labe2_v = Variable(labe2_v, requires_grad=False).float()
            
            macc_v = Variable(macc_v, requires_grad=False).float()
            morgan_v = Variable(morgan_v, requires_grad=False).float()
            rtoplo_v = Variable(rtoplo_v, requires_grad=False).float()
            pharm_v = Variable(pharm_v, requires_grad=False).float()
            
            contextpred_v = Variable(contextpred_v, requires_grad=False).float()
            edgepred_v = Variable(edgepred_v, requires_grad=False).float()
            infomax_v = Variable(infomax_v, requires_grad=False).float()
            masking_v = Variable(masking_v, requires_grad=False).float()
            
            similar_macc_v = Variable(similar_macc_v, requires_grad=False).float()
            similar_morgan_v = Variable(similar_morgan_v, requires_grad=False).float()
            similar_rtoplo_v = Variable(similar_rtoplo_v, requires_grad=False).float()
            similar_pharm_v = Variable(similar_pharm_v, requires_grad = False).float()
            similar_contextpred_v = Variable(similar_contextpred_v, requires_grad=False).float()
            similar_edgepred_v = Variable(similar_edgepred_v, requires_grad=False).float()
            similar_infomax_v = Variable(similar_infomax_v, requires_grad=False).float()
            similar_masking_v = Variable(similar_masking_v, requires_grad = False).float()
            
            
            output_b_vali, output_f_vali = model.forward(macc_v, morgan_v, rtoplo_v, pharm_v, contextpred_v, edgepred_v, infomax_v, masking_v, similar_macc_v, similar_morgan_v, similar_rtoplo_v, similar_pharm_v, similar_contextpred_v, similar_edgepred_v, similar_infomax_v, similar_masking_v, adr_v, adr1_v, adr2_v)

            
            output_b_vali_1 = torch.sigmoid(output_b_vali)
            output_b_vali=output_b_vali_1.cpu().detach().numpy()
            output_f_vali=output_f_vali.cpu().detach().numpy()
        
            label_v = label_v.cpu().detach().numpy()
            labe2_v = labe2_v.cpu().detach().numpy()
            
            vali_pre_b=np.concatenate((vali_pre_b,output_b_vali),axis=0)
            vali_pre_f=np.concatenate((vali_pre_f,output_f_vali),axis=0)
            
            label_v = label_v.reshape(-1,1)
            labe2_v = labe2_v.reshape(-1,1)
            
            vali_true_b=np.concatenate((vali_true_b,label_v),axis=0)
            vali_true_f=np.concatenate((vali_true_f,labe2_v),axis=0)
            
            del macc_v, morgan_v, rtoplo_v, pharm_v, contextpred_v, edgepred_v, infomax_v, masking_v, similar_macc_v, similar_morgan_v, similar_rtoplo_v, similar_pharm_v, similar_contextpred_v, similar_edgepred_v, similar_infomax_v, similar_masking_v, adr_v, adr1_v, adr2_v
            gc.collect()

        _vali_pre_f = torch.from_numpy(vali_pre_f)
        _vali_true_f = torch.from_numpy(vali_true_f)
        _vali_pre_b = torch.from_numpy(vali_pre_b)
        _vali_true_b = torch.from_numpy(vali_true_b)

        
        loss1_v= loss_1(_vali_pre_b, _vali_true_b)
        loss2_v= loss_2(_vali_pre_f[_vali_true_b==1], _vali_true_f[_vali_true_b==1])
        loss_v = b1*loss1_v + loss2_v
        
        loss_epoch_vali.append(loss_v.cpu().detach().item())

        auc_v = roc_auc_score(vali_true_b, vali_pre_b)

        precision_v, recall_v, _ = precision_recall_curve(vali_true_b, vali_pre_b)

        aupr_v = auc(recall_v, precision_v)

        rmse_v = sqrt(mean_squared_error(vali_pre_f[vali_true_b==1], vali_true_f[vali_true_b==1]))
        mae_v = mean_absolute_error(vali_pre_f[vali_true_b==1], vali_true_f[vali_true_b==1])
        spearman_v = stats.spearmanr(vali_pre_f[vali_true_b==1], vali_true_f[vali_true_b==1])

        print(f"Epoch {index} validation set AUC: {auc_v}, AUPR: {aupr_v}, RMSE: {rmse_v}, MAE: {mae_v}, Spearman: {spearman_v}")
        return auc_v,aupr_v,rmse_v, mae_v,spearman_v

if __name__ == "__main__":
    
    parser= argparse.ArgumentParser(prog='MultiFG', description='Predicting drug side effecitve')
    
    # Add arguments with their respective types and default values
    parser.add_argument('-f', '--filename', type=str, default='./dataset/analysis_data.csv', help="File path for input data (default: 'input.csv')") # file path
    parser.add_argument('-b', '--batch', type=int, default=32, help="Batch size (integer, default: 32)") # batch size as an integer
    parser.add_argument('-s', '--feature_size', type=int, default=128, help="Size of features (integer, default: 128)") # feature size as an integer
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help="Learning rate (float, default: 0.001)") # learning rate as a float
    parser.add_argument('-w', '--weight', type=float, nargs='+', default=[0.5, 0.5, 0.5], help="List of weights (float values between 0 and 1, default: [0.5, 0.5, 0.5])") # list of float weights
    parser.add_argument('-e', '--epoch', type=int, nargs='+', default=[10, 20, 30], help="List of epochs (integers, default: [10, 20, 30])") # list of epochs as integers
    
    args = parser.parse_args()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

   # Ensure reproducibility
    seed_value = 3407  # Set the random seed
    np.random.seed(seed_value)
    # random.seed(seed_value)
    # os.environ['PYTHONHASHSEED'] = str(seed_value)  # To prevent hash randomization for reproducible experiments
    torch.manual_seed(seed_value)  # Set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # Set random seed for the current GPU (if only using one GPU)
    torch.cuda.manual_seed_all(seed_value)  # Set random seed for all GPUs (if using multiple GPUs)
    torch.backends.cudnn.deterministic = True

    file_name = args.filename
    data_freq = load_data(file_name)

    # Obtain the positive and negative sample spaces of the drug.
    pps, nns = pps_nns(data_freq)

    # Merge pps and nns datasets and then split into training and validation sets
    data_binary = pd.concat((pps, nns), ignore_index=True)

    del nns, pps
    gc.collect()

    # Merge frequency data using drug-ADR pairs as the identifier, filling missing values with 0
    data_binary1 = pd.merge(data_binary, data_freq[['drugname', 'adr', 'freq']], 'left', on=['drugname', 'adr'])
    data_binary1.freq.fillna(0, inplace=True)

    # Create a drug dataset for calculating various features for each drug
    drug_smiles = data_freq[['drugname', 'smiles', 'Macc', 'pubchem', 'Morgan', 'Rtoplo', 'pharmfp']]
    drug_smiles1 = drug_smiles.drop_duplicates().copy()

    # Create an ADR dataset for calculations
    _adr = data_freq[['adr', 'adr_similar']]
    _adr1 = _adr.drop_duplicates().copy()

    del data_freq, drug_smiles, data_binary, _adr
    gc.collect()

    
    _data_binary = data_binary1.merge(drug_smiles1[['drugname', 'smiles']], 'left', on='drugname')
    _data_binary = _data_binary.merge(_adr1, 'left', on='adr')


    contextpred = pre_molecule(drug_smiles1['smiles'].tolist(), device,name='gin_supervised_contextpred').tolist()
    edgepred = pre_molecule(drug_smiles1['smiles'].tolist(), device,name='gin_supervised_edgepred').tolist()
    infomax = pre_molecule(drug_smiles1['smiles'].tolist(), device, name='gin_supervised_infomax').tolist()
    masking = pre_molecule(drug_smiles1['smiles'].tolist(), device, name='gin_supervised_masking').tolist()
    drug_smiles1.loc[:, 'contextpred'] = contextpred
    drug_smiles1.loc[:, 'edgepred'] = edgepred
    drug_smiles1.loc[:, 'infomax'] = infomax
    drug_smiles1.loc[:, 'masking'] = masking

    del data_binary1, contextpred, edgepred, infomax, masking


    data_binary2 = pd.merge(_data_binary, drug_smiles1[['drugname',  'Macc', 'pubchem','Morgan', 'Rtoplo', 'pharmfp', 'contextpred', 'edgepred', 'infomax', 'masking']], how='left', on='drugname')
    del _data_binary, drug_smiles1
    gc.collect()
    kf = KFold(n_splits=10, shuffle=True, random_state=3407)
    metrics_res = {"dataset": [], "folder":[],"best_epoch": [],"auc_max": [],"aupr_maxauc": [],"rmse_maxauc": [],"mae_maxauc":[],"spearman_maxauc":[], "epoch_maxaupr":[],"aupr_max":[],"auc_maxaupr":[],"rmse_maxaupr":[],"mae_maxaupr":[],"spearman_maxaupr":[],"epoch_minrmse":[], "rmse_min":[],"mae_minrmse":[],"auc_minrmse":[],"aupr_minrmse":[],"spearman_minrmse":[]}
    
    #CV 10
    for idx, (train_index, vali_index) in enumerate(kf.split(data_binary2)):
        
        train_data = data_binary2.iloc[train_index, :].reset_index(drop=True)
        vali_data = data_binary2.iloc[vali_index, :].reset_index(drop=True)
        
        run(train_data, vali_data, idx, device, train, evaluate, metrics_res, args)
    # #Drug cold starting
    # drug_data = data_binary2.loc[:,['drugname']].drop_duplicates().reset_index(drop=True)
    # for idx, (train_index, vali_index) in enumerate(kf.split(drug_data)):
    #     train_drug = drug_data.iloc[train_index, :].reset_index(drop=True)
    #     vali_drug = drug_data.iloc[vali_index, :].reset_index(drop=True)
        
    #     drug_train_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(train_drug['drugname'])].index])
    #     drug_vali_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(vali_drug['drugname'])].index])

    #     print(f"Training set drugs:{train_drug.shape[0]}")
    #     print(f"Validation set drugs:{vali_drug.shape[0]}")
        
    #     train_data = data_binary2.iloc[drug_train_rows, :].reset_index(drop=True)
    #     vali_data = data_binary2.iloc[drug_vali_rows, :].reset_index(drop=True)
    #     run(train_data, vali_data, idx, device, train, evaluate, metrics_res, args)
        
    #gradually    
    # drug_data = data_binary2.loc[:,['drugname']].drop_duplicates().reset_index(drop=True)
    # drug_data.loc[:, 'g'] = drug_data.apply(lambda x: random.randint(0,9), axis=1)
    # for idx, gg in enumerate(range(9)):
    #     vali_drug = drug_data[drug_data['g'] == 9]
    #     number_vali = vali_drug.shape[0]
    #     train_drug = drug_data[drug_data['g'] <= gg]
    #     number_train = train_drug.shape[0]
    #     drug_train_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(train_drug['drugname'])].index])
    #     drug_vali_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(vali_drug['drugname'])].index])

    #     print(f"Training set drugs:{train_drug.shape[0]}")
    #     print(f"Validation set drugs:{vali_drug.shape[0]}")

    #     train_data = data_binary2.iloc[drug_train_rows, :].reset_index(drop=True)
    #     vali_data = data_binary2.iloc[drug_vali_rows, :].reset_index(drop=True)
    #     run(train_data, vali_data, idx, device, train, evaluate, metrics_res, args)
        
    pd.DataFrame(metrics_res).to_csv("./result/finally.csv")
    del data_binary2
    gc.collect()