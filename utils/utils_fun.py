import pandas as pd
import dgl
import numpy as np
import pandas as pd
import torch
import sys
import gc
import time
import ast
import itertools
import time
import dill
import copy
import torch.nn.init as init
import torch.optim as optim

from joblib import Parallel, delayed
from rdkit import Chem
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model.pretrain.moleculenet import *
from dgllife.model.pretrain.generative_models import *
from dgllife.model.pretrain.property_prediction import *
from dgllife.model.pretrain.reaction import *
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from torch.utils.data import DataLoader
from rdkit import Chem
from model import *



#load dataset
def load_data(file_name):
    if not file_name.lower().endswith('.csv'):
        raise ValueError("Invalid file format. Please provide a CSV file.")
    try:
        data2 = pd.read_csv(file_name)
    except Exception as e:
        raise ValueError(f"Error reading the file: {e}")
    data2.columns = ['drugname','adr','freq','smiles','adr_similar', 'Macc','pubchem','Morgan','Rtoplo','pharmfp']
    try:
        data2['freq'] = data2['freq'].astype(int)
    except ValueError:
        raise ValueError("Column 'freq' contains non-integer values.")
    return data2

#Obtain the positive and negative sample spaces of the drug.
def pps_nns(data):
    PPS = data.copy()
    PPS['association'] = 1
    adr_all = data.SideeffectTerm.unique()
    all_drug_adr_paire = pd.DataFrame((list(itertools.product(data.GenericName.unique(), adr_all))))
    all_drug_adr_paire.rename(columns={0 : 'GenericName',
                                   1: 'SideeffectTerm'}, inplace=True)
    filtered_drug_adr_pair = all_drug_adr_paire.merge(PPS, on=['GenericName', 'SideeffectTerm'], how='left', indicator=True)
    NNS_all = filtered_drug_adr_pair[filtered_drug_adr_pair['_merge'] == 'left_only']
    del filtered_drug_adr_pair
    NNS_all['association'] = 0
    NNS_all.drop(columns='_merge', axis=0, inplace=True)
    PPS = PPS.rename(columns={'GenericName': 'drugname', 'SideeffectTerm': 'adr'})
    NNS_all = NNS_all.rename(columns={'GenericName': 'drugname', 'SideeffectTerm': 'adr'})
    return PPS, NNS_all

# create the graph data to extract graph embedding feature
def graph_construction_and_featurization(smiles):
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    return graphs, success

def collate(graphs):
    return dgl.batch(graphs)


def create_model(model_name):
  for func in [create_moleculenet_model, create_generative_model,
        create_property_model, create_reaction_model]:
    model = func(model_name)
    if model is not None:
      return model
  
def main(dataset, device,name='gin_supervised_contextpred'):
    data_loader = DataLoader(dataset, batch_size=128,
                                collate_fn=collate, shuffle=False)
    model = create_model(name)
    checkpoint = torch.load('pre-trained graph model/'+name+ '_pre_trained.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    # model.eval()
    readout = AvgPooling()
    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        # print('Processing batch {:d}/{:d}'.format(batch_id + 1, len(data_loader)))
        nfeats = [bg.ndata.pop('atomic_number').to(device),
                  bg.ndata.pop('chirality_type').to(device)]
        efeats = [bg.edata.pop('bond_type').to(device),
                  bg.edata.pop('bond_direction_type').to(device)]
        with torch.no_grad():
            bg=bg.to(device)
            node_repr = model(bg, nfeats, efeats)

        mol_emb.append(readout(bg, node_repr))
    # print(len(mol_emb))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    return np.array(mol_emb,dtype=np.float32)


def pre_molecule(smiles_list, device,name='gin_supervised_contextpred'):
    dataset, success = graph_construction_and_featurization(smiles=smiles_list)
    drug_molecule = main(dataset, device,name)
    return drug_molecule

# jaccord similarity
def calculate_tanimoto_similarity_matrix(list1, list2):  # list1 is an fp, and list2 is a list of fps
    # Remove sublists from list2 where all elements are 0
    index = [not all(np.array(i) == 0) for i in list2]
    list2 = list(np.array(list2)[index])
    
    if all(np.array(list1) == 0):  # If all elements of list1 are 0, return a zero vector directly
        return np.zeros(len(list2))
    
    # Expand list1 to a matrix with the same shape as list2
    list1_matrix = np.tile(list1, (len(list2), 1))
    
    # Convert list2 to a numpy array
    list2_matrix = np.array(list2)
    
    # Calculate Tanimoto similarity
    intersection_count = np.sum(list1_matrix & list2_matrix, axis=1)
    union_count = np.sum(list1_matrix | list2_matrix, axis=1)
    
    # Avoid division by zero, i.e., if all features of a drug/adr are 0, set the similarity to 0
    similarity = intersection_count / union_count
    
    return similarity  # Returns a numpy array

# cosine similarity
def calculate_cos_sim(fp, fp_list):
    # Remove sublists from fp_list where all elements are 0
    index = [not all(np.array(i) == 0) for i in fp_list]
    fp_list = list(np.array(fp_list)[index])
    
    if all(np.array(fp) == 0):  # If all elements of fp are 0, return a zero vector
        return np.zeros(len(fp_list))
    
    # Repeat fp for each vector in fp_list
    _fp = [fp for i in range(len(fp_list))]
    
    # Calculate the dot product
    _dot = [np.dot(i, j) for i, j in zip(_fp, fp_list)]
    
    # Calculate the norms (L2 norm)
    _norm1 = [np.linalg.norm(i) for i in _fp]
    _norm2 = [np.linalg.norm(i) for i in fp_list]
    
    # Handle cases where the norm is 0 by replacing the norm with 1 (to avoid division by zero)
    _norm1 = [1 if element == 0 else element for element in _norm1]
    _norm2 = [1 if element == 0 else element for element in _norm2]
    
    # Calculate the cosine similarity
    _div = [i * j for i, j in zip(_norm1, _norm2)]
    _result = [i / j for i, j in zip(_dot, _div)]
    
    return np.array(_result)

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, drug, adrname, macc, morgan, rtoplo, pharm,   contextpred, edgepred, infomax, masking, macc_similar, morgan_similar, rtoplo_similar, pharm_similar, contextpred_simila, edgepred_similar, infomax_similar, masking_similar, adr, adr1, adr2 , _y, _y1):
        self.drug = drug
        self.adrname = adrname
        self.data_adr = adr
        self.data_adr1 = adr1
        self.data_adr2 = adr2
        self.data_y = _y
        self.data_y1 = _y1
        #FP
        self.data_macc = macc
        self.data_morgan = morgan
        self.data_rtoplo = rtoplo
        self.data_pharm = pharm
        #molecule
        self.data_contextpred = contextpred
        self.data_edgepred = edgepred
        self.data_infomax = infomax
        self.data_masking = masking
        #similar
        self.data_macc_similar = macc_similar
        self.data_morgan_similar = morgan_similar
        self.data_rtoplo_similar = rtoplo_similar
        self.data_pharm_similar = pharm_similar
        self.data_contextpred_similar = contextpred_simila
        self.data_edgepred_similar = edgepred_similar
        self.data_infomax_similar = infomax_similar
        self.data_masking_similar = masking_similar
        

    def __getitem__(self, index):
        drug = self.drug[index]
        adr  = self.adrname[index]
        adr_1 = self.data_adr[index]
        adr1_1 = self.data_adr1[index]
        adr1_2 = self.data_adr2[index]
        label_1 = self.data_y[index]
        label_2 = self.data_y1[index]
        
        macc_1 = self.data_macc[index]
        morgan_1 = self.data_morgan[index]
        rtoplo_1 = self.data_rtoplo[index]
        pharm_1 = self.data_pharm[index]
        
        contextpred_1 = self.data_contextpred[index]
        edgepred_1 = self.data_edgepred[index]
        infomax_1 = self.data_infomax[index]
        masking_1 = self.data_masking[index]
        
        similar_macc_1 = self.data_macc_similar[index]
        similar_morgan_1 = self.data_morgan_similar[index]
        similar_rtoplo_1 = self.data_rtoplo_similar[index]
        similar_pharm_1 = self.data_pharm_similar[index]
        similar_contextpred_1 = self.data_contextpred_similar[index]
        similar_edgepred_1 = self.data_edgepred_similar[index]
        similar_infomax_1 = self.data_infomax_similar[index]
        similar_masking_1 = self.data_masking_similar[index]

        return drug, adr, macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2, label_1, label_2

    def __len__(self):
        # 返回整个数据集的大小
        return len(self.data_y)

def pre_data(train_data, vali_data, batch, folder):
    print(f"Checking the memory size of variables inside pre_data"*5)
    data_train_binary = train_data
    data_vali_binary = vali_data
    print(f"Shape of the training set: {data_train_binary.shape}, number of drugs: {data_train_binary.drugname.drop_duplicates().shape[0]}, number of ADRs: {data_train_binary.adr.drop_duplicates().shape[0]}")
    print(f"Shape of the validation set: {data_vali_binary.shape}, number of drugs: {data_vali_binary.drugname.drop_duplicates().shape[0]}, number of ADRs: {data_vali_binary.adr.drop_duplicates().shape[0]}")
    print(f"Memory size of data_train_binary: {round(sys.getsizeof(data_train_binary)/1024/1024/1024, 3)}G")
    print(f"Memory size of data_vali_binary: {round(sys.getsizeof(data_vali_binary)/1024/1024/1024, 3)}G")
    # Check if ADR labels in the validation set are within the range
    if ~data_vali_binary['adr'].drop_duplicates().isin(data_train_binary['adr'].drop_duplicates().tolist()).all():
        print(f"Unknown ADRs in validation set split for fold {folder}")
        # Remove unknown ADRs from the validation set
        adr_all = data_vali_binary.adr.drop_duplicates().tolist()
        data_vali_binary = data_vali_binary[data_vali_binary.adr.isin(data_train_binary.adr)]
        remove_adr = list(set(adr_all) - set(data_vali_binary['adr'].drop_duplicates().tolist())) 
        # Reset index
        data_vali_binary.reset_index(drop=True, inplace=True)
        print(f"Removed ADRs: {remove_adr}")
        print(f"Remaining validation set shape: {data_vali_binary.shape}")
        del adr_all, remove_adr
        
    # Calculate ADR similarity in the training set
    # Obtain the drug relationships and frequency distribution list of ADRs in the training set
    adr_drug_b = data_train_binary.pivot_table(index='adr', columns='drugname', values=train_data.columns[2], fill_value=0) # Jaccard coefficient
    adr_drug_f = data_train_binary.pivot_table(index='adr', columns='drugname', values=train_data.columns[3], fill_value=0) # Cosine similarity score
    
    adr_drug_f.loc[:,'adr_drug_freq'] =  adr_drug_f.values.tolist()
    adr_drug_f.loc[:,'adr_drug_binary'] =  adr_drug_b.values.tolist()
    
    adr_drug_f['adr'] = adr_drug_f.index
    adr_drug_f.reset_index(drop=True, inplace=True)
    
    train_adr_drugf = adr_drug_f.loc[:,['adr','adr_drug_freq', 'adr_drug_binary']]
    del adr_drug_b, adr_drug_f
    
    train_adr_freq = train_adr_drugf.adr_drug_freq.tolist()
    train_adr_binary = train_adr_drugf.adr_drug_binary.tolist()

    # Calculate drug similarity for ADRs
    train_adr_drugf.loc[:,'adr_drug_freq_similar'] = train_adr_drugf['adr_drug_freq'].apply(lambda x : calculate_tanimoto_similarity_matrix(x, train_adr_freq))
    train_adr_drugf.loc[:,'adr_drug_binary_similar'] = train_adr_drugf['adr_drug_binary'].apply(lambda x : calculate_cos_sim(x, train_adr_binary))
    del train_adr_freq, train_adr_binary

    # Store similarity calculations for each drug in the training set with other drugs in the training set
    # The similarity calculated here is Jaccard or Cosine similarity, representing fingerprint similarity
    train_durg = data_train_binary[['drugname', 'Macc', 'Morgan', 'Rtoplo', 'pharmfp',
    'contextpred', 'edgepred', 'infomax', 'masking']].drop_duplicates(subset=['drugname'])
    vali_drug = data_vali_binary[['drugname', 'Macc', 'Morgan', 'Rtoplo', 'pharmfp',
    'contextpred', 'edgepred', 'infomax', 'masking']].drop_duplicates(subset=['drugname'])
    

    train_macc_list= [safe_eval(i) for i in train_durg.Macc]
    train_morgan_list= [safe_eval(i) for i in train_durg.Morgan]
    train_rtoplo_list= [safe_eval(i) for i in train_durg.Rtoplo]
    train_pharm_list= [safe_eval(i) for i in train_durg.pharmfp]

    
    # Calculate Cosine similarity and molecular embedding similarity
    train_contextpred_list= train_durg.contextpred
    train_edgepred_list= train_durg.edgepred
    train_infomax_list= train_durg.infomax
    train_masking_list= train_durg.masking


    # Calculate similarity between drugs in the training set
    train_durg1 = train_durg.copy()
    # Calculate similarity between validation set drugs and training set drugs
    vali_drug1 = vali_drug.copy()
    del train_durg, vali_drug

    newcolname = (['macc_similar','morgan_similar','rtoplo_similar'] + ['pharm_similar', 'contextpred_similar', 'edgepred_similar','infomax_similar','masking_similar'])*2
    oldcolname = (['Macc', 'Morgan', 'Rtoplo'] + ['pharmfp', 'contextpred','edgepred', 'infomax', 'masking'])*2
    caculate_list = ([train_macc_list,train_morgan_list, train_rtoplo_list] + [train_pharm_list,train_contextpred_list, train_edgepred_list,train_infomax_list, train_masking_list])*2
    
    print('Starting parallel similarity calculation'*5)
    t_pata = time.time()
    def parall_caculate(i):
        newcol = newcolname[i]
        oldcol = oldcolname[i]
        caculate = caculate_list[i]
        if i in (0,1,2):
            df = train_durg1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : calculate_tanimoto_similarity_matrix(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        elif i in (3,4,5,6,7):
            df = train_durg1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : calculate_cos_sim(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        elif i in (8,9,10):
            df = vali_drug1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : calculate_tanimoto_similarity_matrix(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        elif i in (11,12,13,14,15):
            df = vali_drug1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : calculate_cos_sim(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        del df
        gc.collect()
        return return_data_binary1, i  
    fp_list_data = Parallel(n_jobs= -1)(delayed(parall_caculate)(i) for i in range(16))
    del train_macc_list, train_morgan_list, train_rtoplo_list, train_pharm_list, train_contextpred_list, train_edgepred_list, train_infomax_list, train_masking_list, newcolname, oldcolname, caculate_list
    gc.collect()
    print(f'Length of fp_list_data: {len(fp_list_data)}')
    print(f'Shape of the first element in fp_list_data: {fp_list_data[0][0].shape}')
    print(f"Memory size of fp_list_data: {round(sys.getsizeof(fp_list_data)/1024/1024/1024, 3)}G")

# Iterate through each subset
    for i in range(16):
    # Read the subset
        sub_dataset = fp_list_data[i][0]
        if fp_list_data[i][1] <= 7:
            train_durg1 = pd.merge(train_durg1, sub_dataset, on='drugname')
        else:
            vali_drug1 = pd.merge(vali_drug1, sub_dataset, on='drugname')
    # Merge the subset into the main dataset by the 'id' column
    del fp_list_data
    print('Parallel similarity calculation finished'*5)
    print(f'Time taken for parallel computation: {(time.time()-t_pata)/60} minutes')

# Merge datasets
    data_train_binary1 = pd.merge(data_train_binary, train_durg1[['drugname','macc_similar','morgan_similar','rtoplo_similar','pharm_similar','contextpred_similar','edgepred_similar','infomax_similar','masking_similar']], how='left', on='drugname')
    data_train_binary1 = pd.merge(data_train_binary1, train_adr_drugf[['adr','adr_drug_freq_similar', 'adr_drug_binary_similar']], how='left', on='adr')

    data_vali_binary1 = pd.merge(data_vali_binary, vali_drug1[['drugname','macc_similar','morgan_similar','rtoplo_similar','pharm_similar','contextpred_similar','edgepred_similar','infomax_similar','masking_similar']], how='left', on='drugname')
    data_vali_binary1 = pd.merge(data_vali_binary1, train_adr_drugf[['adr','adr_drug_freq_similar', 'adr_drug_binary_similar']], how='left', on='adr')
    print(f"Memory size of data_train_binary1: {round(sys.getsizeof(data_train_binary1)/1024/1024/1024, 3)}G")
    print(f"Memory size of data_vali_binary1: {round(sys.getsizeof(data_vali_binary1)/1024/1024/1024, 3)}G")

    del data_train_binary, data_vali_binary

    print('Start converting to np'*5)
    t_np = time.time()
    np_index = (['association', 'freq', 'adr_similar',
    'Macc', 'Morgan', 'Rtoplo', 'pharmfp', 'contextpred',
    'edgepred', 'infomax', 'masking', 'macc_similar', 'morgan_similar',
    'rtoplo_similar', 'pharm_similar', 'contextpred_similar',
    'edgepred_similar', 'infomax_similar', 'masking_similar',
    'adr_drug_freq_similar', 'adr_drug_binary_similar'])*2

    def para_convert_np(xx):
        col = np_index[xx]
        if xx <= 20: # train
            if col in [0,1]: # drugname and adrname
                re_np = np.array(data_train_binary1.loc[:,col])
            else: # other features
                re_np = np.array([safe_eval(i) for i in data_train_binary1.loc[:, col]])
        else: # vali
            if col in [0,1]: # drugname and adrname
                re_np = np.array(data_vali_binary1.loc[:,col])
            else: # other features
                re_np = np.array([safe_eval(i) for i in data_vali_binary1.loc[:, col]])
        gc.collect()
        return re_np, col, xx

    # Debug information
    def debug_parallel_task(xx):
        try:
            result = para_convert_np(xx)
            print(f"Task {xx} completed successfully.")
            return result
        except Exception as e:
            print(f"Task {xx} failed with exception: {e}")
            raise

    # Parallel computation
    re_np_list = Parallel(n_jobs=-1)(delayed(debug_parallel_task)(xx) for xx in range(42))
    drug_train = np.array(data_train_binary1.drugname)
    drug_vali = np.array(data_vali_binary1.drugname)
    adr_train = np.array(data_train_binary1.adr)
    adr_vali = np.array(data_vali_binary1.adr)

    print(f'Length of re_np_list: {len(re_np_list)}')
    print(f"Memory size of re_np_list: {round(sys.getsizeof(re_np_list)/1024/1024/1024, 3)} GB")
    print(re_np_list[0][0].shape)
    print(f"{sys.getsizeof(re_np_list[0][0])/1024/1024/1024} GB")
    print(f'Converting to np took {(time.time()-t_np)/60} minutes')

    for i in re_np_list:
        if i[2] <= 20:
            if i[1] == 'association':
                train_label = i[0]
            elif i[1] == 'freq':
                train_label_f = i[0]
            elif i[1] == 'adr_similar':
                train_adr = i[0]
            elif i[1] == 'Macc':
                Macc_train = i[0]
            elif i[1] == 'Morgan':
                morgan_train = i[0]
            elif i[1] == 'Rtoplo':
                Rtoplo_train = i[0]
            elif i[1] == 'pharmfp':
                pharm_train = i[0]
            elif i[1] == 'contextpred':
                contextpred_train = i[0]
            elif i[1] == 'edgepred':
                edgepred_train = i[0]
            elif i[1] == 'infomax':
                infomax_train = i[0]
            elif i[1] == 'masking':
                masking_train = i[0]
            elif i[1] == 'macc_similar':
                macc_similar_train = i[0]
            elif i[1] == 'morgan_similar':
                morgan_similar_train = i[0]
            elif i[1] == 'rtoplo_similar':
                rtoplo_similar_train = i[0]
            elif i[1] == 'pharm_similar':
                pharm_similar_train = i[0]
            elif i[1] == 'contextpred_similar':
                contextpred_simila_train = i[0]
            elif i[1] == 'edgepred_similar':
                edgepred_similar_train = i[0]
            elif i[1] == 'infomax_similar':
                infomax_similar_train = i[0]
            elif i[1] == 'masking_similar':
                masking_similar_train = i[0]
            elif i[1] == 'adr_drug_binary_similar':
                train_adr_1 = i[0]
            elif i[1] == 'adr_drug_freq_similar':
                train_adr_2 = i[0]
            else:
                print("There's an issue during np conversion in pre_data")
        else:
            if i[1] == 'association':
                vali_label = i[0]
            elif i[1] == 'freq':
                vali_label_f = i[0]
            elif i[1] == 'adr_similar':
                vali_adr = i[0]
            elif i[1] == 'Macc':
                Macc_vali = i[0]
            elif i[1] == 'Morgan':
                morgan_vali = i[0]
            elif i[1] == 'Rtoplo':
                Rtoplo_vali = i[0]
            elif i[1] == 'pharmfp':
                pharm_vali = i[0]
            elif i[1] == 'contextpred':
                contextpred_vali = i[0]
            elif i[1] == 'edgepred':
                edgepred_vali = i[0]
            elif i[1] == 'infomax':
                infomax_vali = i[0]
            elif i[1] == 'masking':
                masking_vali = i[0]
            elif i[1] == 'macc_similar':
                macc_similar_vali = i[0]
            elif i[1] == 'morgan_similar':
                morgan_similar_vali = i[0]
            elif i[1] == 'rtoplo_similar':
                rtoplo_similar_vali = i[0]
            elif i[1] == 'pharm_similar':
                pharm_similar_vali = i[0]
            elif i[1] == 'contextpred_similar':
                contextpred_simila_vali = i[0]
            elif i[1] == 'edgepred_similar':
                edgepred_similar_vali = i[0]
            elif i[1] == 'infomax_similar':
                infomax_similar_vali = i[0]
            elif i[1] == 'masking_similar':
                masking_similar_vali = i[0]
            elif i[1] == 'adr_drug_binary_similar':
                vali_adr_1 = i[0]
            elif i[1] == 'adr_drug_freq_similar':
                vali_adr_2 = i[0]

    global similar_feature
    similar_feature = [macc_similar_train.shape[1], morgan_similar_train.shape[1], rtoplo_similar_train.shape[1], pharm_similar_train.shape[1], contextpred_simila_train.shape[1], edgepred_similar_train.shape[1], infomax_similar_train.shape[1], masking_similar_train.shape[1], train_adr.shape[1], train_adr_1.shape[1], train_adr_2.shape[1]]

    train_dataset = Mydataset(drug_train, adr_train, Macc_train, morgan_train, Rtoplo_train, pharm_train, contextpred_train, edgepred_train, infomax_train, masking_train, macc_similar_train, morgan_similar_train, rtoplo_similar_train, pharm_similar_train, contextpred_simila_train, edgepred_similar_train, infomax_similar_train, masking_similar_train, train_adr, train_adr_1, train_adr_2, train_label, train_label_f)

    vail_dataset = Mydataset(drug_vali, adr_vali, Macc_vali, morgan_vali, Rtoplo_vali, pharm_vali, contextpred_vali, edgepred_vali, infomax_vali, masking_vali, macc_similar_vali, morgan_similar_vali, rtoplo_similar_vali, pharm_similar_vali, contextpred_simila_vali, edgepred_similar_vali, infomax_similar_vali, masking_similar_vali, vali_adr, vali_adr_1, vali_adr_2, vali_label, vali_label_f)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, num_workers=8, shuffle=True)
    vail_loader = DataLoader(dataset=vail_dataset, batch_size=batch, num_workers=8, shuffle=True)

    print(f"drug_train shape: {drug_train.shape}")
    print(f"adr_train shape: {adr_train.shape}")
    print(f"Macc_train shape: {Macc_train.shape}")
    print(f"train_adr shape: {train_adr.shape}")
    print(f"train_adr_1 shape: {train_adr_1.shape}")
    print(f"train_adr_2 shape: {train_adr_2.shape}")
    print(f"morgan_similar_train shape: {morgan_similar_train.shape} Memory size: {round(sys.getsizeof(morgan_similar_train)/1024/1024/1024, 3)}G")
    print(f"train_dataset Memory size: {round(sys.getsizeof(train_dataset)/1024/1024/1024, 3)}G")
    print(f"vail_dataset Memory size: {round(sys.getsizeof(vail_dataset)/1024/1024/1024, 3)}G")
    print(f"train_loader Memory size: {round(sys.getsizeof(train_loader)/1024/1024/1024, 3)}G")
    print(f"vail_loader Memory size: {round(sys.getsizeof(vail_loader)/1024/1024/1024, 3)}G")

    del Macc_train, morgan_train, Rtoplo_train, pharm_train, contextpred_train, edgepred_train, infomax_train, masking_train, macc_similar_train, morgan_similar_train, rtoplo_similar_train, pharm_similar_train, contextpred_simila_train, edgepred_similar_train, infomax_similar_train, masking_similar_train, train_adr, train_adr_1, train_adr_2, train_label, train_label_f, Macc_vali, morgan_vali, Rtoplo_vali, pharm_vali, contextpred_vali, edgepred_vali, infomax_vali, masking_vali, macc_similar_vali, morgan_similar_vali, rtoplo_similar_vali, pharm_similar_vali, contextpred_simila_vali, edgepred_similar_vali, infomax_similar_vali, masking_similar_vali, vali_adr, vali_adr_1, vali_adr_2, vali_label, vali_label_f, train_dataset, vail_dataset, drug_train, drug_vali, adr_train, adr_vali
    gc.collect()
    return train_loader, vail_loader

def run(train_data, vali_data, idx, device, train, evaluate, metrics_res, args):
    # Sampling the training set
    train_data_p = train_data[train_data.association == 1]
    train_data_n = train_data[train_data.association == 0]
    train_data_n_sample = train_data_n.sample(train_data_p.shape[0], random_state=123456)
    train_data_sample = pd.concat([train_data_p, train_data_n_sample], axis=0).reset_index(drop=True)
    del train_data, train_data_p, train_data_n, train_data_n_sample
    gc.collect()
    loss_batch_train = []
    loss_epoch_vali = []
    folder=idx+1
    t0 = time.time()
    auc_max, aupr_max, rmse_min = 0, 0 , 100
    train_loader,vail_loader = pre_data(train_data_sample, vali_data, args.batch, folder)
    
    # Save
    with open(f'./result/loader/vail_loader{folder}.pkl', 'wb') as f:
        dill.dump(vail_loader, f)
    print(f"Memory size of train_loader: {sys.getsizeof(train_loader)/1024/1024/1024} G")
    print(f"Memory size of vail_loader: {sys.getsizeof(vail_loader)/1024/1024/1024} G")


    feature_size = args.feature_size
    if feature_size % 8 ==0:
        model = MultiFG(output_size=1, feature_size=feature_size, similar_feature_size=similar_feature, out_layer="KAN", device=device, full_layerb=[feature_size*10,1024,512,256, 1], full_layerf=[feature_size*10,256, 1], activation=nn.ReLU()) #full_layer
    else:
        model = MultiFG(output_size=1, feature_size=feature_size, similar_feature_size=similar_feature, out_layer="KAN", device=device, full_layerb=[feature_size*10 + 8, 1024,512,256, 1], full_layerf=[feature_size*10 + 8, 256, 1], activation=nn.ReLU()) #full_layer
    for name, param in model.named_parameters():
            if 'weight' in name:
                if len(param.size()) ==2:
                    init.xavier_normal
            elif 'bias' in name:
                try:
                    init.constant_(param, 0) 
                except:
                    print("Network bias initialization failed.")
                else:
                    pass
    model.to(device)
    loss_1 = nn.BCEWithLogitsLoss()
    loss_2 = nn.MSELoss()
    #all 100 epoch
    for epoch in range(args.epoch[0]):
        print("-"*20+'all'+'-'*20)
        print(f'**********{epoch+1}*********')
        # weight
        weitght = args.weight[0]
        optimizerall = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=0.005)
        model.train()
        _,_,_, _,_ = train(model, train_loader, epoch, device, optimizerall, loss_batch_train, loss_1, loss_2, weitght, choose_mode='all')
    # fine-tuning binary 10 epoch
    for epoch in range(args.epoch[1]):
        print("-"*20+'binary'+'-'*20)
        weitght = args.weight[1]
        optimizerbinary = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.005)
        model.train()
        _,_,_,_,_ = train(model, train_loader, epoch, device, optimizerbinary, loss_batch_train, loss_1, loss_2, weitght, choose_mode='binary')
    #fine-tuning freq 10 epoch
    for epoch in range(args.epoch[2]):
        print("-"*20+'freq'+'-'*20)
        weitght = args.weight[2]
        optimizerfreq = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=0.005)
        model.train()
        auc_1,aupr_1,rmse_1, mae_1,spearman_1 = train(model, train_loader, epoch, device, optimizerfreq, loss_batch_train, loss_1, loss_2, weitght, choose_mode='freq') 

        model.eval()
        auc_v, aupr_v, rmse_v, mae_v, spearman_v= evaluate(model, vail_loader, epoch, device, loss_epoch_vali, loss_1, loss_2, weitght)
        
        #Save the model from the epoch with the highest validation AUC.
        if auc_v > auc_max:
            auc_max, aupr_maxauc, rmse_maxauc, mae_maxauc, spearman_maxauc, auc_max_train, aupr_maxauc_train, rmse_maxauc_train, mae_maxauc_train, spearman_maxauc_train, epoch_maxauc, model_auc, = auc_v, aupr_v, rmse_v, mae_v, spearman_v, auc_1, aupr_1, rmse_1, mae_1, spearman_1, epoch, copy.deepcopy(model)
            
        if aupr_v > aupr_max:
            aupr_max, auc_maxaupr, rmse_maxaupr, mae_maxaupr, spearman_maxaupr, aupr_max_train, auc_maxaupr_train, rmse_maxaupr_train, mae_maxaupr_train, spearman_maxaupr_train, epoch_maxaupr = aupr_v, auc_v, rmse_v, mae_v, spearman_v, aupr_1, auc_1, rmse_1, mae_1, spearman_1, epoch
            # model_aupr = copy.deepcopy(model)
        
        if rmse_v < rmse_min:
            rmse_min, mae_minrmse, auc_minrmse, aupr_minrmse, spearman_minrmse, rmse_min_train, mae_minrmse_train, auc_minrmse_train, aupr_minrmse_train, spearman_minrmse_train, epoch_minrmse, = rmse_v, mae_v, auc_v, aupr_v, spearman_v, rmse_1, mae_1, auc_1, aupr_1, spearman_1, epoch
            # model_rmse = copy.deepcopy(model)

            
    #save model performance
    metrics_res['dataset'].append('train')
    metrics_res['folder'].append(folder)
    metrics_res["best_epoch"].append(epoch_maxauc),
    metrics_res["auc_max"].append(auc_max_train)
    metrics_res["aupr_maxauc"].append(aupr_maxauc_train)
    metrics_res["rmse_maxauc"].append(rmse_maxauc_train)
    metrics_res['mae_maxauc'].append(mae_maxauc_train)
    metrics_res["spearman_maxauc"].append(spearman_maxauc_train) 
    metrics_res["epoch_maxaupr"].append(epoch_maxaupr)
    metrics_res["aupr_max"].append(aupr_max_train)
    metrics_res["auc_maxaupr"].append(auc_maxaupr_train)
    metrics_res["rmse_maxaupr"].append(rmse_maxaupr_train)
    metrics_res['mae_maxaupr'].append(mae_maxaupr_train)
    metrics_res["spearman_maxaupr"].append(spearman_maxaupr_train)
    metrics_res["epoch_minrmse"].append(epoch_minrmse) 
    metrics_res["rmse_min"].append(rmse_min_train)
    metrics_res['mae_minrmse'].append(mae_minrmse_train)
    metrics_res["auc_minrmse"].append(auc_minrmse_train)
    metrics_res["aupr_minrmse"].append(aupr_minrmse_train)
    metrics_res["spearman_minrmse"].append(spearman_minrmse_train)
    #vali dataset
    metrics_res['dataset'].append('vali')
    metrics_res['folder'].append(folder)
    metrics_res["best_epoch"].append(epoch_maxauc),
    metrics_res["auc_max"].append(auc_max)
    metrics_res["aupr_maxauc"].append(aupr_maxauc)
    metrics_res["rmse_maxauc"].append(rmse_maxauc)
    metrics_res['mae_maxauc'].append(mae_maxauc) 
    metrics_res["spearman_maxauc"].append(spearman_maxauc) 
    metrics_res["epoch_maxaupr"].append(epoch_maxaupr)
    metrics_res["aupr_max"].append(aupr_max)
    metrics_res["auc_maxaupr"].append(auc_maxaupr)
    metrics_res["rmse_maxaupr"].append(rmse_maxaupr)
    metrics_res['mae_maxaupr'].append(mae_maxaupr)
    metrics_res["spearman_maxaupr"].append(spearman_maxaupr)
    metrics_res["epoch_minrmse"].append(epoch_minrmse) 
    metrics_res["rmse_min"].append(rmse_min)
    metrics_res['mae_minrmse'].append(mae_minrmse)
    metrics_res["auc_minrmse"].append(auc_minrmse)
    metrics_res["aupr_minrmse"].append(aupr_minrmse)
    metrics_res["spearman_minrmse"].append(spearman_minrmse)


    
    del train_loader, vail_loader, model, loss_1, loss_2
    gc.collect()

    print(f"The time for one fold is: {(time.time() - t0) / 60} minutes")
    torch.save(model_auc.state_dict(), f'./result/model/dicit_model{folder}.pth')