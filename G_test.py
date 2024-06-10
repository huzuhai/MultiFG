import os
import rdkit
import torch.optim as optim
import dgl
import numpy as np
import pandas as pd
import torch
import pubchempy
import re
import sys
import torch.nn.functional as F
import math
import pickle
import gc
import time
import psutil
import tracemalloc
import ast
import dill
import copy
import torch.nn.init as init
import random

from scipy import stats
from joblib import Parallel, delayed
from multiprocessing import pool
from tqdm import tqdm
from math import sqrt
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem import Draw, AllChem
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.model.pretrain.moleculenet import *
from dgllife.model.pretrain.generative_models import *
from dgllife.model.pretrain.property_prediction import *
from dgllife.model.pretrain.reaction import *
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch import nn
from rdkit import Chem
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc, mean_absolute_error,mean_squared_error
from torch.autograd import Variable

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

def load_binary(path, file_name, file_type, columns):
    assert file_type=='.csv' , "only csv filetype be witten"
    file=os.path.join(path,file_name+file_type)
    data=pd.read_csv(file)
    data.columns = columns
    data1 = data[['drugname', 'adr', 'association']]
    return data1

def load_data(path,file_name,file_type):
    if file_type== ".csv":
        file=os.path.join(path,file_name+file_type)
    data2=pd.read_csv(file)
    data2.columns = ['drugname','adr','freq','smiles','adr_similar', 'Macc','pubchem','Morgan','Rtoplo','pharmfp']
    data2['freq'] = data2['freq'].astype(int)
    return data2

#获取分子的FP
def getcompund(smiles,drugname):
    try:
        compound = pubchempy.get_compounds(drugname,'name')[0]
    except:
        compound = pubchempy.get_compounds(smiles,'smiles')[0]
    pubchem = compound.cactvs_fingerprint
    pubchem = np.array([int(i) for i in pubchem])
    return list(pubchem)
def fingerpint(smiles):
    #Macc
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Macc = AllChem.GetMACCSKeysFingerprint(mol)
    Macc = np.array(Macc)
    

    #Morgan
    Morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    Morgan = np.array(Morgan)
    
    #rdkit(Topological)
    #rdkit独有的灵感来源于daylight
    Rtoplo = Chem.RDKFingerprint(mol)
    Rtoplo = np.array(Rtoplo)
    
    # #pharmacophore
    phar = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
    phar = np.array(phar)
    return list(Macc), list(Morgan), list(Rtoplo), list(phar)

#预训练模型
#返回的graphs的顺序与输入的smiles顺序一致
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
    checkpoint = torch.load(name+ '_pre_trained.pth')
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


class Mydataset(torch.utils.data.Dataset):
    def __init__(self, macc, morgan, rtoplo, pharm,   contextpred, edgepred, infomax, masking, macc_similar, morgan_similar, rtoplo_similar, pharm_similar, contextpred_simila, edgepred_similar, infomax_similar, masking_similar, adr, adr1, adr2 , _y, _y1):
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

        return macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2, label_1, label_2

    def __len__(self):
        # 返回整个数据集的大小
        return len(self.data_y)

def safe_eval(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x
    
#相似性函数,计算jaccord系数相似性
def calculate_tanimoto_similarity_matrix(list1, list2): #list1是一个fp，而list2是一个fp列表
    #删除list2中全部元素为0的子列表
    index = [not all(np.array(i) == 0) for i in list2]
    list2 = list(np.array(list2)[index])
    if all(np.array(list1) == 0): #如果输入的list1需要计算的个体的值的全部元素为0则直接返回0向量
        return np.zeros(len(list2))
    # 将 list1 扩展成与 list2 相同形状的矩阵
    list1_matrix = np.tile(list1, (len(list2), 1))
    # 将 list2 转换为 numpy 数组
    list2_matrix = np.array(list2)
    # 计算 Tanimoto similarity
    intersection_count = np.sum(list1_matrix & list2_matrix, axis=1)
    union_count = np.sum(list1_matrix | list2_matrix, axis=1)
    # 避免除以零的情况，也就是说如果一个药物/adr的特征全部为0，则其与其他所有包括他自己的相似性都设置为0
    similarity = intersection_count / union_count 
    return similarity  #返回的是一个np格式的数据


# 计算余弦相似度
def caculate_cos_sim(fp, fp_list):
    #删除list2中全部元素为0的子列表
    index = [not all(np.array(i) == 0) for i in fp_list]
    fp_list = list(np.array(fp_list)[index])
    if all(np.array(fp) == 0): #如果输入的list1需要计算的个体的值的全部元素为0则直接返回0向量
        return np.zeros(len(fp_list))
    
    _fp = [fp for i in range(len(fp_list))]
    _dot = [np.dot(i,j) for i,j in zip(_fp, fp_list)]
    _norm1 = [np.linalg.norm(i) for i in _fp]
    _norm2 = [np.linalg.norm(i) for i in fp_list]
    # #对于范数为0的向量将分布变为1，分子不变恒等于0
    # _norm1 = [1 if element == 0 else element for element in _norm1]
    # _norm2 = [1 if element == 0 else element for element in _norm2]
    _div = [i*j for i,j in zip(_norm1, _norm2)]
    _result = [i/j for i,j in zip(_dot, _div)]
    return np.array(_result)




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
            loss2 = criterion2(output2[target1==1], target2[target1==1])
        except:
            print('当前batch没有已知的药物不良反应对')
            loss = torch.zeros(1, requires_grad=True)
        else:
            loss = loss2
    if criterion1 is not None and criterion2 is not None:
        loss1 = criterion1(output1, target1)
        try:
            loss2 = criterion2(output2[target1==1], target2[target1==1])
        except:
            print('当前batch没有已知的药物不良反应对')
            loss = loss1
        else:   
            loss = a*loss1 + loss2
    loss.requires_grad_()
    loss.backward()
    optimizer.step()
    return loss, output1, output2, target1, target2

class ScaledDotProductionAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductionAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v] 全文两处用到注意力，一处是self attention，另一处是co attention，前者不必说，后者的k和v都是encoder的输出，所以k和v的形状总是相同的
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        # 1) 计算注意力分数QK^T/sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores: [batch_size, n_heads, len_q, len_k]
        # 2)  进行 mask 和 softmax
        # scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)  # attn: [batch_size, n_heads, len_q, len_k]
        # 3) 乘V得到最终的加权和
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        '''
        得出的context是每个维度(d_1-d_v)都考虑了在当前维度(这一列)当前token对所有token的注意力后更新的新的值，
        换言之每个维度d是相互独立的，每个维度考虑自己的所有token的注意力，所以可以理解成1列扩展到多列

        返回的context: [batch_size, n_heads, len_q, d_v]本质上还是batch_size个句子，
        只不过每个句子中词向量维度512被分成了8个部分，分别由8个头各自看一部分，每个头算的是整个句子(一列)的512/8=64个维度，最后按列拼接起来
        '''
        return context # context: [batch_size, n_heads, len_q, d_v]
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, n_heads, d_k, d_v, device): #128 2 64 64
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, input_Q, input_K, input_V): #bath_size, 9, 64
        '''
        input_Q: [batch_size, len_q, d_model] len_q是作为query的句子的长度，比如enc_inputs（2,5,512）作为输入，那句子长度5就是len_q
        input_K: [batch_size, len_k, d_model]
        input_K: [batch_size, len_v(len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)

        # 1）linear projection [batch_size, seq_len, d_model] ->  [batch_size, n_heads, seq_len, d_k/d_v]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2) # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # 2）计算注意力
        # # 自我复制n_heads次，为每个头准备一份mask
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductionAttention(self.d_k)(Q, K, V) # context: [batch_size, n_heads, len_q, d_v]

        # 3）concat部分
        context = torch.cat([context[:,i,:,:] for i in range(context.size(1))], dim=-1)
        output = self.concat(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.device)(output + residual)  # output: [batch_size, len_q, d_model]
        '''        
        最后的concat部分，网上的大部分实现都采用的是下面这种方式（也是哈佛NLP团队的写法）
        context = context.transpose(1, 2).reshape(batch_size, -1, d_model)
        output = self.linear(context)
        但是我认为这种方式拼回去会使原来的位置乱序，于是并未采用这种写法，两种写法最终的实验结果是相近的
        '''

class Residual(nn.Module):  # @save
	def __init__(self, input_channels, num_channels,
	             use_1x1conv=False, strides=1):
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels, num_channels,
		                       kernel_size=3, padding=1, stride=strides)
		self.conv2 = nn.Conv2d(num_channels, num_channels,
		                       kernel_size=3, padding=1)
		if use_1x1conv:
			self.conv3 = nn.Conv2d(input_channels, num_channels,
			                       kernel_size=1, stride=strides)
		else:
			self.conv3 = None
		self.bn1 = nn.BatchNorm2d(num_channels)
		self.bn2 = nn.BatchNorm2d(num_channels)

	def forward(self, X):  # batch*1*167
		Y = F.relu(self.bn1(self.conv1(X)))
		Y = self.bn2(self.conv2(Y))
		if self.conv3:
			X = self.conv3(X)
		Y += X
		return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
	blk = []
	for i in range(num_residuals):
		if i == 0 and not first_block:
			blk.append(Residual(input_channels, num_channels,
			                    use_1x1conv=True, strides=2))
		else:
			blk.append(Residual(num_channels, num_channels))
	return blk

class map_layer(nn.Module):
    def __init__(self, hide: list=['infeature', 128, 32, 1], activate=nn.ReLU(),  normal=True):#hide:[]
        super(map_layer, self).__init__()
        self.map_layer = nn.ModuleList()
        for i, j in zip(hide, hide[1:]):
            self.map_layer.append(nn.Linear(i, j))
            if normal:
                self.map_layer.append(nn.BatchNorm1d(j))
            self.map_layer.append(activate)

    def forward(self, x):
        for layer in self.map_layer:
            x = layer(x)
        return x

class MLP(nn.Module):
    def __init__(self, hide: list=['infeature', 128, 32, 1], activate=nn.ReLU(), drop=0.5, normal=True):#hide:[]
        super(MLP, self).__init__()
        self.combine_layerf = nn.ModuleList()
        for i, j in zip(hide, hide[1:]):
            self.combine_layerf.append(nn.Linear(i, j))
            if j != 1:
                if normal:
                    self.combine_layerf.append(nn.BatchNorm1d(j))
                self.combine_layerf.append(activate)
                self.combine_layerf.append(nn.Dropout(drop))

    def forward(self, x):
        for layer in self.combine_layerf:
            x = layer(x)
        return x
    
class MultiFG(nn.Module):
    def __init__(self, feature_size, output_size, similar_feature_size, device, out_layer:str="KAN", full_layerb:list = ['feature_size'*10, 256, 64, 1], full_layerf: list=['feature_size'*10, 256, 64, 1], activation=nn.ReLU()):
        super(MultiFG, self).__init__()
        self.out_layer = out_layer
        self.device=device
        self.output_size = output_size
        self.feature_size = feature_size
        #macc指纹的映射层
        self.share_size = [feature_size]
        self.map_macc = map_layer([167]+self.share_size, activate=activation, normal=True)

        #pubchem指纹的映射层
        self.map_pubchem = map_layer([881]+self.share_size, activate=activation, normal=True)
        
        #morgan指纹的映射层
        self.map_morgan = map_layer([2048]+self.share_size, activate=activation, normal=True)

        #Rtoplo指纹的映射层
        self.map_Rtoplo = map_layer([2048]+self.share_size, activate=activation, normal=True)
        
        #pubchem指纹的映射层
        self.map_pharm = map_layer([441]+self.share_size, activate=activation, normal=True)


        #molecule的映射层
        self.map_molecule = map_layer([300]+self.share_size, activate=activation)
        
        #adr的映射层,是基于相似的特征，所以这里的输入尺度应该是指训练集所有adr的维度
        self.map_adr = map_layer([similar_feature_size[8]]+self.share_size, activate=activation)
        self.map_adr_12 = map_layer([similar_feature_size[9]]+self.share_size, activate=activation)
 
        
        #相似的映射层，训练集所有药物数量维度
        self.map_similar_macc = map_layer([similar_feature_size[0]]+self.share_size,  activate=activation)
        self.map_similar_morgan = map_layer([similar_feature_size[1]]+self.share_size,  activate=activation)
        self.map_similar_rtoplo = map_layer([similar_feature_size[2]]+self.share_size,  activate=activation)
        self.map_similar_pharm = map_layer([similar_feature_size[3]]+self.share_size,  activate=activation)
        self.map_similar_context = map_layer([similar_feature_size[4]]+self.share_size,  activate=activation)
        self.map_similar_edge = map_layer([similar_feature_size[5]]+self.share_size,  activate=activation)
        self.map_similar_infomax = map_layer([similar_feature_size[6]]+self.share_size,  activate=activation)
        self.map_similar_masking = map_layer([similar_feature_size[7]]+self.share_size,  activate=activation)
        
        #一个高维卷积+两个残差块
        self.residual_cov = nn.Sequential(
            nn.Conv2d(2,2,7,1,3),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            *resnet_block(2, 8, 1, first_block=False),
            *resnet_block(8, 16, 1),
            *resnet_block(16,32,1),
            nn.AvgPool2d(3,1,1)
        )

        self.fusion_layer = MultiHeadAttention(feature_size, 2, int(feature_size/2), int(feature_size/2), self.device)
        
        # 预测层
        self.com_layerf = KAN(full_layerf)
        self.com_layerb = MLP(full_layerb, activation)

    def forward(self, macc, morgan, Rtoplo, pharm, contextpred, edgepred, infomax, masking, macc_similar, morgan_similar, rtoplo_similar, pharm_similar, contextpred_similar, edgepred_similar, infomax_similar, masking_similar, adr_feature, adr_feature1, adr_feature2): #macc batch*167
        '''
        param
            fplist：指纹特征列表
            molecule：分子图嵌入特征
            adrfeature：adr相似特征
            hid1：映射的特征空间的维度
        '''
        #map FP
        macc_1 = self.map_macc(macc)
        morgan_1 = self.map_morgan(morgan)
        Rtoplo_1 = self.map_Rtoplo(Rtoplo)
        pharm_1 = self.map_pharm(pharm)
        # rtoplo_1 = self.create_hiddle(rtoplo.size()[1])(rtoplo)
        #map molecule
        contextpred_1, edgepred_1, infomax_1, masking_1 = self.map_molecule(contextpred), self.map_molecule(edgepred), self.map_molecule(infomax), self.map_molecule(masking)
        
        #map similar
        macc_similar_1, morgan_similar_1, rtoplo_similar_1, pharm_similar_1, contextpred_simila_1, edgepred_similar_1, infomax_similar_1, masking_similar_1 = self.map_similar_macc(macc_similar), self.map_similar_morgan(morgan_similar), self.map_similar_rtoplo(rtoplo_similar), self.map_similar_pharm(pharm_similar), self.map_similar_context(contextpred_similar), self.map_similar_edge(edgepred_similar), self.map_similar_infomax(infomax_similar), self.map_similar_masking(masking_similar)

        #map adr_feature
        adr_feature_1 = self.map_adr(adr_feature)
        adr_feature1_1 = self.map_adr_12(adr_feature1)
        adr_feature1_2 = self.map_adr_12(adr_feature2)
        

        #为每个特征增加一个维度
        macc_2, morgan_2, Rtoplo_2, pharm_2, contextpred_2, edgepred_2, infomax_2, masking_2, macc_similar_2, morgan_similar_2, rtoplo_similar_2, pharm_similar_2, contextpred_simila_2, edgepred_similar_2, infomax_similar_2, masking_similar_2 = macc_1.unsqueeze(1), morgan_1.unsqueeze(1), Rtoplo_1.unsqueeze(1), pharm_1.unsqueeze(1), contextpred_1.unsqueeze(1), edgepred_1.unsqueeze(1), infomax_1.unsqueeze(1), masking_1.unsqueeze(1), macc_similar_1.unsqueeze(1), morgan_similar_1.unsqueeze(1), rtoplo_similar_1.unsqueeze(1), pharm_similar_1.unsqueeze(1), contextpred_simila_1.unsqueeze(1), edgepred_similar_1.unsqueeze(1), infomax_similar_1.unsqueeze(1), masking_similar_1.unsqueeze(1)
        adr_feature_2, adr_feature1_3, adr_feature1_4 = adr_feature_1.unsqueeze(1), adr_feature1_1.unsqueeze(1), adr_feature1_2.unsqueeze(1)
        
        
        # feature_stack = torch.cat([macc_2, morgan_2, Rtoplo_2, pharm_2, molecule_2, macc_2_similar, morgan_2_similar, Rtoplo_2_similar, pharm_2_similar], dim=1) #batchsize * 9 * feature_size
        
        #FP堆积
        feature_FP = torch.cat([macc_2, morgan_2, Rtoplo_2, pharm_2], dim=1) #batch_size * 4 *feature_size
        
        #molecule堆积
        feature_mol = torch.cat([contextpred_2, edgepred_2, infomax_2, masking_2], dim=1) #batch_size * 4 * feature
        
        #similar堆积
        feature_sim = torch.cat([macc_similar_2, morgan_similar_2, rtoplo_similar_2, pharm_similar_2, contextpred_simila_2, edgepred_similar_2, infomax_similar_2, masking_similar_2], dim=1) #batch_size * 8 *feature_size
        
        #adr堆积
        feature_adr = torch.cat([adr_feature_2, adr_feature1_3, adr_feature1_4], dim=1) #batch_size * 3 * feature_size
        
          
        #残差块
        feature_re1,feature_re2,feature_re3 = feature_FP.unsqueeze(1), feature_mol.unsqueeze(1),feature_sim.unsqueeze(1)
        #残差块
        feature_re12 = torch.cat([feature_re1, feature_re2], dim=2)
        feature_fusion2 = self.residual_cov(torch.cat((feature_re12, feature_re3), dim=1))
        #batch_size, 2, 8, feature_size --> batch_size, 32, 1, feature_size/8
        cm1 = feature_fusion2.view(feature_fusion2.size(0), -1)
        #尝试一下只利用药物与不良反应特征经过注意力机制计算后的特征


        #利用feature_adr作为q进行注意力机制,获取药物加权后的特征
        #adr --> FP
        feature_drug = torch.cat([feature_FP,feature_mol,feature_sim], dim=1)
        feature_fusion1 = self.fusion_layer(feature_adr, feature_drug, feature_drug ) #bactch_size,  16, feature_size  -->  batch_size, 3, feature_size
        cm2 = feature_fusion1.view(feature_fusion1.size(0), -1) #摊平batch_size*feature*3
        
        adr_cm = feature_adr.view(feature_adr.size(0), -1)
        combined = torch.cat([cm1, cm2, adr_cm], dim=1)

        # combined = torch.cat(scores_list, dim=1)
        # output = self.combine_layer(combined)
        output_f = self.com_layerf(combined)
        output_b = self.com_layerb(combined)

        return output_b, output_f

def pre_data(train_data, vali_data, batch, folder):
    print(f"查看pre_data内部变量的内存大小"*5)
    data_train_binary = train_data
    data_vali_binary = vali_data
    print(f"训练集的形状：{data_train_binary.shape}，药物数量{data_train_binary.drugname.drop_duplicates().shape[0]},adr数量: {data_train_binary.adr.drop_duplicates().shape[0]}")
    print(f"验证集的形状：{data_vali_binary.shape}，药物数量{data_vali_binary.drugname.drop_duplicates().shape[0]},adr数量: {data_vali_binary.adr.drop_duplicates().shape[0]}")
    print(f"data_train_binary的内存大小为:{round(sys.getsizeof(data_train_binary)/1024/1024/1024, 3)}G")
    print(f"data_vali_binary的内存大小为:{round(sys.getsizeof(data_vali_binary)/1024/1024/1024, 3)}G")
    #判断验证集的adr标签是否在范围之内
    if ~data_vali_binary['adr'].drop_duplicates().isin(data_train_binary['adr'].drop_duplicates().tolist()).all():
        print(f"第{folder}折划分的验证集中包含训练集中未知的adr")
        #移除验证集中未知的adr
        adr_all = data_vali_binary.adr.drop_duplicates().tolist()
        data_vali_binary = data_vali_binary[data_vali_binary.adr.isin(data_train_binary.adr)]
        remove_adr = list(set(adr_all) - set(data_vali_binary['adr'].drop_duplicates().tolist())) 
        # 重置索引
        data_vali_binary.reset_index(drop=True, inplace=True)
        print(f"移除的不良反应为:{remove_adr}")
        print(f"余下验证集的形状为{data_vali_binary.shape}")
        del  adr_all, remove_adr


    #训练集不良反应相似计算
    #获取训练集的不良反应的药物的联系、频率分布列表
    adr_drug_b = data_train_binary.pivot_table(index='adr', columns='drugname', values=train_data.columns[2], fill_value=0) #jaccord系数
    adr_drug_f = data_train_binary.pivot_table(index='adr', columns='drugname', values=train_data.columns[3], fill_value=0) #余弦相似分数
    
    adr_drug_f.loc[:,'adr_drug_freq'] =  adr_drug_f.values.tolist()
    adr_drug_f.loc[:,'adr_drug_binary'] =  adr_drug_b.values.tolist()
    
    adr_drug_f['adr'] = adr_drug_f.index
    adr_drug_f.reset_index(drop=True, inplace=True)
    
    train_adr_drugf = adr_drug_f.loc[:,['adr','adr_drug_freq', 'adr_drug_binary']]
    del adr_drug_b, adr_drug_f
    
    train_adr_freq = train_adr_drugf.adr_drug_freq.tolist()
    train_adr_binary = train_adr_drugf.adr_drug_binary.tolist()

    #计算不良反应的药物相似性
    train_adr_drugf.loc[:,'adr_drug_freq_similar'] = train_adr_drugf['adr_drug_freq'].apply(lambda x : calculate_tanimoto_similarity_matrix(x, train_adr_freq))
    train_adr_drugf.loc[:,'adr_drug_binary_similar'] = train_adr_drugf['adr_drug_binary'].apply(lambda x : caculate_cos_sim(x, train_adr_binary))
    del train_adr_freq, train_adr_binary

    #储存训练集每个药物与训练集药物的相似性计算,这里计算的是每个药物与训练集的所有药物之间的相似性
    #计算的是jaccard相似度或者余弦相似度，指纹相似度
    train_durg = data_train_binary[['drugname', 'Macc', 'Morgan', 'Rtoplo', 'pharmfp',
    'contextpred', 'edgepred', 'infomax', 'masking']].drop_duplicates(subset=['drugname'])
    vali_drug = data_vali_binary[['drugname', 'Macc', 'Morgan', 'Rtoplo', 'pharmfp',
    'contextpred', 'edgepred', 'infomax', 'masking']].drop_duplicates(subset=['drugname'])
    

    train_macc_list= [safe_eval(i) for i in train_durg.Macc]
    train_morgan_list= [safe_eval(i) for i in train_durg.Morgan]
    train_rtoplo_list= [safe_eval(i) for i in train_durg.Rtoplo]
    train_pharm_list= [safe_eval(i) for i in train_durg.pharmfp]

    
    #计算余弦相似性，分子嵌入相似性
    train_contextpred_list= train_durg.contextpred
    train_edgepred_list= train_durg.edgepred
    train_infomax_list= train_durg.infomax
    train_masking_list= train_durg.masking


    #计算训练集与训练集药物的相似性
    train_durg1 = train_durg.copy()
    #计算验证集各个药物与训练集药物之间的相似性
    vali_drug1 = vali_drug.copy()
    del train_durg, vali_drug

    newcolname = (['macc_similar','morgan_similar','rtoplo_similar'] + ['pharm_similar', 'contextpred_similar', 'edgepred_similar','infomax_similar','masking_similar'])*2
    oldcolname = (['Macc', 'Morgan', 'Rtoplo'] + ['pharmfp', 'contextpred','edgepred', 'infomax', 'masking'])*2
    caculate_list = ([train_macc_list,train_morgan_list, train_rtoplo_list] + [train_pharm_list,train_contextpred_list, train_edgepred_list,train_infomax_list, train_masking_list])*2

    print('开始并行计算相似性'*5)
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
            df.loc[:, newcol] = df[oldcol].apply(lambda x : caculate_cos_sim(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        elif i in (8,9,10):
            df = vali_drug1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : calculate_tanimoto_similarity_matrix(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        elif i in (11,12,13,14,15):
            df = vali_drug1[['drugname', oldcol]].copy()
            df.loc[:, newcol] = df[oldcol].apply(lambda x : caculate_cos_sim(safe_eval(x), caculate))
            return_data_binary1 = df.loc[:,['drugname', newcol]]
        del df
        gc.collect()
        return return_data_binary1,i  
    fp_list_data = Parallel(n_jobs= -1)(delayed(parall_caculate)(i) for i in range(16))
    del train_macc_list, train_morgan_list, train_rtoplo_list, train_pharm_list, train_contextpred_list, train_edgepred_list,train_infomax_list, train_masking_list,newcolname, oldcolname,caculate_list
    gc.collect()
    print(f'fp_list_data 的长度:{len(fp_list_data)}')
    print(f'fp_list_data 的第一个元素尺寸:{fp_list_data[0][0].shape}')
    print(f"fp_list_data 的内存大小为:{round(sys.getsizeof(fp_list_data)/1024/1024/1024, 3)}G")

        
# 循环遍历每个子数据集
    for i in range(16):
    # 读取子数据集
        sub_dataset = fp_list_data[i][0]
        if fp_list_data[i][1] <=7:
            train_durg1 = pd.merge(train_durg1, sub_dataset, on='drugname')
        else:
            vali_drug1 = pd.merge(vali_drug1, sub_dataset, on='drugname')
    # 将子数据集按照 id 列合并到主数据集中
    del fp_list_data
    print('并行计算相似性结束'*5)
    print(f'并行花费时间：{(time.time()-t_pata)/60}分钟')
#合并数据集
    data_train_binary1 = pd.merge(data_train_binary, train_durg1[['drugname','macc_similar','morgan_similar','rtoplo_similar','pharm_similar','contextpred_similar','edgepred_similar','infomax_similar','masking_similar']], how='left', on='drugname')
    data_train_binary1 = pd.merge(data_train_binary1, train_adr_drugf[['adr','adr_drug_freq_similar', 'adr_drug_binary_similar']], how='left', on='adr')

    data_vali_binary1 = pd.merge(data_vali_binary, vali_drug1[['drugname','macc_similar','morgan_similar','rtoplo_similar','pharm_similar','contextpred_similar','edgepred_similar','infomax_similar','masking_similar']], how='left', on='drugname')
    data_vali_binary1 = pd.merge(data_vali_binary1, train_adr_drugf[['adr','adr_drug_freq_similar', 'adr_drug_binary_similar']], how='left', on='adr')
    print(f"data_train_binary1 的内存大小为:{round(sys.getsizeof(data_train_binary1)/1024/1024/1024, 3)}G")
    print(f"data_vali_binary1 的内存大小为:{round(sys.getsizeof(data_vali_binary1)/1024/1024/1024, 3)}G")

    del data_train_binary, data_vali_binary
    
    
    print('开始转化为np'*5)
    t_np = time.time()
    np_index = (['association', 'freq', 'adr_similar',
       'Macc', 'Morgan', 'Rtoplo', 'pharmfp', 'contextpred',
       'edgepred', 'infomax', 'masking', 'macc_similar', 'morgan_similar',
       'rtoplo_similar', 'pharm_similar', 'contextpred_similar',
       'edgepred_similar', 'infomax_similar', 'masking_similar',
       'adr_drug_freq_similar', 'adr_drug_binary_similar'])*2
    def para_convert_np(xx):
        col = np_index[xx]
        if xx <=20:#train
            if col in [0,1]:#drugname and adrname
                re_np = np.array(data_train_binary1.loc[:,col])
            else:#other features
                re_np = np.array([safe_eval(i) for i in data_train_binary1.loc[:, col]])
        else:#vali
            if col in [0,1]:#drugname and adrname
                re_np = np.array(data_vali_binary1.loc[:,col])
            else:#others features
                re_np = np.array([safe_eval(i) for i in data_vali_binary1.loc[:, col]])
        gc.collect()
        return re_np,col,xx
    
        # 调试信息
    def debug_parallel_task(xx):
        try:
            result = para_convert_np(xx)
            print(f"Task {xx} completed successfully.")
            return result
        except Exception as e:
            print(f"Task {xx} failed with exception: {e}")
            raise
    # 并行计算
    re_np_list = Parallel(n_jobs=-1)(delayed(debug_parallel_task)(xx) for xx in range(42))


    print(f're_np_list 的长度:{len(re_np_list)}')
    print(f"re_np_list 的内存大小为:{round(sys.getsizeof(re_np_list)/1024/1024/1024, 3)}G")
    print(re_np_list[0][0].shape)
    print(f"{sys.getsizeof(re_np_list[0][0])/1024/1024/1024}G")
    print(f'转化为np结束用了{(time.time()-t_np)/60}分钟')
    
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
                print("转化为np的时候有点问题pre_data")
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


    train_dataset = Mydataset(Macc_train, morgan_train, Rtoplo_train, pharm_train,   contextpred_train,  edgepred_train, infomax_train, masking_train, macc_similar_train, morgan_similar_train, rtoplo_similar_train,  pharm_similar_train, contextpred_simila_train, edgepred_similar_train, infomax_similar_train,  masking_similar_train, train_adr, train_adr_1, train_adr_2, train_label, train_label_f)
    
    vail_dataset = Mydataset(Macc_vali, morgan_vali, Rtoplo_vali, pharm_vali, contextpred_vali, edgepred_vali,  infomax_vali, masking_vali, macc_similar_vali, morgan_similar_vali, rtoplo_similar_vali, pharm_similar_vali,  contextpred_simila_vali, edgepred_similar_vali, infomax_similar_vali, masking_similar_vali, vali_adr, vali_adr_1, vali_adr_2, vali_label, vali_label_f)
    
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch, num_workers=8,shuffle=True)
    vail_loader = DataLoader(dataset=vail_dataset,batch_size=batch, num_workers=8,shuffle=True)
    
    print(f"Macc_train的形状:{Macc_train.shape}")
    print(f"train_adr的形状:{train_adr.shape}")
    print(f"train_adr_1的形状:{train_adr_1.shape}")
    print(f"train_adr_2的形状:{train_adr_2.shape}")
    print(f"morgan_similar_train 的形状: {morgan_similar_train.shape} 所占内存大小为：{round(sys.getsizeof(morgan_similar_train)/1024/1024/1024, 3)}G")
    print(f"train_dataset 所占内存大小为：{round(sys.getsizeof(train_dataset)/1024/1024/1024, 3)}G")
    print(f"vail_dataset 所占内存大小为：{round(sys.getsizeof(vail_dataset)/1024/1024/1024, 3)}G")
    print(f"train_loader 所占内存大小为：{round(sys.getsizeof(train_loader)/1024/1024/1024, 3)}G")
    print(f"vail_loader 所占内存大小为：{round(sys.getsizeof(vail_loader)/1024/1024/1024, 3)}G")
    
    del Macc_train, morgan_train, Rtoplo_train, pharm_train, contextpred_train, edgepred_train, infomax_train, masking_train, macc_similar_train, morgan_similar_train, rtoplo_similar_train, pharm_similar_train, contextpred_simila_train, edgepred_similar_train, infomax_similar_train, masking_similar_train, train_adr, train_adr_1, train_adr_2, train_label, train_label_f, Macc_vali, morgan_vali, Rtoplo_vali, pharm_vali, contextpred_vali, edgepred_vali,  infomax_vali, masking_vali, macc_similar_vali, morgan_similar_vali, rtoplo_similar_vali, pharm_similar_vali,  contextpred_simila_vali, edgepred_similar_vali, infomax_similar_vali, masking_similar_vali, vali_adr, vali_adr_1, vali_adr_2, vali_label, vali_label_f, train_dataset, vail_dataset
    gc.collect()
    return train_loader,vail_loader

def train(model, loader, index, device, optimizer, loss_batch_train, loss_1, loss_2, b1, choose_mode:str = 'all'):
    trainy_pre_b = np.empty((0,1))
    trainy_true_b = np.empty((0,1))
    
    trainy_pre_f = np.empty((0,1))
    trainy_true_f = np.empty((0,1))
    
    batch_n = 0
    for macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2, label_1, label_2 in loader: #lable1_1:binary, lable1_2:freq
        
        batch_n +=1
        if batch_n % 100 == 0:
            print(f"第{index}个epoch中的第{batch_n}个batch")
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
        # 模型训练
        #一起训练
        for name, value in model.named_parameters():
            value.requires_grad=True
        if choose_mode == 'all':
            loss, output_b, output_f, label_1,label_2 = opti(model, optimizer, criterion1=loss_1,criterion2=loss_2, target1=label_1, target2=label_2, a=b1, inputs=inputs)
        #微调二分类的mlpb
        if choose_mode == 'binary':
             # 确保所有参数 requires_grad 属性设置正确
            for name, value in model.named_parameters():
                value.requires_grad = False
            for name, value in model.named_parameters():
                if 'com_layerb.combine_layerf' in name:
                    value.requires_grad = True
            loss, output_b, output_f, label_1,label_2 = opti(model, optimizer, criterion1=loss_1, target1=label_1, target2=label_2, a=b1, inputs=inputs)

        #微调频率预测的mlpf
        if choose_mode == "freq":
            # 确保所有参数 requires_grad 属性设置正确
            for name, value in model.named_parameters():
                value.requires_grad = False
            for name, value in model.named_parameters():
                if 'com_layerf.layers' in name:
                    value.requires_grad = True
            loss, output_b, output_f, label_1, label_2 = opti(model, optimizer, criterion2=loss_2, target1=label_1, target2=label_2, a=b1, inputs=inputs)
  

        # output = torch.softmax(output, dim=1)

        output_b_1 = torch.sigmoid(output_b)
        output_b=output_b_1.cpu().detach().numpy()
        output_f=output_f.cpu().detach().numpy()

        label_1 = label_1.cpu().detach().numpy()
        label_2 = label_2.cpu().detach().numpy()
        
        
        trainy_pre_b=np.concatenate((trainy_pre_b,output_b),axis=0)
        trainy_pre_f=np.concatenate((trainy_pre_f,output_f),axis=0)
        
        trainy_true_b=np.concatenate((trainy_true_b,label_1),axis=0) 
        trainy_true_f=np.concatenate((trainy_true_f,label_2),axis=0) 
        
        
        loss_score_1=loss.cpu().detach() #detach是将数据从计算图上面分离出来
        loss_batch_train.append(loss_score_1.item()) 
        
        del macc_1, morgan_1, rtoplo_1, pharm_1, contextpred_1, edgepred_1, infomax_1, masking_1, similar_macc_1, similar_morgan_1, similar_rtoplo_1, similar_pharm_1, similar_contextpred_1, similar_edgepred_1, similar_infomax_1, similar_masking_1, adr_1, adr1_1, adr1_2,loss_score_1
        
        gc.collect()

    #  trainy_pre #np
    #  trainy_true #np
    auc_train = roc_auc_score(trainy_true_b, trainy_pre_b)
    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(trainy_true_b, trainy_pre_b)
    
    aupr_train = auc(recall, precision)

    
    #freq评价指标，只计算阳性的损失大小
    rmse_1 = sqrt(mean_squared_error(trainy_true_f[trainy_true_b==1], trainy_pre_f[trainy_true_b==1])) #标量
    mae_1 = mean_absolute_error(trainy_true_f[trainy_true_b==1], trainy_pre_f[trainy_true_b==1]) #标量
    speaman_1 = stats.spearmanr(trainy_true_f[trainy_true_b==1], trainy_pre_f[trainy_true_b==1])

    
    print(f"epoch{index}训练集的AUC:{auc_train},AUPR:{aupr_train},RMSE:{rmse_1},MAE:{mae_1},Spearman:{speaman_1}")
    return auc_train,aupr_train,rmse_1, mae_1 ,speaman_1

def evaluate(model, loader, index, device, loss_epoch_vali, loss_1, loss_2, b1):
    with torch.no_grad():  # 在评估模式下不计算梯度
        model.eval()  # 将模型设置为评估模式
        vali_true_f = np.empty((0,1))
        vali_pre_f = np.empty((0,1))
        
        vali_true_b = np.empty((0,1))
        vali_pre_b = np.empty((0,1))
        
        for macc_v, morgan_v, rtoplo_v, pharm_v, contextpred_v, edgepred_v, infomax_v, masking_v, similar_macc_v, similar_morgan_v, similar_rtoplo_v, similar_pharm_v, similar_contextpred_v, similar_edgepred_v, similar_infomax_v, similar_masking_v, adr_v, adr1_v, adr2_v, label_v, labe2_v in loader:     
            
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

        # loss_binary = loss_1(output_b, label_1)

        # loss_freq = loss_2(output_f[label_1 == 1], label_2[label_1 == 1]) #只计算阳性样本的freq损失
        
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

        print(f"epoch{index}验证集的AUC:{auc_v},AUPR:{aupr_v},RMSE:{rmse_v},MAE:{mae_v},Spearman:{spearman_v}")
        return auc_v,aupr_v,rmse_v, mae_v,spearman_v
if __name__ == "__main__":
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #保证可重复性
    seed_value = 3407   # 设定随机数种子
    np.random.seed(seed_value)
    # random.seed(seed_value)
    # os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

    path = "./dataset"
    file_name_pps = "PPS"
    file_name_nns = "NNS_all"
    file_type = ".csv"
    columns = ['drugname','adr','freq','smiles','adr_similar', 'Macc','pubchem','Morgan','Rtoplo','macc_simi', 'morgan_simi', 'rtoplo_simi', 'com_simi', 'association']
    pps = load_binary(path, file_name_pps, file_type, columns) #加载正例集合
    nns = load_binary(path, file_name_nns, file_type, columns) #加载未知得集合 
    
    #将pps和nns数据集进行合并然后划分训练集和验证集
    data_binary = pd.concat((pps, nns), ignore_index=True)

    del nns, pps
    gc.collect()
    
    file_name = "data6"
    data_freq = load_data(path, file_name, file_type)

    #将freq匹配过来，以药物不良反应对为id，缺失的以0填充
    data_binary1 = pd.merge(data_binary, data_freq[['drugname', 'adr', 'freq']], 'left', on=['drugname', 'adr'])
    data_binary1.freq.fillna(0, inplace=True)

    #得到一个计算用的药物数据集，用以计算每个药物的各种特征
    drug_smiles=data_freq[['drugname', 'smiles', 'Macc', 'pubchem','Morgan', 'Rtoplo', 'pharmfp']]
    drug_smiles1 = drug_smiles.drop_duplicates().copy()

    #计算用的adr数据集
    _adr=data_freq[['adr', 'adr_similar']]
    _adr1 = _adr.drop_duplicates().copy()
    
    del data_freq, drug_smiles, data_binary, _adr
    gc.collect()
    
    _data_binary = data_binary1.merge(drug_smiles1[['drugname', 'smiles']], 'left', on='drugname')
    _data_binary = _data_binary.merge(_adr1, 'left', on='adr')


    #计算药物分子的四种gin嵌入特征,由于在计算的时候设定了shuffle=False所以输出的嵌入特征与输入的smiles特征顺序一致
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
    #保存评价指标为字典，好方便直接tocsv
    metrics_res = {"dataset": [], "folder":[],"best_epoch": [],"auc_max": [],"aupr_maxauc": [],"rmse_maxauc": [],"spearman_maxauc":[], "epoch_maxaupr":[],"aupr_max":[],"auc_maxaupr":[],"rmse_maxaupr":[],"spearman_maxaupr":[],"epoch_minrmse":[], "rmse_min":[],"auc_minrmse":[],"aupr_minrmse":[],"spearman_minrmse":[]}
    #冷启动
    drug_data = data_binary2.loc[:,['drugname']].drop_duplicates().reset_index(drop=True)

    #gradually
    #分成10份，用其中一份验证，然后评估随着训练的药物越来越多，对这部分的药物的预测性能
    drug_data.loc[:, 'g'] = drug_data.apply(lambda x: random.randint(0,9), axis=1)

    for idx, gg in enumerate(range(9)):
        # if idx==3:
        #     break
        #g=9的药物用以验证集
        vali_drug = drug_data[drug_data['g'] == 9]
        number_vali = vali_drug.shape[0]
        train_drug = drug_data[drug_data['g'] <= gg]
        number_train = train_drug.shape[0]
        drug_train_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(train_drug['drugname'])].index])
        drug_vali_rows = np.array([data_binary2.index.get_loc(i) for i in data_binary2[data_binary2['drugname'].isin(vali_drug['drugname'])].index])

        print(f"训练集药物:{train_drug.shape[0]}")
        print(f"验证集药物:{vali_drug.shape[0]}")

        train_data = data_binary2.iloc[drug_train_rows, :].reset_index(drop=True)
        vali_data = data_binary2.iloc[drug_vali_rows, :].reset_index(drop=True)
        #对训练集进行采样
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
        train_loader,vail_loader = pre_data(train_data_sample, vali_data, 128, folder)
        
        print(f"train_loader所占内存大小为:{sys.getsizeof(train_loader)/1024/1024/1024}G")
        print(f"vail_loader所占内存大小为:{sys.getsizeof(vail_loader)/1024/1024/1024}G")

        feature_size = 256
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
                        print("网络bias的初始化失败了")
                    else:
                        pass
        model.to(device)

        loss_1 = nn.BCEWithLogitsLoss()
        loss_2 = nn.MSELoss()
        #all 100 epoch
        for epoch in range(15):
            print("-"*20+'all'+'-'*20)
            print(f'**********{epoch+1}*********')
            # weight
            weitght = 0.95
            optimizerall = optim.Adam(model.parameters(), lr=0.00003707,weight_decay=0.005)
            model.train()
            _,_,_, _,_ = train(model, train_loader, epoch, device, optimizerall, loss_batch_train, loss_1, loss_2, weitght, choose_mode='all')
            # for i,j in model.named_parameters():
            #     if i == "map_macc.map_layer.0.weight":
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerb.combine_layerf.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerf.layers.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}")
        #微调 binary 10 epoch
        for epoch in range(20):
            print("-"*20+'binary'+'-'*20)
            weitght = 1
            optimizerbinary = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00003707, weight_decay=0.005)
            model.train()
            _,_,_,_,_ = train(model, train_loader, epoch, device, optimizerbinary, loss_batch_train, loss_1, loss_2, weitght, choose_mode='binary')
            # for i,j in model.named_parameters():
            #     if i == "map_macc.map_layer.0.weight":
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerb.combine_layerf.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerf.layers.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}") 
        #微调 freq 10 epoch
        for epoch in range(15):
            print("-"*20+'freq'+'-'*20)
            weitght = 1
            optimizerfreq = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00003707, weight_decay=0.005)
            model.train()
            auc_1,aupr_1,rmse_1, mae_1,spearman_1 = train(model, train_loader, epoch, device, optimizerfreq, loss_batch_train, loss_1, loss_2, weitght, choose_mode='freq') 
            # for i,j in model.named_parameters():
            #     if i == "map_macc.map_layer.0.weight":
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerb.combine_layerf.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            #     elif "com_layerf.layers.0" in i:
            #         print(f"{i}的参数的和:{torch.sum(j)}")
            model.eval()
            auc_v, aupr_v, rmse_v, mae_v, spearman_v= evaluate(model, vail_loader, epoch, device, loss_epoch_vali, loss_1, loss_2, weitght)
            
            #保存验证集auc最大的那一轮的模型
            if auc_v > auc_max:
                auc_max = auc_v
                aupr_maxauc = aupr_v
                rmse_maxauc = rmse_v
                spearman_maxauc = spearman_v
                auc_max_train = auc_1
                aupr_maxauc_train = aupr_1
                rmse_maxauc_train = rmse_1
                spearman_maxauc_train = spearman_1        
                epoch_maxauc = epoch
                # model_auc = copy.deepcopy(model)
                
            if aupr_v > aupr_max:
                aupr_max = aupr_v
                auc_maxaupr = auc_v
                rmse_maxaupr = rmse_v
                spearman_maxaupr = spearman_v
                aupr_max_train = aupr_1
                auc_maxaupr_train = auc_1
                rmse_maxaupr_train = rmse_1
                spearman_maxaupr_train = spearman_1
                epoch_maxaupr = epoch
                # model_aupr = copy.deepcopy(model)
            
            if rmse_v < rmse_min:
                rmse_min = rmse_v
                auc_minrmse = auc_v
                aupr_minrmse = aupr_v
                spearman_minrmse = spearman_v
                rmse_min_train = rmse_1
                auc_minrmse_train = auc_1
                aupr_minrmse_train = aupr_1
                spearman_minrmse_train = spearman_1
                epoch_minrmse = epoch
                # model_rmse = copy.deepcopy(model)

                
        #保存模型评价指标
        metrics_res['dataset'].append('train')
        metrics_res['folder'].append(folder)
        metrics_res["best_epoch"].append(epoch_maxauc),
        metrics_res["auc_max"].append(auc_max_train)
        metrics_res["aupr_maxauc"].append(aupr_maxauc_train)
        metrics_res["rmse_maxauc"].append(rmse_maxauc_train)
        metrics_res["spearman_maxauc"].append(spearman_maxauc_train) 
        metrics_res["epoch_maxaupr"].append(epoch_maxaupr)
        metrics_res["aupr_max"].append(aupr_max_train)
        metrics_res["auc_maxaupr"].append(auc_maxaupr_train)
        metrics_res["rmse_maxaupr"].append(rmse_maxaupr_train)
        metrics_res["spearman_maxaupr"].append(spearman_maxaupr_train)
        metrics_res["epoch_minrmse"].append(epoch_minrmse) 
        metrics_res["rmse_min"].append(rmse_min_train)
        metrics_res["auc_minrmse"].append(auc_minrmse_train)
        metrics_res["aupr_minrmse"].append(aupr_minrmse_train)
        metrics_res["spearman_minrmse"].append(spearman_minrmse_train)
        #验证集
        metrics_res['dataset'].append('vali')
        metrics_res['folder'].append(folder)
        metrics_res["best_epoch"].append(epoch_maxauc),
        metrics_res["auc_max"].append(auc_max)
        metrics_res["aupr_maxauc"].append(aupr_maxauc)
        metrics_res["rmse_maxauc"].append(rmse_maxauc) 
        metrics_res["spearman_maxauc"].append(spearman_maxauc) 
        metrics_res["epoch_maxaupr"].append(epoch_maxaupr)
        metrics_res["aupr_max"].append(aupr_max)
        metrics_res["auc_maxaupr"].append(auc_maxaupr)
        metrics_res["rmse_maxaupr"].append(rmse_maxaupr)
        metrics_res["spearman_maxaupr"].append(spearman_maxaupr)
        metrics_res["epoch_minrmse"].append(epoch_minrmse) 
        metrics_res["rmse_min"].append(rmse_min)
        metrics_res["auc_minrmse"].append(auc_minrmse)
        metrics_res["aupr_minrmse"].append(aupr_minrmse)
        metrics_res["spearman_minrmse"].append(spearman_minrmse)


        
        del train_loader, vail_loader, model, loss_1, loss_2
        gc.collect()

        print(f"一折的时间为：{(time.time()-t0)/60}分钟")
    # torch.save(model_auc.state_dict(), f'./result/model/dicit_model{folder}.pth')
    # torch.save(model_auc, f'./result/model/whole_model{folder}.pth')
    # torch.save(model_aupr.state_dict(), f'./result/model/seperate_model_aupr{folder}.th')
    # torch.save(model_aupr, f'./result/model/seperate_model_aupr{folder}.pkl')
    # torch.save(model_rmse.state_dict(), f'./result/model/seperate_model_rmse{folder}.th')
    # torch.save(model_rmse, f'./result/model/seperate_model_rmse{folder}.pkl')
    pd.DataFrame(metrics_res).to_csv("./result/gradually.csv")
    del data_binary2
    gc.collect()