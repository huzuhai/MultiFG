import torch
import numpy as np
import torch.nn.functional as F

from torch import nn


#map layer
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

# feature extracting layer
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

#prediction layer
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

#model
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