original data:所有数据的原始来源文件
dataset: 训练模型需要的数据集
CV10: 10折交叉验证，运行之后会得到10个model，10个vali loader，一个csv文件，保存的是训练集和验证集的模型评价指标(部分)
Cold_CV10: 冷启动下的10折交叉验证，运行之后会得到10个 model_cold，10个 vali loader_cold，一个csv文件保存的是训练集和验证集的模型评价指标(部分)
G_test: 模型性能随着训练集药物数量的变化情况，运行之后会得到一个csv文件,保存的是模型的训练集和验证集评价指标
vali:用以计算CV10、Cold_CV10场景下验证集的模型评价指标
