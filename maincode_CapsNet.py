##############这个代码的主要思想是BioBERT+REGCN+CapsNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense import Linear as DenseLinear
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AdamW
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.cuda
import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dgl
import numpy as np
import torch as th
from dgl.nn import RelGraphConv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from torch.nn import MultiheadAttention
import spacy
import scispacy 


# --------------------------------------------------------------------------- #
# 1. 决策：句法分析器和依赖类型处理
# --------------------------------------------------------------------------- #
# 尝试加载推荐的科学模型，如果失败则回退到通用模型
SPACY_MODEL_NAMES = ['en_core_sci_lg', 'en_core_web_lg', 'en_core_web_sm']
nlp = None
for model_name in SPACY_MODEL_NAMES:
    try:
        nlp = spacy.load(model_name)
        print(f"Successfully loaded spaCy model: {model_name}")
        break
    except OSError:
        print(f"spaCy model '{model_name}' not found. Trying next...")
if nlp is None:
    print(f"Fatal: Could not load any of the specified spaCy models: {SPACY_MODEL_NAMES}")
    print("Please install one, e.g., 'python -m spacy download en_core_web_sm' or a scientific model.")
    exit()

# 定义一个常见的句法依赖类型到整数ID的映射
# 您可以根据您的数据和需求扩展或修改这个列表
# 这些类型将用于RelGraphConv的边类型 (etypes)
# 增加一个 'SELF_LOOP' 类型和一个 'UNKNOWN_DEP' 类型
COMMON_DEP_TYPES = [
    # 主谓宾核心关系
    'nsubj', 'dobj', 'iobj', 'csubj', 'xsubj', 'agent', 'attr',
    # 名词修饰
    'amod', 'nmod', 'appos', 'nummod', 'compound',
    # 动词/从句修饰
    'advmod', 'advcl', 'aux', 'cop', 'mark',
    # 其他重要关系
    'conj', 'cc', 'prep', 'pobj', 'acl', 'relcl',
    # DDI中可能相关的
    'dep', # 通用依赖，当没有更精确的标签时
]
# 创建映射，为未知类型和自环类型留出ID
DEP_TYPE_TO_ID = {dep: i for i, dep in enumerate(COMMON_DEP_TYPES)}
SELF_LOOP_ETYPE_ID = len(DEP_TYPE_TO_ID)
UNKNOWN_DEP_ETYPE_ID = len(DEP_TYPE_TO_ID) + 1
NUM_DEP_TYPES = len(DEP_TYPE_TO_ID) + 2 # +2 for SELF_LOOP and UNKNOWN_DEP

print(f"Number of unique dependency relation types (num_rels for GCN): {NUM_DEP_TYPES}")
print(f"Self-loop edge type ID: {SELF_LOOP_ETYPE_ID}")
print(f"Unknown dependency edge type ID: {UNKNOWN_DEP_ETYPE_ID}")


#############################################定义 BERT 模型和 tokenizer##############################################

#导入Biobert
model_path = './model_path/biobert'                     #这个要用相对路径，不要用绝对路径
biobert_tokenizer = AutoTokenizer.from_pretrained(model_path)
biobert_model = AutoModel.from_pretrained(model_path)


# #导入bert
# model_path_1 = './model_path/bert_pretrain'                     #这个要用相对路径，不要用绝对路径
# bert_tokenizer = AutoTokenizer.from_pretrained(model_path_1)
# bert_model = AutoModel.from_pretrained(model_path_1)



####################################################################################################################

#############################################读取数据################################################################

df_train = pd.read_csv('./data/ddi2013ms/train.tsv', sep='\t')
df_dev = pd.read_csv('./data/ddi2013ms/dev.tsv', sep='\t')
df_test = pd.read_csv('./data/ddi2013ms/test.tsv', sep='\t')
print("read")

# print("训练集数据量：", df_train.shape)
# print("验证集数据量：", df_dev.shape)
# print("测试集数据量：", df_test.shape)

####################################################################################################################

#######################################################定义模型参数##################################################
#定义训练设备，默认为GPU，若没有GPU则在CPU上训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
print(f"Using device: {device}") # <--- 加入这行

num_label=5

# 定义模型参数
max_length = 300
batch_size = 32


# #############################################定义数据集和数据加载器###################################################
# # 定义数据集类
# 定义标签到整数的映射字典
label_map = {
    'DDI-false': 0,
    'DDI-effect': 1,
    'DDI-mechanism': 2,
    'DDI-advise': 3,
    'DDI-int': 4
    # 可以根据你的实际标签情况添加更多映射关系
}

# 定义数据集类
class DDIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_len): # Renamed max_length to max_seq_len for clarity
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def _build_dependency_graph(self, sentence_text, encoding):
        # encoding: Hugging Face tokenizer output
        # sentence_text: The original sentence string

        adj_matrix = np.zeros((self.max_seq_len, self.max_seq_len), dtype=np.float32)
        # edge_types_matrix用于存储每条边的类型，DGL的RelGraphConv可以直接使用一个边类型列表
        # 我们这里先创建一个矩阵，方便构建，之后再转换成DGL需要的格式 (src, dst, etype list)
        # 或者直接构建 (src_nodes, dst_nodes, edge_types) 列表
        # 为了简化，直接在adj_matrix上构建，并在create_dgl_graph中提取边和类型
        # 此处，我们将返回一个更丰富的结构，包含边和它们的类型
        
        src_nodes, dst_nodes, edge_types = [], [], []

        # 1. 使用 spaCy 解析句子
        doc = nlp(sentence_text)

        # 2. 获取 BioBERT 的 word_ids 来对齐 spaCy 词元和 BioBERT 子词
        # word_ids() 会为每个子词给出其所属原始词的索引，特殊token为None
        # 需要确保tokenizer调用时已启用（新版transformers默认启用，旧版可能需手动加参数）
        try:
            word_ids = encoding.word_ids()
        except AttributeError:
            # 如果 word_ids 不可用 (例如，旧版transformers)，需要回退到 offset_mapping
            # print("Warning: encoding.word_ids() not available. Graph construction might be less accurate.")
            # print("Consider upgrading transformers library or implementing offset_mapping based alignment.")
            # For now, returning an empty graph placeholder if word_ids is not available
            # return adj_matrix, [], [], [] # adj_matrix, src_nodes, dst_nodes, edge_types
             # Fallback or raise error: For this example, let's assume word_ids is available.
             # In a real scenario, you'd implement the offset_mapping fallback.
             raise ValueError("encoding.word_ids() is required. Please check your transformers version or tokenizer call.")


        # 3. 创建从原始词索引到其在BioBERT中的第一个子词索引的映射
        # BioBERT的input_ids[0] 通常是 [CLS]
        word_idx_to_first_subword_idx = {}
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in word_idx_to_first_subword_idx:
                word_idx_to_first_subword_idx[word_id] = i # i 是子词在 tokenized_output.ids 中的索引

        # 4. 遍历 spaCy 的依赖关系，填充邻接信息和边类型列表
        for spacy_token in doc:
            # 获取当前词 (依赖词) 和其头词在原始句子中的索引
            src_word_idx = spacy_token.i
            dst_word_idx = spacy_token.head.i
            dep_type = spacy_token.dep_.lower() # 依赖关系类型

            # 获取对应的BioBERT子词索引
            # (word_idx_to_first_subword_idx 已经包含了正确的子词索引，无需+1 for [CLS])
            subword_idx_src = word_idx_to_first_subword_idx.get(src_word_idx)
            subword_idx_dst = word_idx_to_first_subword_idx.get(dst_word_idx)

            if subword_idx_src is not None and subword_idx_dst is not None:
                # 确保索引在邻接矩阵的有效范围内 (0 to max_seq_len-1)
                if subword_idx_src < self.max_seq_len and subword_idx_dst < self.max_seq_len:
                    # 添加边 (无向)
                    adj_matrix[subword_idx_src, subword_idx_dst] = 1.0
                    adj_matrix[subword_idx_dst, subword_idx_src] = 1.0

                    # 获取边类型ID
                    etype_id = DEP_TYPE_TO_ID.get(dep_type, UNKNOWN_DEP_ETYPE_ID)
                    
                    # 添加到边列表 (DGL可以直接使用这些)
                    src_nodes.extend([subword_idx_src, subword_idx_dst])
                    dst_nodes.extend([subword_idx_dst, subword_idx_src]) # 无向图，两条边
                    edge_types.extend([etype_id, etype_id]) # 假设关系类型是对称的

        # 5. 添加自环 (为每个实际参与图的token添加)
        # 自环的类型使用 SELF_LOOP_ETYPE_ID
        # word_ids 包含了 None 对于特殊 tokens ([CLS], [SEP])
        # 我们只为实际的词（word_id is not None）对应的第一个子词添加自环
        processed_subword_indices_for_self_loop = set()
        for word_id_val, subword_idx_val in word_idx_to_first_subword_idx.items():
            if subword_idx_val < self.max_seq_len and subword_idx_val not in processed_subword_indices_for_self_loop:
                adj_matrix[subword_idx_val, subword_idx_val] = 1.0
                src_nodes.append(subword_idx_val)
                dst_nodes.append(subword_idx_val)
                edge_types.append(SELF_LOOP_ETYPE_ID)
                processed_subword_indices_for_self_loop.add(subword_idx_val)
        
        return adj_matrix, torch.tensor(src_nodes, dtype=torch.long), \
               torch.tensor(dst_nodes, dtype=torch.long), \
               torch.tensor(edge_types, dtype=torch.long)


    def __getitem__(self, idx):
        sentence = str(self.data['sentence'][idx])
        label_str = self.data['label'][idx]
        label = label_map[label_str]

        # Tokenize with BioBERT. Ensure word_ids are generated.
        # padding='max_length' and truncation=True are crucial.
        encoding = self.tokenizer(
            sentence,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
            # For word_ids(), no specific param needed if transformers version is recent.
            # For offset_mapping fallback, add: return_offsets_mapping=True
        )
        
        # Flatten input_ids and attention_mask if they are [1, seq_len]
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        adj_matrix, src_nodes, dst_nodes, graph_edge_types = self._build_dependency_graph(sentence, encoding)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long),
            'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32), # For inspection or non-DGL GCN
            'graph_src_nodes': src_nodes,   # For DGL graph construction
            'graph_dst_nodes': dst_nodes,   # For DGL graph construction
            'graph_edge_types': graph_edge_types # For DGL RelGraphConv
        }
        

def custom_collate_fn(batch_samples):
    # batch_samples 是一个列表，其中每个元素都是 DDIDataset.__getitem__ 返回的字典
    collated_batch = {}
    # 假设所有样本都有相同的键
    if not batch_samples: # 如果批次为空
        return collated_batch
        
    keys = batch_samples[0].keys()

    for key in keys:
        if key in ['graph_src_nodes', 'graph_dst_nodes', 'graph_edge_types']:
            # 对于图结构张量 (它们的长度可能在不同样本间变化)，
            # 我们将它们收集到一个 Python 列表中。
            # GCNBranch 的 forward 方法会逐个处理这些列表中的张量。
            collated_batch[key] = [sample[key] for sample in batch_samples]
        elif key == 'adj_matrix': # adj_matrix 是固定大小的 (max_seq_len, max_seq_len)
            collated_batch[key] = torch.stack([sample[key] for sample in batch_samples])
        else: 
            # 其他如 input_ids, attention_mask, labels 是固定大小的，可以直接堆叠
            try:
                collated_batch[key] = torch.stack([sample[key] for sample in batch_samples])
            except Exception as e:
                print(f"Error stacking key '{key}': {e}")
                # 如果出现问题，也放入列表，方便调试
                collated_batch[key] = [sample[key] for sample in batch_samples]
    return collated_batch



# 定义数据加载器
def create_data_loader(df, tokenizer, max_seq_len, batch_size): # 参数名已统一为 max_seq_len
    dataset = DDIDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # 通常训练时 shuffle=True, 测试时 shuffle=False
        drop_last=True,
        collate_fn=custom_collate_fn # <--- 在这里添加自定义的 collate_fn
    )

# # 加载数据集和数据加载器
# 注意，之前这里用的是全局变量 max_length，现在作为参数传递
train_data_loader = create_data_loader(df_train, biobert_tokenizer, max_length, batch_size) # max_length 仍是全局变量 300
dev_data_loader = create_data_loader(df_dev, biobert_tokenizer, max_length, batch_size)
test_data_loader = create_data_loader(df_test, biobert_tokenizer, max_length, batch_size)

# for batch in test_data_loader:
#     print(batch)
#     break  # 这将打印第一批数据并中断循环。




# SharedEncoder
class SharedEncoder(nn.Module):
    def __init__(self, freeze_bert=True): # 移除了 hidden_dim 和 num_layers 参数
        super(SharedEncoder, self).__init__()
        self.bert = biobert_model # biobert_model 是全局加载的
        # SharedEncoder 的输出维度现在是 BioBERT 的隐藏层大小
        self.output_dim = self.bert.config.hidden_size # 例如，对于 biobert-base 是 768

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0] # (batch_size, max_seq_len, bert_hidden_size)
        # 不再有 LSTM 和额外的线性层、dropout
        return sequence_output


# 将节点特征和关系矩阵转换为 DGL 图对象
def create_dgl_graph(node_features, src_nodes, dst_nodes, edge_types, max_seq_len, device):
    """
    使用预先计算的源节点、目标节点和边类型创建DGL图。

    参数:
        node_features (torch.Tensor): 节点的特征张量，形状为 (num_nodes_in_sequence, feature_dim)。
                                      num_nodes_in_sequence 应该等于 max_seq_len。
        src_nodes (torch.Tensor):     源节点列表。
        dst_nodes (torch.Tensor):     目标节点列表。
        edge_types (torch.Tensor):    边类型列表。
        max_seq_len (int):            序列的最大长度，即图中节点的数量。
        device (torch.device):        图和特征所在的设备。

    返回:
        dgl.DGLGraph: 构建好的DGL图对象。
    """
    num_nodes = max_seq_len # 图中的节点总数等于序列的最大长度

    # 确保输入的张量在正确的设备上
    src_nodes = src_nodes.to(device)
    dst_nodes = dst_nodes.to(device)
    edge_types = edge_types.to(device)
    node_features = node_features.to(device)

    # 创建 DGL 图对象
    # 检查是否有边，DGL要求在没有边的情况下使用空张量创建图
    if src_nodes.numel() == 0: # numel() 返回张量中元素的总数
        g = dgl.graph((torch.tensor([], dtype=torch.long, device=device), 
                       torch.tensor([], dtype=torch.long, device=device)), 
                      num_nodes=num_nodes)
    else:
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)

    # 将节点特征添加到图中
    # 确保 node_features 的第一维与图的节点数匹配
    if node_features.shape[0] != num_nodes:
        # 如果不匹配，可能需要调整node_features的padding或截断，
        # 但理想情况下，DDIDataset处理后的node_features (例如来自BERT的输出) 应该已经是(max_seq_len, hidden_dim)
        # 这里我们假设它是匹配的，或者上游模块 (如 SharedEncoder) 会确保这一点。
        # 如果不匹配，并且node_features少于num_nodes，可以考虑填充零；如果多于，则截断。
        # 但更稳妥的是确保输入时就对齐。
        # 为了安全，如果实际的node_features长度（通常是BERT token数，包括CLS/SEP）小于max_seq_len，
        # 而图的num_nodes是max_seq_len，需要确保特征也扩展到max_seq_len。
        # 假设node_features已经正确处理为 (max_seq_len, feature_dim)
        pass # 假设node_features的维度已经是 (max_seq_len, feature_dim)

    g.ndata['feat'] = node_features

    # 添加边类型 (如果存在边)
    if g.num_edges() > 0:
        g.edata['etype'] = edge_types
    # else: # 如果没有边，DGL图的 g.edata 也是空的，不需要显式设置

    # 注意：我们之前在 DDIDataset._build_dependency_graph 中已经添加了自环和对应的边类型。
    # 因此，这里不需要再使用 dgl.add_self_loop(g)，除非你想添加不同类型的、或额外的自环。
    # 如果 DDIDataset 中没有处理自环，那么在这里 dgl.add_self_loop(g) 就是必要的。
    # 鉴于我们已在数据准备阶段加入自环，此处不再重复添加。

    return g.to(device) #确保图在正确的设备上

# --------------- BEGIN: 定义 CapsuleNetwork --------------- #
class CapsuleNetwork(nn.Module):
    def __init__(self, input_dim, # 来自 SharedEncoder (biobert_hidden_size, 例如 768)
                 seq_len,         # 序列最大长度 (max_length, 例如 300)
                 num_primary_caps_types=32, # 主胶囊的类型数量
                 primary_caps_dim=8,        # 每个主胶囊的维度
                 num_output_capsules=num_label, # 输出胶囊的数量 (可以设为类别数，或一个固定的超参数)
                 output_capsule_dim=16,     # 每个输出胶囊的维度
                 conv_kernel_size=9,        # 主胶囊卷积层的核大小
                 conv_stride=2,             # 主胶囊卷积层的步长
                 num_routing_iterations=3,  # 动态路由的迭代次数
                 dropout_rate=0.8):         # 可选的dropout
        super(CapsuleNetwork, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.num_primary_caps_types = num_primary_caps_types
        self.primary_caps_dim = primary_caps_dim
        self.num_output_capsules = num_output_capsules
        self.output_capsule_dim = output_capsule_dim
        self.num_routing_iterations = num_routing_iterations

        # 1. 主胶囊层 (PrimaryCaps Layer)
        # 使用1D卷积从输入序列特征中提取局部模式
        # 输出通道数 = num_primary_caps_types * primary_caps_dim
        self.primary_caps_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.num_primary_caps_types * self.primary_caps_dim,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
            padding=(conv_kernel_size - 1) // 2 # 保持与输入长度相关的输出（配合stride）
        )
        
        # 计算卷积后的序列长度
        # L_out = floor((L_in + 2*padding - kernel_size) / stride + 1)
        # 对于 'same' 类型的padding（近似），如果 stride=1, L_out = L_in
        # 如果 stride=2, L_out approx L_in / 2
        # (L_in + 2*P - K)/S + 1
        conv_out_len = ((self.seq_len + 2 * ((conv_kernel_size - 1) // 2) - conv_kernel_size) // conv_stride) + 1
        
        self.num_primary_capsules = self.num_primary_caps_types * conv_out_len

        if self.num_primary_capsules <= 0 :
             raise ValueError(f"Primary capsules calculated to be {self.num_primary_capsules}. Check conv params, seq_len.")


        # 2. 动态路由的权重矩阵 W_ij
        # W_ij 将主胶囊 i 的输出 u_i 转换为对输出胶囊 j 的预测向量 û_j|i
        # 形状: (1, num_primary_capsules, num_output_capsules, output_capsule_dim, primary_capsule_dim)
        # 这样设计使得每个主胶囊到每个输出胶囊都有一个独立的变换矩阵
        self.W = nn.Parameter(torch.randn(1, self.num_primary_capsules, self.num_output_capsules, 
                                          self.output_capsule_dim, self.primary_caps_dim))

        self.dropout = nn.Dropout(dropout_rate)
        # CapsNet分支最终输出的特征维度
        self.output_dim = self.num_output_capsules * self.output_capsule_dim

    def squash(self, tensor, dim=-1):
        """Squashing HBF"""
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # Squash因子
        return scale * tensor / (torch.sqrt(squared_norm + 1e-8) + 1e-8) # 加上epsilon防止除以0和NaN

    def forward(self, x_input, attention_mask=None): # x: (batch_size, seq_len, input_dim)
        batch_size = x_input.shape[0]

        # 1. 通过主胶囊卷积层
        # permute x_input to (batch_size, input_dim, seq_len) for Conv1d
        x_input_permuted = x_input.permute(0, 2, 1)
        primary_caps_raw = self.primary_caps_conv(x_input_permuted) # (batch_size, num_primary_types*primary_dim, conv_out_len)
        
        # Reshape: (batch_size, num_primary_types, primary_dim, conv_out_len)
        primary_caps_raw = primary_caps_raw.view(batch_size, self.num_primary_caps_types, 
                                                 self.primary_caps_dim, -1) 
        # Permute and flatten: (batch_size, num_primary_types*conv_out_len, primary_dim)
        #  -> (batch_size, num_primary_capsules, primary_caps_dim)
        primary_caps_reshaped = primary_caps_raw.permute(0, 1, 3, 2).contiguous()
        primary_caps_flat = primary_caps_reshaped.view(batch_size, self.num_primary_capsules, self.primary_caps_dim)
        
        # 应用 squash 激活函数到主胶囊的输出
        u_i = self.squash(primary_caps_flat, dim=2) # (batch_size, num_primary_capsules, primary_caps_dim)
        u_i = self.dropout(u_i) # 可选的dropout

        # 2. 动态路由 (Dynamic Routing)
        # u_i 扩展维度以进行矩阵乘法: (batch_size, num_primary_capsules, 1, 1, primary_caps_dim)
        u_i_expanded = u_i.unsqueeze(2).unsqueeze(-1)
        
        # û_j|i = W_ij * u_i
        # self.W: (1, num_primary_capsules, num_output_capsules, output_caps_dim, primary_caps_dim)
        # u_hat: (batch_size, num_primary_capsules, num_output_capsules, output_caps_dim, 1)
        u_hat = torch.matmul(self.W, u_i_expanded)
        u_hat = u_hat.squeeze(-1) # (batch_size, num_primary_caps, num_output_caps, output_caps_dim)

        # 初始化路由权重 b_ij (logits) 为0
        # (batch_size, num_primary_capsules, num_output_capsules, 1)
        b_ij = torch.zeros(batch_size, self.num_primary_capsules, self.num_output_capsules, 1, device=x_input.device)

        for r in range(self.num_routing_iterations):
            c_ij = F.softmax(b_ij, dim=2) # 耦合系数 (batch_size, num_primary_caps, num_output_caps, 1)
                                        # dim=2 表示在 num_output_capsules 维度上 softmax

            # s_j = sum_i (c_ij * û_j|i)
            # c_ij: (batch_size, num_primary_caps, num_output_caps, 1)
            # u_hat: (batch_size, num_primary_caps, num_output_caps, output_caps_dim)
            # s_j (weighted sum): (batch_size, 1, num_output_caps, output_caps_dim)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True) 
            
            # v_j = squash(s_j)
            v_j = self.squash(s_j, dim=3) # Squash沿输出胶囊的维度 (output_caps_dim)
                                        # v_j: (batch_size, 1, num_output_caps, output_caps_dim)

            if r < self.num_routing_iterations - 1:
                # 更新 b_ij: b_ij = b_ij + û_j|i ⋅ v_j (内积)
                # u_hat: (batch_size, num_primary_caps, num_output_caps, output_caps_dim)
                # v_j (需要扩展以匹配u_hat的维度进行点积): (batch_size, 1, num_output_caps, output_caps_dim)
                # agreement: (batch_size, num_primary_caps, num_output_caps, 1)
                agreement = (u_hat * v_j).sum(dim=3, keepdim=True) 
                b_ij = b_ij + agreement
        
        # v_j 是输出胶囊的最终激活向量
        # (batch_size, 1, num_output_capsules, output_capsule_dim) -> (batch_size, num_output_capsules, output_capsule_dim)
        output_capsules = v_j.squeeze(1) 
        
        # 可以展平输出胶囊作为最终的特征向量用于分类
        # (batch_size, num_output_capsules * output_capsule_dim)
        final_features = output_capsules.view(batch_size, -1)
        
        return final_features
# --------------- END: 定义 CapsuleNetwork --------------- #



# 重命名并修改 GCNRelationModel
class GCNBranch(nn.Module):
    # gcn_hidden_dim 是 GCN 层的输出维度
    # gcn_output_dim 是 GCN 分支最终输出给分类器的特征维度
    def __init__(self, input_feature_dim, # <--- 修改参数名，更通用
                 gcn_hidden_dim=256, 
                 gcn_output_dim=256, 
                 dropout=0.8):
        super().__init__()

        self.RelGraphConv = RelGraphConv(in_feat=input_feature_dim, # <--- 使用新的输入维度
                                         out_feat=gcn_hidden_dim, 
                                         num_rels=NUM_DEP_TYPES,
                                         regularizer=None, 
                                         num_bases=None,
                                         bias=True)
        self.dropout_gcn = nn.Dropout(dropout)
        self.attention = MultiheadAttention(embed_dim=gcn_hidden_dim, num_heads=8, dropout=dropout)
        self.MLP = nn.Linear(gcn_hidden_dim, gcn_output_dim)
        self.dropout_mlp = nn.Dropout(dropout)
        self.output_dim = gcn_output_dim

    # forward 方法保持不变，它接收的 node_features_batch 的最后一维现在是 input_feature_dim
    def forward(self, node_features_batch, # (batch_size, max_seq_len, input_feature_dim)
                graph_src_nodes_batch,
                graph_dst_nodes_batch,
                graph_edge_types_batch,
                device):
        # ... (forward 内部逻辑不变) ...
        gcn_processed_features_list = []
        for i in range(node_features_batch.shape[0]): # 逐个样本处理
            single_node_features = node_features_batch[i] 
            current_src_nodes = graph_src_nodes_batch[i]
            current_dst_nodes = graph_dst_nodes_batch[i]
            current_edge_types = graph_edge_types_batch[i]

            g = create_dgl_graph(single_node_features, 
                                   current_src_nodes, 
                                   current_dst_nodes, 
                                   current_edge_types, 
                                   max_length, 
                                   device)

            if g.num_edges() > 0:
                etypes_for_gcn = g.edata['etype']
                node_feats_for_gcn = g.ndata['feat']
                gcn_node_outputs = self.RelGraphConv(g, node_feats_for_gcn, etypes_for_gcn)
            elif g.num_nodes() > 0: 
                if single_node_features.shape[-1] == self.RelGraphConv.out_feat: # 通常不相等了
                    # gcn_node_outputs = single_node_features # 这行不成立了，维度不匹配
                    # 需要一个从 input_feature_dim 到 gcn_hidden_dim 的线性变换
                    # 或者，如果 GCN 的 in_feat 直接就是 input_feature_dim，这里就不需要额外变换
                    # 我们的 RelGraphConv 的 in_feat 已经是 input_feature_dim 了
                    # 所以，如果图中没有边，输出的节点特征可以认为是GCN“处理”后的（尽管只是通过了一个 dropout）
                    # 更好的做法是在无边情况下，让节点特征经过一个与GCN层输出维度相同的线性层
                    temp_linear_no_edge = nn.Linear(single_node_features.shape[-1], self.RelGraphConv.out_feat).to(device)
                    gcn_node_outputs = temp_linear_no_edge(single_node_features)

                else: # 旧逻辑，以防万一
                    temp_linear = nn.Linear(single_node_features.shape[-1], self.RelGraphConv.out_feat).to(device)
                    gcn_node_outputs = temp_linear(single_node_features)
            else: 
                gcn_node_outputs = torch.zeros((max_length, self.RelGraphConv.out_feat), device=device)

            gcn_node_outputs = self.dropout_gcn(gcn_node_outputs)
            attn_input = gcn_node_outputs.unsqueeze(1) 
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            attn_output_nodes = attn_output.squeeze(1)
            gcn_processed_features_list.append(attn_output_nodes)

        gcn_batch_node_features = torch.stack(gcn_processed_features_list, dim=0)
        pooled_features = torch.mean(gcn_batch_node_features, dim=1) 
        output = self.MLP(pooled_features)
        output = self.dropout_mlp(output)
        return output


#分类器
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)  
        return logits


class BioMedRelationExtractor(nn.Module):
    def __init__(self):
        super(BioMedRelationExtractor, self).__init__()

        self.shared_encoder = SharedEncoder() 

        self.gcn_branch = GCNBranch(input_feature_dim=self.shared_encoder.output_dim,
                                    gcn_hidden_dim=256, 
                                    gcn_output_dim=128, # <--- 例如，让GCN分支输出128维
                                    dropout=0.8)

        # 实例化 CapsuleNetwork
        self.capsule_branch = CapsuleNetwork(
            input_dim=self.shared_encoder.output_dim, # BioBERT hidden size
            seq_len=max_length, # 全局的 max_length
            num_output_capsules=num_label, # 例如，输出胶囊数等于类别数
            output_capsule_dim=16,         # 每个输出胶囊16维
            # 其他参数使用默认值或按需调整
            dropout_rate=0.8 # CapsuleNet内的dropout
        )

        # 更新分类器的输入维度
        fused_dim = self.gcn_branch.output_dim + self.capsule_branch.output_dim
        self.classifier = Classifier(hidden_dim=fused_dim, num_classes=num_label)

    def forward(self, input_ids, attention_mask, labels,
                graph_src_nodes, graph_dst_nodes, graph_edge_types):

        current_device = input_ids.device
        node_features_from_shared_encoder = self.shared_encoder(input_ids, attention_mask)

        gcn_branch_output = self.gcn_branch(node_features_from_shared_encoder,
                                            graph_src_nodes,
                                            graph_dst_nodes,
                                            graph_edge_types,
                                            current_device)

        # 获取CapsuleNetwork的输出
        # 注意：如果CapsNet使用了attention_mask，这里也需要传入
        capsule_branch_output = self.capsule_branch(node_features_from_shared_encoder, attention_mask=None) # 示例

        # 特征融合
        final_features = torch.cat((gcn_branch_output, capsule_branch_output), dim=1)

        logits = self.classifier(final_features)
        return logits


# 在训练和测试之前定义 true_labels 和 predicted_probs
true_labels = []
predicted_probs = []


# 训练代码
def train_model(model, train_data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    epoch_true_labels = []
    epoch_pred_labels = []

    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # 从batch中获取图相关数据
        graph_src_list = batch['graph_src_nodes']   # <--- 直接获取列表
        graph_dst_list = batch['graph_dst_nodes']   # <--- 直接获取列表
        graph_etypes_list = batch['graph_edge_types'] # <--- 直接获取列表
        # mat = batch['adj_matrix'].to(device) # adj_matrix 现在主要用于调试或非DGL的GCN

        optimizer.zero_grad()
        # 将图数据传递给模型
        outputs = model(input_ids, attention_mask, labels,
                        graph_src_list, graph_dst_list, graph_etypes_list) # <--- 与上面保持一致
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predicted == labels).sum().item()
        
        epoch_true_labels.extend(labels.cpu().numpy())
        epoch_pred_labels.extend(predicted.cpu().numpy())

        # true_labels 和 predicted_probs 的更新保持不变
        true_labels.extend(labels.cpu().numpy())
        predicted_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())

    train_loss = running_loss / len(train_data_loader)
    train_acc = correct_preds / total_preds
    
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    if precision + recall == 0: # 避免除以零
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return train_loss, train_acc, conf_matrix, f1


# 测试代码
def test_model(model, test_data_loader, criterion, device): # criterion在测试时通常不直接用，除非要计算test loss
    model.eval()
    epoch_true_labels = []
    epoch_pred_labels = []

    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 从batch中获取图相关数据
            graph_src_list = batch['graph_src_nodes']   # <--- 直接获取列表
            graph_dst_list = batch['graph_dst_nodes']   # <--- 直接获取列表
            graph_etypes_list = batch['graph_edge_types'] # <--- 直接获取列表

            outputs = model(input_ids, attention_mask, labels,
                            graph_src_list, graph_dst_list, graph_etypes_list) # <--- 与上面保持一致
            _, predicted = torch.max(outputs, 1)

            epoch_true_labels.extend(labels.cpu().numpy())
            epoch_pred_labels.extend(predicted.cpu().numpy())

    # 指标计算部分保持不变
    conf_matrix = confusion_matrix(epoch_true_labels, epoch_pred_labels)
    accuracy = accuracy_score(epoch_true_labels, epoch_pred_labels)
    precision = precision_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(epoch_true_labels, epoch_pred_labels, average='weighted', zero_division=1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    class_report = classification_report(epoch_true_labels, epoch_pred_labels, output_dict=True, zero_division=1)
    f1_per_class = {label_map[key_label]: metrics['f1-score'] 
                    for key_label, metrics in class_report.items() 
                    if key_label in label_map} # 确保使用 label_map 转换回数字标签
    
    return conf_matrix, accuracy, precision, recall, f1, f1_per_class


#模型实例化
model = BioMedRelationExtractor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 20

# 存储训练过程中每个 epoch 的结果
epoch_train_losses = []
epoch_train_accuracies = []
epoch_train_f1_scores = []
epoch_train_conf_matrices = []

# 打开文件，以追加模式（'a'）写入
with open('./figure/training_results.txt', 'a') as f:
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):

        train_loss, train_acc, conf_matrix, f1 = train_model(model, train_data_loader, optimizer, criterion, device)

        #保存结果文件
        f.write(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix) + '\n')

        #打印输出
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        
        # 保存每个 epoch 的结果用于后续可视化
        epoch_train_losses.append(train_loss)
        epoch_train_accuracies.append(train_acc)
        epoch_train_f1_scores.append(f1)
        epoch_train_conf_matrices.append(conf_matrix)



with open('./figure/test_results.txt', 'w') as f:
    test_conf_matrix, test_accuracy, test_precision, test_recall, test_f1, test_f1_per_class = test_model(model, test_data_loader, criterion, device)
    f.write("Test Results:\n")
    f.write("Confusion Matrix:\n")
    f.write(str(test_conf_matrix) + '\n')
    f.write("Accuracy: " + str(test_accuracy) + '\n')
    f.write("Precision: " + str(test_precision) + '\n')
    f.write("Recall: " + str(test_recall) + '\n')
    f.write("F1 Score: " + str(test_f1) + '\n')
    f.write("F1 Score per Class:\n")
    for label, f1 in test_f1_per_class.items():
        f.write(f"Class {label}: {f1:.4f}\n")
    print("Test Results:")
    print("Confusion Matrix:")
    print(test_conf_matrix)
    print("Accuracy:", test_accuracy)
    print("Precision:", test_precision)
    print("Recall:", test_recall)
    print("F1 Score:", test_f1)
    print("F1 Score per Class:")
    for label, f1 in test_f1_per_class.items():
        print(f"Class {label}: {f1:.4f}")



##############################################画图####################################################
# 计算每个类别的 AUC
# 假设你有 `true_labels` 和 `predicted_probs` 以及 `label_map`
num_classes = 5  # 根据你的情况调整
fpr = dict()
tpr = dict()
roc_auc = dict()

# 对于每个类别，计算fpr, tpr和AUC
for i in range(num_classes):
    # 获取每个类的真值和预测概率
    class_true_labels = [1 if true_label == i else 0 for true_label in true_labels]
    class_predicted_probs = [probs[i] for probs in predicted_probs]

    # 计算 fpr 和 tpr
    fpr[i], tpr[i], _ = roc_curve(class_true_labels, class_predicted_probs)
    roc_auc[i] = auc(fpr[i], tpr[i])


##################################训练集结果画图####################################################
# 画训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Loss Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('./figure/training_loss_over_epochs.png', dpi=300)
plt.show()

# 画训练准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Training Accuracy Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('./figure/training_accuracy_over_epochs.png', dpi=300)
plt.show()

# 画训练 F1 分数曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), epoch_train_f1_scores, marker='o', label='Train F1 Score')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('F1 Score', fontsize=14)
plt.title('Training F1 Score Over Epochs', fontsize=16)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('./figure/training_f1_score_over_epochs.png', dpi=300)
plt.show()




################################################测试集结果画图##############################
# 画混淆矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')   # 设置x轴标签，旋转一定角度以避免重叠（如果需要）
plt.yticks(rotation=0)           # 设置y轴标签水平显示
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/confusion_matrix_heatmap.png', dpi=300)  # 保存混淆矩阵热力图
plt.show()

# 画准确率
plt.figure(figsize=(10, 8))
bar_width = 0.3  # # 设置柱子的宽度,可以根据需要调整这个值
plt.bar(range(len(label_map)), [test_accuracy]*len(label_map), color='skyblue', width=bar_width, align='center')
plt.xlabel('Class', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('', fontsize=16)
plt.xticks(range(len(label_map)), label_map.keys(), rotation=45, ha='right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/accuracy_by_class.png', dpi=300)  # 保存准确率图
plt.show()

#画AUC曲线图
plt.figure(figsize=(10, 6))

for i, label in enumerate(label_map):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 随机分类器的线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (ROC)', fontsize=16)
plt.legend(loc='lower right')
plt.tight_layout()  # 调整子图布局以适应标签
plt.savefig('./figure/auc_by_class.png', dpi=300)
plt.show()
