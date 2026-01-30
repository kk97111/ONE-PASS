import os
from re import T
import time
import argparse

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
#数据读取
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import load_checkpoint,find_latest_checkpoint,setup_logging,save_checkpoint,GenerationDataset,extract_from_string
from grop_func import build_prefix_tree
import torch.nn as nn   
from itertools import combinations,permutations
import torch.nn.functional as F
from eval_func import evaluate
from datasets import load_from_disk
import logging
import copy


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Mobius Inversion Training Script")
    parser.add_argument("--backbone", type=str, required=True,
                      help="Base model name")
    parser.add_argument("--data_name", type=str, required=True,
                      help="data name")
    parser.add_argument("--model_path", type=str, 
                      default="/data/zoo/%(backbone)s",
                      help="Base model path")
    parser.add_argument("--dataset_path", type=str,
                      default="./dataset/%(data_name)s/%(backbone)s/",
                      help="Path to preprocessed dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                      help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--n_group", type=int, default=5)
    parser.add_argument("--variant", type=str, default='combinations')
    parser.add_argument("--loss_top_k", type=int, default=2)
    return parser.parse_args()




def initialize_models(args):
    """Initialize base model and tokenizer"""
    # 使用示例 ----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='cuda',
        # torch_dtype=torch.bfloat16,
        # attn_implementation='flash_attention_2',
    )
    model.eval()
    args.hidden_state_dim = model.config.hidden_size
    tree_llm = TreeMaskLlama(model)# 
    return tree_llm, tokenizer

def load_datasets(dataset_path):
    """Load and split datasets"""
    print(dataset_path)
    ds = load_from_disk(dataset_path)
    def func_1(example): # 去除有重复元素
        extracted = extract_from_string(example['conversations'][2]['value'])
        return len(extracted) >= 20 #保留ranking大于10的部分    
    ds = ds.filter(func_1)
    size = ds.num_rows
    print(size)
    train_index = int(size - 1000)
    valid_index = int(size - 500)
    
    return (
        ds.select(range(train_index)),
        ds.select(range(train_index, valid_index)),
        ds.select(range(valid_index, size))
    )


class TreeMaskLlama(nn.Module):
    def __init__(self, llm):
        super(TreeMaskLlama, self).__init__()
        self.llm = llm
    def build_tree_mask(self, tree_structure, seq_len,device = None):
        """生成正确维度的树状掩码"""
        #tree_structure: [seq_len]
        #seq_len: int
        mask = torch.full((1, 1, seq_len, seq_len), float('-inf')).cuda()
        mask[:,:,0,0] = 0
        for i in range(1,seq_len):
            mask[:,:,i,:] = mask[:,:,tree_structure[i],:]
            mask[:,:,i,i] = 0
        return mask

    def forward(self, input_ids, tree_structure=None, position_ids=None, **kwargs):
        if position_ids is None:
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)

        if tree_structure is not None:
            seq_len = input_ids.shape[1]
            tree_mask = self.build_tree_mask(
                tree_structure=tree_structure,
                seq_len=seq_len,
            ).to(dtype=self.llm.dtype)  # 关键：与模型匹配的 dtype

            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                causal_mask = kwargs['attention_mask'].to(dtype=tree_mask.dtype)
                combined_mask = torch.minimum(causal_mask, tree_mask)
            else:
                combined_mask = tree_mask
        # causal_mask = torch.triu(torch.ones(L, L), diagonal=1)
        # 把上三角部分（未来的信息）设置为 -inf
        # causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        # 统一通过 kwargs 传参，避免重复
        return self.llm(input_ids=input_ids, position_ids=position_ids, attention_mask=combined_mask, output_hidden_states=True ,**kwargs)
    



def group_division(num = 20, n_group = 3):
    # 构造combinations, 每个group的元素个数为num/n_group
    #其中元素是字符串，如['A','B','C','D','E','F','G','H','I','J']
    lst = [[] for _ in range(n_group*2)]
    for i in range(num):
        lst[i % (2*n_group)].append(chr(ord('A')+i))
    target_lst = [[] for _ in range(n_group)]
    dual_lst = [[] for _ in range(n_group)]
    for i in range(n_group): #[0,1],[2,3],[4,5]
        target_lst[i] = sorted(lst[2*i] + lst[2*i+1])
    for i in range(n_group): #[-1,0],[1,2],[3,4]
        dual_lst[i] = sorted(lst[2*i-1] + lst[2*i])
    return target_lst,dual_lst




class SelfAttentionNoValueMLP(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(SelfAttentionNoValueMLP, self).__init__()
        self.dim = dim
        self.scale = dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape  # Batch, SeqLen, Dim

        q = self.q(x)  # [B, N, D]
        k = self.k(x)  # [B, N, D]
        v = (x)          # 不做任何变换，直接作为 value

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, N, N]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v) +  x  # [B, N, D]
        return out


class Tree_inference(nn.Module):
    def __init__(self,model,tokenizer,args = None):
        super(Tree_inference,self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.num_candidate = args.num_candidate
        self.n_group = args.n_group
        self.length = 2
        self.corpus_id = tokenizer.convert_tokens_to_ids([chr(ord('A')+i) for i in range(self.num_candidate)])
        self.temperature = 1.0
        self.target_lst,self.dual_lst = group_division(num = self.num_candidate, n_group = self.n_group)# target_lst 和 dual_lst 是两个列表list[list[str]]；
        self.transformer = SelfAttentionNoValueMLP(dim=self.num_candidate).cuda()
        self.weight =  nn.Parameter(torch.ones([self.num_candidate])).cuda()
        self.tree_freature = self.construct_tree(mode = args.mode)

    def set_eval(self):
        self.transformer.eval()
    def set_train(self):
        self.transformer.train()
    def construct_combinations(self,num = 20, n_group = 3):
        # 构造combinations, 每个group的元素个数为num/n_group,组合shu
        all_combinations = []
        target_combinations = []
        dual_combinations = []
        for i in range(n_group):
            #有一个subset的组合，从1到subset的元素个数，每个subset的组合，其中每个组合(e.g.,target_combinations)是list[list[str]]
            target_combinations.extend([list(c) for r in range(1, len(self.target_lst[i]) + 1) for c in combinations(self.target_lst[i], r)])
            dual_combinations.extend([list(c) for r in range(1, len(self.dual_lst[i]) + 1) for c in combinations(self.dual_lst[i], r)])
        
        all_combinations = target_combinations + dual_combinations
        # 将每个组合转换为字符串
        self.target_combinations_keys = [''.join(f"{element}" for element in combinations) for combinations in target_combinations]
        self.dual_combinations_keys = [''.join(f"{element}" for element in combinations) for combinations in dual_combinations]
        all_combinations_keys = [''.join(f"{element}" for element in combinations) for combinations in all_combinations]
        all_combinations = [self.expand_with_arrows(combinations) for combinations in all_combinations]
        return all_combinations, all_combinations_keys
    def construct_permutation(self,num = 20, length= 3):
        all_permutations = []
        items = [chr(ord('A')+i) for i in range(num)]
        for i in range(length):
            all_permutations.extend([list(c) for c in permutations(items,i+1)])
        all_permutations_keys = [''.join(f"{element}" for element in combinations) for combinations in all_permutations]
        all_permutations = [self.expand_with_arrows(combinations) for combinations in all_permutations]
        return all_permutations, all_permutations_keys
    def expand_with_arrows(self,lst):
        #这个函数的作用是，将一个list[str]转换为list[str]，其中每个元素是list[str]，
        #如['A','B','C']转换为['|','A','|','Ġ>','Ġ|','B','|','Ġ>','Ġ|','C','|','Ġ>','Ġ|']
        result = ['|']
        for  elem in lst:
            result.append(elem)
            result.extend(['|', 'Ġ>', 'Ġ|'])
        return result
    def construct_tree(self,mode = 'combinations'):
        if mode in ['combinations','Mobius']: 
            all_combinations, all_combinations_keys = self.construct_combinations(num = self.num_candidate, n_group = self.n_group)
        elif mode == 'permutation':
            all_combinations, all_combinations_keys = self.construct_permutation(num = self.num_candidate, length=self.length ) 
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # 构建前缀树
        # 输入：all_combinations: list[list[str]]
        # 输出：tree_list: list[int], values: list[str], depths: list[int], search_path: list[list[int]]
        # 其中tree_list是父节点索引，values是每个节点的值，depths是每个节点的深度，search_path是每个节点的搜索路径
        tree_list, values, depths, search_path = build_prefix_tree(all_combinations)
        # 构建搜索路径字典
        search_path_dict = {all_combinations_keys[i]:  search_path[i][-1] for i in range(len(all_combinations))}
        search_path_dict[''] = 0
        input = torch.tensor([tokenizer.convert_tokens_to_ids(values)])
        tree_freature = {
            'input':input,
            'tree_list':tree_list,
            'values':values,
            'depths':depths,
            'search_path':search_path,
            'search_path_dict':search_path_dict
        }
        return tree_freature
    def hidden_state_prob(self,prompt,mode = 'combinations'):

        """
        mode: 'combinations','Mobius','permutation'
        Mobius,combinations: 每个group的元素个数为num/n_group,组合数
        permutation: 每个元素的组合数
        return: hidden_states: np.array, prob: np.array, search_path_dict: dict
        hidden_states: [batch_size, sequence_length, hidden_size] 意义是每个token的hidden_state
        prob: [batch_size, sequence_length, num_candidate] 意义是每个token的每个候选词的概率
        search_path_dict: dict, key: str, value: int 意义是每个token的每个候选词的搜索路径
        """
        

        # 将prompt转换为token
        prompt_tokenID = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )['input_ids']        
        # 将combinations ranking 转换为token
        len_prompt = prompt_tokenID.size()[1]
        combined_tokenID = torch.concat([prompt_tokenID,self.tree_freature['input']],dim=-1).cuda()
        # print(len_prompt,len(tree_list))
        # tree_structure
        tree_structure_concate = [i-1 for i in range(len_prompt)] + [ i + len_prompt for i in self.tree_freature['tree_list']]
        with torch.no_grad():
            position_structure_concate = torch.tensor([[i for i in range(len_prompt)] + [ i + len_prompt for i in self.tree_freature['depths']]]).cuda()
            outputs =  self.model(input_ids=combined_tokenID,tree_structure=tree_structure_concate,position_ids=position_structure_concate)
            hidden_states = outputs.hidden_states[-1][:,len_prompt:,:]  # 形状是 [batch_size, sequence_length, hidden_size]
            # 计算logit 和 prob
            wanted_weight = self.model.llm.lm_head.weight[self.corpus_id, :]  # [num_wanted_tokens, hidden_size]
            logits = F.linear(hidden_states, wanted_weight)
            if self.temperature > 0: 
                logits = logits / self.temperature  # 应用温度参数调整logits
            prob_corpus = torch.softmax(logits,dim=-1)
            prob_np = prob_corpus.cpu().detach().numpy()
            hidden_states = hidden_states.cpu().detach().numpy()
        return hidden_states,prob_np,self.tree_freature['search_path_dict']
    
    def count_token(self, mode = 'combinations'):
        if mode == 'combinations':
            all_combinations, all_combinations_keys = self.construct_combinations(num = self.num_candidate, n_group = self.n_group)
        elif mode == 'permutation':
            all_combinations, all_combinations_keys = self.construct_permutation(num = self.num_candidate, length=self.length )
        else:
            raise ValueError(f"Invalid mode: {mode}")
        tree_list, values, depths, search_path = build_prefix_tree(all_combinations)

        return len(tree_list)

    def generate(self,generate_item,candidate_items,prob_np,search_path_dict,mode = 'combinations',heuristic = False):
        if len(generate_item) == 0:
            prob = prob_np[0][search_path_dict['']]
            generate_item.append(chr(ord('A')+prob.argmax()))
            return generate_item
        if mode == 'combinations':
            if heuristic:
                group_prob = []
                for i in range(self.n_group):
                    for lst in [self.target_lst[i]]:
                        group_i = [item for item in generate_item if item in lst]
                        group_i_prob = prob_np[0][search_path_dict[''.join(sorted(group_i))]] if len(group_i) > 0 else np.zeros(self.num_candidate)
                        group_prob.append(group_i_prob)
                group_prob = np.stack(group_prob,axis=0)
                prob = group_prob.sum(axis=0) #[num_candidate]
                prob = prob[[ord(item)-ord('A') for item in candidate_items]]
                # prob = prob / prob.sum()
            else:
                prob = self.transformer_prob(generate_item,prob_np,search_path_dict).cpu().detach().numpy() #[num_candidate]
                prob = prob[[ord(item)-ord('A') for item in candidate_items]]
            generate_item.append(candidate_items[prob.argmax()])
            return generate_item
        elif mode == 'Mobius':
            if heuristic:
                individual_contribution,individual_contribution_dual = [],[]
                # 计算每个group的indivudal contribution
                for i in range(self.n_group):
                    group_i = [item for item in generate_item if item in self.target_lst[i]]
                    if len(group_i) == 0:
                        individual_contribution.append(np.zeros(self.num_candidate))
                    else:
                        # individual_contribution.append(self.compute_individual_contribution(group_i,prob_np,search_path_dict)) #[num_candidate]
                        individual_contribution.append(prob_np[0][search_path_dict[''.join(sorted(group_i))]]-prob_np[0][search_path_dict['']]) #[num_candidate]
                individual_contribution = np.sum(np.stack(individual_contribution,axis=0),axis=0) # [num_candidate]
                # 计算每个group的indivudal contribution
                for i in range(self.n_group):
                    group_i = [item for item in generate_item if item in self.dual_lst[i]]
                    if len(group_i) == 0:
                        individual_contribution_dual.append(np.zeros(self.num_candidate))
                    else:
                        individual_contribution_dual.append(self.compute_individual_contribution(group_i,prob_np,search_path_dict)) #[num_candidate]
                individual_contribution_dual = np.sum(np.stack(individual_contribution_dual,axis=0),axis=0) # [num_candidate]
                prob =   individual_contribution + individual_contribution_dual    #[num_candidate]
                prob = prob[[ord(item)-ord('A') for item in candidate_items]]
            else:
                prob = self.transformer_prob_mobius(generate_item,prob_np,search_path_dict).cpu().detach().numpy() #[num_candidate]
                prob = prob[[ord(item)-ord('A') for item in candidate_items]]
            generate_item.append(candidate_items[prob.argmax()])
            return generate_item
        elif mode == 'permutation':
            prob = prob_np[0][search_path_dict[''.join(generate_item[:self.length])]] # [num_candidate]
            prob = prob[[ord(item)-ord('A') for item in candidate_items]]
            # print(prob.shape)
            generate_item.append(candidate_items[prob.argmax()])
            return generate_item


    def transformer_prob_mobius(self,generate_item,prob_np,search_path_dict):
        #generate_item是已经生成的候选词序列，如['A','B','C']
        individual_contribution = []
        individual_contribution_dual = []
        # 计算每个group的indivudal contribution
        for i in range(self.n_group):
            group_i = [item for item in generate_item if item in self.target_lst[i]]
            if len(group_i) == 0:
                individual_contribution.append(np.zeros(self.num_candidate))
            else:
                individual_contribution.append(self.compute_individual_contribution(group_i,prob_np,search_path_dict)) #[num_candidate]
        individual_contribution = np.stack(individual_contribution,axis=0)
        individual_contribution = torch.sum(torch.tensor(individual_contribution,dtype=torch.float32).cuda(),dim=0) # [num_candidate]

        # 计算每个group的indivudal contribution
        for i in range(self.n_group):
            group_i = [item for item in generate_item if item in self.dual_lst[i]]
            if len(group_i) == 0:
                individual_contribution_dual.append(torch.zeros(self.num_candidate).cuda()) # [num_candidate]
            else:
                feature_i = torch.tensor(self.compute_individual_contribution(group_i,prob_np,search_path_dict,False),dtype=torch.float32).cuda().unsqueeze(0) #[1，?,num_candidate]
                logits = self.transformer(feature_i) # [1,n_group,hidden_state_dim]
                individual_contribution_dual.append(torch.sum(logits[0],dim=0)) # [hidden_state_dim]
                
        individual_contribution_dual = torch.stack(individual_contribution_dual,dim=0).unsqueeze(0)#[1,n_group,num_candidate]
        individual_contribution_dual = torch.sum(self.transformer(individual_contribution_dual)[0],dim=0) # [num_candidate]
        # individual_contribution_dual = torch.sum(torch.stack(individual_contribution_dual,dim=0),dim=0) # [num_candidate]
        fusion_logit =   individual_contribution + individual_contribution_dual    #[num_candidate]
        actor_prob_fusion = fusion_logit
        return actor_prob_fusion # [num_candidate]
    
    def transformer_prob(self,generate_item,prob_np,search_path_dict):
        hidden_states_stack = []
        # 计算每个group的indivudal contribution
        for i in range(self.n_group):
            group_i = [item for item in generate_item if item in self.target_lst[i]]
            stack_i = []
            if len(group_i) == 0:
                hidden_states_stack.append(torch.zeros(self.num_candidate).cuda())
            else:
                stack_i.append(prob_np[0][search_path_dict[''.join(sorted(group_i))]])
                stack_i = torch.tensor(stack_i,dtype=torch.float32).cuda().unsqueeze(0)
                transformer_prob_i = self.transformer(stack_i)
                transformer_prob_i = transformer_prob_i[0].sum(dim=0)
                hidden_states_stack.append(transformer_prob_i)
        # 计算每个group的indivudal contribution
        for i in range(self.n_group):
            stack_i = []
            group_i = [item for item in generate_item if item in self.dual_lst[i]]
            if len(group_i) == 0:
                hidden_states_stack.append(torch.zeros(self.num_candidate).cuda())
            else:
                if self.args.variant == 'wo_Mobius-1':
                    combinations_i = [''.join(sorted(combo)) for combo in combinations(group_i,len(group_i))]
                if self.args.variant == 'wo_Mobius-2':
                # print(combinations_i)
                    combinations_i = [''.join(sorted(combo))
                            for r in range(1, len(group_i) + 1)
                            for combo in combinations(group_i, r)]
                # print(combinations_i)
                for combination_i in combinations_i:
                    stack_i.append(prob_np[0][search_path_dict[combination_i]])
                stack_i = torch.tensor(stack_i,dtype=torch.float32).cuda().unsqueeze(0)
                transformer_prob_i = self.transformer(stack_i)
                transformer_prob_i = transformer_prob_i[0].sum(dim=0)
                hidden_states_stack.append(transformer_prob_i)
        hidden_states_stack = torch.stack(hidden_states_stack,dim=0)
        hidden_states_stack = hidden_states_stack.unsqueeze(0) # [1,n_group,num_candidate]
        hidden_states_stack = self.transformer(hidden_states_stack) # [1,n_group,num_candidate]
        actor_prob_fusion =   hidden_states_stack[0].sum(dim=0)    #[num_candidate]
        return actor_prob_fusion # [num_candidate]


    def compute_individual_contribution(self,itemB, prob_np,search_path_dict,none_sum = True) -> float:
        """
        计算集合B的总贡献 ∑_{T⊆B} g_B(T)
        
        参数:
            itemB: 基础元素列表，如 ['A','B','C']
            subset_probs: 子集概率字典，键为字符串格式如 "AB"表示{A,B}
        
        返回:
            B的总贡献值（float）
        """
        subset_probs = copy.deepcopy({key: prob_np[0][search_path_dict[key]] for key in search_path_dict.keys()})
        # subset_probs = {key: prob_np[0][search_path_dict[key]] for key in search_path_dict.keys()}
        n = len(itemB)
        
        # 1. 构建所有子集的规范表示（按长度排序）
        all_subsets = []
        for r in range(n+1):
            for combo in combinations(itemB, r):
                # 将子集转为排序后的字符串键，如 ('B','A') -> "AB"
                subset_key = ''.join(sorted(combo))
                all_subsets.append(subset_key)
        
        # 2. 检查输入完整性
        required_keys = set(all_subsets)
        missing_keys = required_keys - set(subset_probs.keys())
        if missing_keys:
            raise KeyError(f"缺少以下子集概率: {missing_keys}")
        
        # 3. 快速Möbius变换（基于包含-排除原理）
        mobius = {}
        mobius[''] = subset_probs['']  # 空集
        
        # 按子集长度从小到大处理
        for subset in sorted(all_subsets[1:], key=len):
            # 当前子集的概率
            mobius[subset] = subset_probs[subset]
            
            # 减去所有真子集的Möbius系数
            for k in range(1, len(subset)):
                for sub_subset in combinations(subset, k):
                    sub_key = ''.join(sorted(sub_subset))
                    mobius[subset] -= mobius[sub_key]
        
        # 4. 计算总贡献（排除空集）
        if none_sum:
            total = np.sum([v for k,v in mobius.items() if k != ''],axis=0)
            return total
        else:
            total = np.stack([v for k,v in mobius.items() if k != ''],axis=0)
            return total

def train_loop(args, tree_llm, tokenizer, train_data,valid_data,test_data):
    """Main training loop"""
    # 初始化Actor和优化器
    actor = Tree_inference(tree_llm,tokenizer,args=args)#            
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    
    # 初始化训练状态
    start_batch = 0
    loss_stats = {'epoch_loss': 0.0, 'epoch_kl_loss': 0.0, 'epoch_grpo_loss': 0.0}
    log_dir = setup_logging(args, resume=False)

    
    
    # 初始化数据集和数据加载器
    train_dataset = GenerationDataset(train_data, tokenizer, combined=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True,drop_last=True)
    valid_dataset = GenerationDataset(valid_data, tokenizer, combined=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataset = GenerationDataset(test_data, tokenizer, combined=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    stats_file = f"{log_dir}/logs/training.log"
    if not args.heuristic:
        # 训练循环
        total_batches = len(train_dataloader)
        start_time = time.time()
        # 创建训练统计文件
        with open(stats_file, 'a') as f:
            f.write("epoch,batch_idx,loss,time\n")
        ranking_save = - np.inf
        train_tolerance = 0
        # 初始化优化器和损失函数
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
        # 训练循环
        for epoch in range(0, 1):
            epoch_loss = loss_stats['epoch_loss']
            
            # 初始化进度条
            pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", 
                        postfix={"loss": 0.0},
                        disable=False)
            
            # 跳过已经处理过的batch
            dataloader_iter = enumerate(train_dataloader)
            
            for batch_idx, batch in dataloader_iter:
                prompts = list(batch[0])#每个batch揪一个prompt
                GT_label = [extract_from_string(label) for label in batch[1]][0]
                _,prob_np,search_path_dict = actor.hidden_state_prob(prompts,mode = args.mode)
                # print(search_path_dict)
                loss = []
                pred_generate_item = [GT_label[0]]#从第一个候选词开始；因为第一个候选词一定是GT_label[0]
                for k in range(1, args.num_candidate-1):
                    if args.mode == 'Mobius':
                        transformer_prob = actor.transformer_prob_mobius(pred_generate_item,prob_np,search_path_dict) #[num_candidate]
                    else:
                        transformer_prob = actor.transformer_prob(pred_generate_item,prob_np,search_path_dict) #[num_candidate]
                    
                    candidate_items_index = [idx for idx in range(actor.num_candidate) if chr(ord('A')+idx) not in pred_generate_item]
                    pred_generate_item.append(chr(ord('A')+candidate_items_index[transformer_prob[candidate_items_index].argmax().item()]))
                    targets = [item for item in GT_label if item not in pred_generate_item]
                    target_indices = [idx for idx in range(args.num_candidate) if chr(ord('A')+idx) in targets[:args.loss_top_k]]
                    label_ranking = torch.zeros(args.num_candidate).cuda()
                    label_ranking[target_indices] = 1


                    # loss.append(our_rank_net_loss(transformer_prob.unsqueeze(0),label_ranking.unsqueeze(0),weighted=True))
                    loss.append(loss_fn(transformer_prob.unsqueeze(0),label_ranking.unsqueeze(0)))
                loss = torch.mean(torch.stack(loss))
                # 反向传播
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # 记录损失
                batch_loss = loss.item()
                
                epoch_loss += batch_loss
                
                # 更新进度条
                pbar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                })
                pbar.update(1)
                
                # 记录到统计文件
                with open(stats_file, 'a') as f:
                    f.write(f"{epoch},{batch_idx},{batch_loss},{time.time()-start_time}\n")
                
                # 定期保存模型和日志
                if (batch_idx ) % args.save_interval == 0 or (batch_idx) == total_batches:
                    actor.set_eval()
                    ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate(valid_dataloader, actor,mode = args.mode,heuristic = args.heuristic,eval_mode = 'fast')
                    print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15])
                    actor.set_train()
                    loss_stats = {
                        'epoch_loss': epoch_loss,
                    }
                    if ranking_save < ranking:
                        train_tolerance = 0
                        ranking_save = ranking
                        save_checkpoint(args, actor, optimizer, batch_idx + 1, epoch, log_dir, loss_stats)

                    else:
                        train_tolerance = train_tolerance + 1
                    if train_tolerance > 4:
                        break

                    
                    # 记录训练进度
                    avg_loss = epoch_loss / (batch_idx + 1 - (start_batch if epoch == 0 else 0))
                    logging.info(
                        f"Epoch {epoch+1} Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Time: {(time.time()-start_time)/60:.2f} min"
                    )
                    with open(stats_file, 'a') as f:
                        f.write(f"Epoch:{epoch},Batch:{batch_idx},Ranking:{ranking},Spearman:{spearman},Footrule:{footrule},Kemeny:{kemeny},Jaccard_at_10:{jaccard_at_10},Jaccard_at_15:{jaccard_at_15}\n")
                    # 重置计时器
                    start_time = time.time()
            
            # 关闭当前epoch的进度条
            pbar.close()

        # 训练结束保存最终模型
        logging.info("Training completed!")
        # for test
        
        checkpoint_path = find_latest_checkpoint(args, timestamp_dir=log_dir.split('/')[-1]) 
        # 加载checkpoint
        load_checkpoint(checkpoint_path, args, actor, optimizer)
    actor.set_eval()
    time_start = time.time()
    ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate(test_dataloader, actor,mode = args.mode,heuristic = args.heuristic,eval_mode = 'complete')    
    time_end = time.time()
    total_time = (time_end - time_start)/len(test_dataloader)
    print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15])
    with open(stats_file, 'a') as f:
            f.write(f"TEST evaluation,Ranking:{ranking},Spearman:{spearman},Footrule:{footrule},Kemeny:{kemeny},Jaccard_at_10:{jaccard_at_10},Jaccard_at_15:{jaccard_at_15},Time:{total_time:.2f}s/instance\n")



if __name__ == "__main__":
    args = parse_args()
    args.num_candidate = int(args.data_name.split('_')[-1])
    # 路径格式化
    args.model_path = args.model_path % {"backbone": args.backbone}
    args.dataset_path = args.dataset_path % {"backbone": args.backbone,"data_name":args.data_name}
    variant = args.variant
    
    
    heuristic_dict = {'Permutation':True,
                      'ONE-PASS':False,
                      'Naive_add':True,
                      'wo_HAS':True,
                      'wo_Mobius-1':False,
                      'wo_Mobius-2':False}
    args.heuristic = heuristic_dict[variant]
    mode_dict = {'Permutation':'permutation',
                 'ONE-PASS':'Mobius',
                 'Naive_add':'combinations',
                 'wo_HAS':'Mobius',
                 'wo_Mobius-1':'combinations',
                 'wo_Mobius-2':'combinations'}

    args.mode = mode_dict[variant]
    
    
    
    # args.heuristic = True if 'heuristic' in args.mode else False
    # args.mode = args.mode.split('_')[0]
    # 初始化模型和数据集
    llm, tokenizer = initialize_models(args)
    train_data, valid_data, test_data= load_datasets(args.dataset_path)
    
    # 开始训练
    train_loop(args, llm, tokenizer, train_data, valid_data,test_data)
