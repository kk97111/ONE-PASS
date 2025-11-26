
import torch
#数据读取
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from grop_func import initial_step_logit
# Custom imports
from utils import GenerationDataset, extract_from_string,load_datasets
import copy
import time
from transformers import AutoTokenizer,AutoModelForCausalLM

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--backbone", type=str, required=True,
                      help="Base model name (e.g. 'Llama-3.2-3B-Instruct')")
    parser.add_argument("--data_name", type=str, required=True,
                      help="Dataset name (e.g. 'rank_zephyr_IR_20')")
    parser.add_argument("--n_step", type=int, default=5,
                      help="Number of training steps")
    parser.add_argument("--model_path", type=str, 
                      default="/data/zoo/",
                      help="Base directory for model weights")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4) #[5e-4 for rank_zephyr]

    return parser.parse_args()

def get_label_ranking(generated_label):
    # 生成标签到排名的映射（排名 = 索引 + 1）
    ranking_dict = {label: idx  for idx, label in enumerate(generated_label)}
    
    # 按字母顺序排序标签
    sorted_labels = sorted(ranking_dict.keys())
    
    # 生成对应的排名列表
    sorted_ranking = [ranking_dict[label] for label in sorted_labels]
    
    # 返回排序后的标签字典、标签列表、排名列表
    return sorted_ranking

class TreeMaskLlama(nn.Module):
    def __init__(self, llm):
        super(TreeMaskLlama, self).__init__()
        self.llm = llm
    def build_tree_mask(self, tree_structure, seq_len,device = None):
        """生成正确维度的树状掩码"""
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
                # device=input_ids.device
            ).to(dtype=self.llm.dtype)  # 关键：与模型匹配的 dtype

            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                causal_mask = kwargs['attention_mask'].to(dtype=tree_mask.dtype)
                combined_mask = torch.minimum(causal_mask, tree_mask)
            else:
                combined_mask = tree_mask

            kwargs['attention_mask'] = combined_mask


        # 统一通过 kwargs 传参，避免重复
        return self.llm(input_ids=input_ids, position_ids=position_ids, **kwargs)


import copy
class Actor(nn.Module):
    def __init__(self, args,hidden_size,vocab_size,tree_llm,tokenizer):
        super(Actor, self).__init__()
        self.args = args
        self.vocab_size=vocab_size
        self.hidden_size = hidden_size
        self.tree_llm = tree_llm
        self.token_corpus_id = tokenizer.convert_tokens_to_ids([chr(i+ord('A')) for i in range(vocab_size)])
        init_weight = self.tree_llm.llm.lm_head.weight[self.token_corpus_id, :].detach().clone()
        self.wanted_weight = nn.Parameter(init_weight) # based on paper, init_weight as lm head
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size).cuda(),
                nn.SiLU()
            ) for _ in range(vocab_size)
        ])
        self.index2char = {i: chr(i+ord('A')) for i in range(25)}
        self.char2index = {chr(i+ord('A')):i for i in range(25)}
        self.corpus = set([self.index2char[i] for i in range(self.vocab_size)])
        with torch.no_grad():
            for layer in self.output_layers:
                linear = layer[0]  # 取出第0个模块，就是Linear
                linear.weight.zero_()
    def set_eval(self):
        pass
    def set_train(self):
        pass
    def longest_greedy(self, pred_label, matrix):
        correct = [] # 按照贪婪算法一直正确的list
        for i, pred in enumerate(pred_label):
            index = np.argmax(matrix[i])
            matrix[:,index] = -1
            if pred == self.index2char[index]:
                correct.append(pred)
                if len(pred_label) == len(correct):
                    return correct
            else: # 贪一个再说
                correct.append(self.index2char[index])
                break
        return correct




    def last_sampled_action(self, current_sequence, logits):
        next_docs = []
        remain_doc_indices = [self.char2index[doc] for doc in set(self.corpus - set(current_sequence))] #对应剩下的index；
        for i,line in enumerate(logits):
            select_doc_index = remain_doc_indices[np.argmax(line[remain_doc_indices])] #最大index
            next_docs.append(self.index2char[select_doc_index])
            remain_doc_indices.remove(select_doc_index)
        return next_docs
    def parallel_sampled_action(self, current_sequence, logits,k=3):
        next_docs = []
        remain_doc_indices = [self.char2index[doc] for doc in set(self.corpus - set(current_sequence))] #对应剩下的index；
        for i,line in enumerate(logits):    
            top4_indice_k = np.argsort(line[remain_doc_indices])[-k:][::-1]
            if i == 0:
                next_docs.append([self.index2char[remain_doc_indices[top4_indice_k[0]]]])
            else:
                next_docs.append([self.index2char[remain_doc_indices[index]] for index in top4_indice_k])
        return next_docs
            




               
    def drafting(self,current_sequence,original_decoder_output,last = False):
        # for this action, it receives a batch of
        #batch_matrices -->[batch,25,25]
        multi_head_hidden_state = [original_decoder_output]
        for i in range(1,self.vocab_size):
            multi_head_hidden_state.append(self.output_layers[i](original_decoder_output) + original_decoder_output) 
        multi_head_hidden_state = torch.stack(multi_head_hidden_state,dim=0)#[K,hidden_size]
        

        remain_logits = F.linear(multi_head_hidden_state, self.wanted_weight)[:self.vocab_size-len(current_sequence),:]#[K,prob]

        remain_prob = torch.softmax(remain_logits,dim=-1).to(torch.float32).cpu().detach().numpy() #[K,prob]
        # 行为抽样
        if last:
            next_docs = self.last_sampled_action(current_sequence, remain_prob)
            return current_sequence + next_docs
        else:
            next_docs = self.parallel_sampled_action(current_sequence, remain_prob)
            return [[entity] for entity in current_sequence] + next_docs,remain_logits

    def verifying(self,prob_mat,candidate_sequence,node_name):
        # because temperature = 0 in our case, therefore the sample strategy degrades into greedy stategy.  
        select = []
        for line in candidate_sequence:
            if len(line) == 1:
                select.append(line[0])
            else:
                break
            


        for i in range(100):
            index = node_name.index('-'.join(select)) 
            prob = prob_mat[index] #对应元素；
            #next item 
            remain_doc_indices = [self.char2index[doc] for doc in set(self.corpus - set([s for s in select]))] #对应剩下的index；
            if len(remain_doc_indices) == 0:
                break
            next_item = self.index2char[remain_doc_indices[np.argmax(prob[remain_doc_indices])]]
            # print(next_item)

            if '-'.join(select + [next_item]) not in node_name:
                break
            else:
                select.append(next_item)

        return select,index
    

def generate_tree(tree_levels,step=4):
    """
    自动展开树（任意层数任意宽度），返回节点列表和父节点对应关系。
    tree_levels: List[List[str]]，每层使用什么标签。
    """
    tree_levels_cp = copy.deepcopy(tree_levels[:step]) + [['Z']]
    node_list = ''
    node_name = []
    tree_structure = []
    position = []
    
    # 初始层：根节点
    current_nodes = []
    crrennt_end = []
    for name in tree_levels_cp[0]:
        node_name.append(name)
        node_list+=' |%s| >'%name
        tree_structure.extend([-1,0,1,2])
        current_nodes.append(len(node_name) - 1)  # 保存索引
        crrennt_end.append((len(node_name) - 1) * 4 + 3)
        position.extend([0,1,2,3])

    # 后续每层
    for level in range(1, len(tree_levels_cp)):
        next_nodes = []
        next_end = []
        for j,parent_idx in enumerate(current_nodes):
            for child_name in tree_levels_cp[level]:
                # 新节点名字 = 父节点名字 + '-' + 当前名字
                full_name = f"{node_name[parent_idx]}-{child_name}"
                node_name.append(full_name)
                tree_structure.extend([crrennt_end[j] + i for i in range(4)])
                position.extend([level*4 + i for i in range(4)])
                node_list += ' |%s| >'%child_name
                next_nodes.append(len(node_name) - 1)
                next_end.append((len(node_name) - 1) * 4 + 3)
        current_nodes = next_nodes
        crrennt_end = next_end
    
    
    return node_name,node_list[1:],tree_structure,position

def ranking_prob_tree_mask(actor,tree_llm,tokenizer,prompt,node_list,tree_structure,position,node_name):
    # 这个函数输入prompt和ranking，已知位置i之前的ranking,输出位置i对应的生成的概率；P(X_i|X_1,...,X_{i-1});
    
    #step:分别对prompt和ranking进行tokenize
    corpus_id = actor.token_corpus_id
    with torch.no_grad():
        generated_ranking_str = node_list 
        Z_count = generated_ranking_str.count('Z')
        prompt_tokenID = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                )['input_ids']
        generated_ranking_tokenID = tokenizer(
                    generated_ranking_str,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True, 
                    add_special_tokens=False
                )['input_ids']
        #input_ids
        len_prompt = prompt_tokenID.size()[1]
        combined_tokenID = torch.concat([prompt_tokenID,generated_ranking_tokenID],dim=-1).cuda()
        # tree_structure
        tree_structure_concate = [i-1 for i in range(len_prompt)] + [ i + len_prompt for i in tree_structure]
        position_structure_concate = torch.tensor([[i for i in range(len_prompt)] + [ i + len_prompt for i in position]]).cuda()
        
        # 推理
        # print([combined_tokenID.shape,len(tree_structure_concate),position_structure_concate.shape])
        outputs =  tree_llm(input_ids=combined_tokenID,tree_structure=tree_structure_concate[:combined_tokenID.shape[1]],position_ids=position_structure_concate[:,:combined_tokenID.shape[1]], output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 形状是 [batch_size, sequence_length, hidden_size]
        #找到对应的prob：
        end_indices = [(i) * 4 + 3  + len_prompt for i in range(int(len(tree_structure)/4 - Z_count))]
        pred_indices = [tree_structure_concate.index(index) for index in end_indices]


        # start_indices = [(i+1) * 4  + len_prompt for i in range(int(len(tree_structure)/4))] # 关键节点（下一个要预测char）的位置
        start_token_embedding = hidden_states[:, pred_indices , :]  # 选择最后一个token的嵌入  [batch_size, start_indices, hidden_size]
        logits = tree_llm.llm.lm_head(start_token_embedding) #[batch_size, start_indices,all_tokne] 
        
        prob = torch.softmax(logits[:,:,corpus_id],dim=-1).detach() # 对应概率 [batch_size, start_indices,all_tokne]
        return prob, start_token_embedding


from eval_func import *

def evaluate_tree(test_dataloader,tree_llm,tokenizer,actor,step = 2,eval_mode = 'fast'):
    ranking_save = []
    spearman_save = []
    footrule_save = []
    kemeny_save = []
    jaccard_at_10_save = []
    jaccard_at_15_save = []
    for index,batch in tqdm(enumerate(test_dataloader)):

        batch_size = len(batch[0])
        prompts = list(batch[0])
        GT_label = [extract_from_string(label) for label in batch[1]] # [:-10] remove eos 
        label_corpus = sorted(GT_label[0])
        # get predicted ranking;
        current_sequence = []
        logit =  initial_step_logit(tree_llm.llm,tokenizer,prompts,label_corpus,return_hidden=True)[0]
        candidate_sequence,logits_action = actor.drafting(current_sequence,logit)

        for j in range( step - 1):
            node_name,node_list,tree_structure,position = generate_tree(candidate_sequence,len(current_sequence) + 4)
            prob_mat, hidden_state = ranking_prob_tree_mask(actor,tree_llm,tokenizer,prompts,node_list ,tree_structure,position,node_name) #第2/3次过LLMs
            
            
            prob_mat, hidden_state = prob_mat[0].to(torch.float32).cpu().detach().numpy(), hidden_state[0].detach() 
            current_sequence,index_ = actor.verifying(prob_mat, candidate_sequence,node_name)
            # current_sequence = current_sequence + accepted_sequence
            logit = hidden_state[index_,:]
            if j == step - 2:
                output_sequence = actor.drafting(current_sequence,logit,last=True)
            else:
                candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
            # print(current_sequence)
            if len(current_sequence) == actor.vocab_size:
                output_sequence = current_sequence
                break
        generated_label = output_sequence        #get the initial state
        target_label_list = GT_label
        output_label_list = [generated_label]
        for i in range(len(target_label_list)):
        #other error:
            target_label = target_label_list[i]
            output_label = output_label_list[i]
            ranking_save.append(ranking_distance(target_label,output_label))
            spearman_save.append(spearman_distance(target_label,output_label))
            footrule_save.append(footrule_distance(target_label,output_label))
            kemeny_save.append(kemeny_distance(target_label,output_label))
            jaccard_at_10_save.append(recall_at_k(target_label,output_label,10))
            jaccard_at_15_save.append(ndcg_at_k(target_label,output_label,10))
        
        if eval_mode=='fast' and index == 100:
            return [np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(jaccard_at_10_save),np.mean(jaccard_at_15_save)]
    return [np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(jaccard_at_10_save),np.mean(jaccard_at_15_save)]






def save_results(args, metrics):
    """Save results to result.log in current directory"""
    with open("result.log", "a") as f:
        # First line: model_name = blockwise, data, backbone, step
        f.write(f"model_name = Mesuda, {args.data_name}, {args.backbone}, {args.n_step}\n")
        # Second line: results
        f.write(f"{','.join(map(str, metrics))}\n")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    # if args.data_name == 'rank_zephyr_IR_20':
    #     args.learning_rate = 5e-4
    # Initialize models
    # Set full model path before initialization
    args.model_path = os.path.join(args.model_path, args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # llm = TreeMaskLlama.from_pretrained(args.model_path).cuda()

    if args.backbone == "Llama-3.2-3B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map='cuda',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            # attn_implementation='flash_attention_2',
        )
    tree_llm = TreeMaskLlama(model)# 
    
    # Load datasets
    data_path = f"./dataset/{args.data_name}/{args.backbone}/"
    train_data,valid_data,test_data= load_datasets(data_path)
    train_dataset = GenerationDataset(train_data, tokenizer, combined=False)
    train_dataloader = DataLoader(train_dataset, batch_size=16,shuffle=True)
    valid_dataset = GenerationDataset(valid_data, tokenizer, combined=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataset = GenerationDataset(test_data, tokenizer, combined=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize actor
    n_output = len(extract_from_string(test_dataset[0][1]))
    hidden_size = tree_llm.llm.config.hidden_size
    actor = Actor(args,hidden_size,n_output,tree_llm,tokenizer)
    actor = actor.bfloat16() if args.backbone != "Llama-3.2-3B-Instruct" else actor
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    # 初始化训练状态
    best_ranking = float('-inf')  # 初始为负无穷

    dataloader_iter = enumerate(train_dataloader)
    for batch_idx, batch in tqdm(dataloader_iter):
        
    
        batch_action_sequences = [] 
        # sample for each query  
        # 
        batch_loss = []
        for i in range(len(batch[0])):
            prompt = batch[0][i]
            GT_label = extract_from_string(batch[1][i]) # [:-10] remove eos 
            label_corpus = sorted(GT_label)
            label = torch.tensor(get_label_ranking(GT_label)).cuda()
            # THE FIRST inference and matrix
            current_sequence = []
            logit =  initial_step_logit(tree_llm.llm,tokenizer,[prompt],label_corpus,return_hidden=True)[0]
            candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
            batch_loss.append(criterion(logits_action, label))
            
            # generated_label = greedy_ranking_init(llm, tokenizer, [prompt], label_corpus) #第一次过LLMs
            for j in range(args.n_step):
                node_name,node_list,tree_structure,position = generate_tree(candidate_sequence,len(current_sequence) + 4)
                prob_mat, hidden_state = ranking_prob_tree_mask(actor,tree_llm,tokenizer,prompt,node_list ,tree_structure,position,node_name) #第2/3次过LLMs
                
                
                prob_mat, hidden_state = prob_mat[0].to(torch.float32).cpu().detach().numpy(), hidden_state[0].detach() 
                current_sequence,index = actor.verifying(prob_mat, candidate_sequence,node_name)
                # current_sequence = current_sequence + accepted_sequence
                logit = hidden_state[index,:]
                candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
                batch_loss.append(criterion(logits_action, label[len(current_sequence):]))
                # print(current_sequence)
                if len(current_sequence) == actor.vocab_size:
                    break
        
            output_sequence = actor.drafting(current_sequence,logit,last=True)
            # print(current_sequence)
        
        batch_loss = torch.mean(torch.stack(batch_loss))
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 5 == 0:
            actor.set_eval()
            time_start = time.time()
            metrics = evaluate_tree(test_dataloader, tree_llm, tokenizer, actor, args.n_step,eval_mode='nonfast')
            ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15 = metrics
            time_end = time.time()
            metrics.append((time_end - time_start)/len(test_dataloader))
            print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15,metrics[-1]])
            
            actor.set_train()
            if ranking > best_ranking:
                best_ranking = ranking
                metrics_save = metrics
                tol = 0 
            else:
                tol = tol + 1
            if tol==2:
                break



    # Evaluation
    
    save_results(args, metrics_save)
    print(f"Results saved to result.log: {metrics_save}")

if __name__ == "__main__":
    main()