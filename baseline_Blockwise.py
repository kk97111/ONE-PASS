
import torch
#数据读取
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from grop_func import ranking_prob, initial_step_logit
from utils import GenerationDataset, extract_from_string,load_datasets, initialize_models
# Custom imports
from eval_func import evaluate_Blockwise
import copy

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--backbone", type=str, required=True,
                      help="Base model name (e.g. 'Llama-3.2-3B-Instruct')")
    parser.add_argument("--data_name", type=str, required=True,
                      help="Dataset name (e.g. 'rank_zephyr_IR_20')")
    parser.add_argument("--n_step", type=int, default=7,
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

class Actor(nn.Module):
    def __init__(self, args,hidden_size,vocab_size,llm,tokenizer):
        super(Actor, self).__init__()
        self.args = args
        self.vocab_size=vocab_size
        self.hidden_size = hidden_size
        self.llm = llm
        self.token_corpus_id = tokenizer.convert_tokens_to_ids([chr(i+ord('A')) for i in range(vocab_size)])

        self.hidden_layer = nn.Linear(self.hidden_size, self.args.d_model).cuda()
        self.output_layers = nn.ModuleList([
            nn.Linear(self.args.d_model, self.hidden_size).cuda() for _ in range(vocab_size)
        ])

        self.index2char = {i: chr(i+ord('A')) for i in range(25)}
        self.char2index = {chr(i+ord('A')):i for i in range(25)}
        self.corpus = set([self.index2char[i] for i in range(self.vocab_size)])
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




    def sampled_action(self, current_sequence, logits):
        next_docs = []
        remain_doc_indices = [self.char2index[doc] for doc in set(self.corpus - set(current_sequence))] #对应剩下的index；
        for i,line in enumerate(logits):
            select_doc_index = remain_doc_indices[np.argmax(line[remain_doc_indices])] #最大index
            next_docs.append(self.index2char[select_doc_index])
            remain_doc_indices.remove(select_doc_index)
        return next_docs
            
            




               
    def drafting(self,current_sequence,original_decoder_output):
        # for this action, it receives a batch of
        #batch_matrices -->[batch,25,25]
        hidden_state = self.hidden_layer(original_decoder_output)#[d]
        multi_head_hidden_state = [original_decoder_output]
        for i in range(1,self.vocab_size):
            multi_head_hidden_state.append(self.output_layers[i](hidden_state) + original_decoder_output) 
        multi_head_hidden_state = torch.stack(multi_head_hidden_state,dim=0)#[K,hidden_size]
        

        wanted_weight = self.llm.lm_head.weight[self.token_corpus_id, :].detach()  # [num_wanted_tokens, hidden_size]
        remain_logits = F.linear(multi_head_hidden_state, wanted_weight)[:self.vocab_size-len(current_sequence),:]#[K,prob]

        remain_prob = torch.softmax(remain_logits,dim=-1).to(torch.float32).cpu().detach().numpy() #[K,prob]
        # 行为抽样
        next_docs = self.sampled_action(current_sequence, remain_prob)

        return next_docs,remain_logits
    def verifying(self,prob_mat,candidate_sequence,topK=5):
        for i,doc in enumerate(candidate_sequence):
            doc_index = self.char2index[doc]
            rank = np.argsort(-prob_mat[i]).tolist().index(doc_index)
            if rank > topK-1:
                break
        return candidate_sequence[:i]





def save_results(args, metrics):
    """Save results to result.log in current directory"""
    with open("result.log", "a") as f:
        # First line: model_name = blockwise, data, backbone, step
        f.write(f"model_name = blockwise, {args.data_name}, {args.backbone}, {args.n_step}\n")
        # Second line: results
        f.write(f"{','.join(map(str, metrics))}\n")

def main():
    args = parse_args()

    # Initialize models
    # Set full model path before initialization
    args.model_path = os.path.join(args.model_path, args.backbone)
    llm, tokenizer = initialize_models(args)
    
    # Load datasets
    data_path = f"./dataset/{args.data_name}/{args.backbone}/"
    train_data,valid_data,test_data= load_datasets(data_path)
    train_dataset = GenerationDataset(train_data, tokenizer, combined=False)
    train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle=True)
    valid_dataset = GenerationDataset(valid_data, tokenizer, combined=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataset = GenerationDataset(test_data, tokenizer, combined=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize actor
    n_output = len(extract_from_string(test_dataset[0][1]))
    actor = Actor(args, llm.config.hidden_size, n_output, llm, tokenizer)
    actor = actor.bfloat16() if args.backbone != "Llama-3.2-3B-Instruct" else actor
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    # 初始化训练状态
    best_ranking = float('-inf')  # 初始为负无穷


    for epoch in range(0, 1):
        # 跳过已经处理过的batch
        dataloader_iter = enumerate(train_dataloader)
        for batch_idx, batch in tqdm(dataloader_iter):
            
        
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
                logit =  initial_step_logit(llm,tokenizer,[prompt],label_corpus,return_hidden=True)[0]
                candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
                batch_loss.append(criterion(logits_action, label))
                # generated_label = greedy_ranking_init(llm, tokenizer, [prompt], label_corpus) #第一次过LLMs
                for j in range(args.n_step-1):
                    prob_mat, hidden_state = ranking_prob(llm, tokenizer, [prompt], [current_sequence + candidate_sequence],return_hidden=True) #第2/3次过LLMs
                    prob_mat, hidden_state = prob_mat[0][len(current_sequence):].to(torch.float32).cpu().detach().numpy(), hidden_state[0].detach() 
                    accepted_sequence = actor.verifying(prob_mat,candidate_sequence)
                    current_sequence = current_sequence + accepted_sequence
                    logit = hidden_state[len(current_sequence),:]
                    candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
                    batch_loss.append(criterion(logits_action, label[len(current_sequence):]))

                    if len(current_sequence) == actor.vocab_size:
                        break
                generated_label = current_sequence + candidate_sequence
                    
            batch_loss = torch.mean(torch.stack(batch_loss))
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 5 == 0:
                actor.set_eval()
                ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate_Blockwise(valid_dataloader, llm, tokenizer, actor,args.n_step)
                print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15])
                actor.set_train()
                if ranking > best_ranking:
                    best_ranking = ranking
                    actor_copy = copy.deepcopy(actor)
                    tol = 0 
                else:
                    tol = tol + 1
                if tol==2:
                    break



    # Evaluation
    actor.eval()
    with torch.no_grad():
        import time
        time_start = time.time()
        metrics = evaluate_Blockwise(test_dataloader, llm, tokenizer, actor_copy, args.n_step, eval_mode='nonfast')
        time_end = time.time()
        print(f"Time cost: {time_end - time_start} seconds")
    metrics.append((time_end - time_start)/len(test_dataloader))
    save_results(args, metrics)
    print(f"Results saved to result.log: {metrics}")

if __name__ == "__main__":
    main()