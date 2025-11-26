
import torch
#数据读取
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from eval_func import ranking_distance, spearman_distance, footrule_distance,kemeny_distance,recall_at_k,ndcg_at_k
from grop_func import greedy_ranking_init,ranking_prob,initial_step_logit
from utils import GenerationDataset, extract_from_string
# Custom imports
from utils import load_datasets, initialize_models
import numpy as np  


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
    parser.add_argument("--d_model", type=int, default=25)
    parser.add_argument("--nhead", type=int, default=5)
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
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        # 注意：实际上你只用了 linear 层做投影或者根本没用内部模型，
        # 如果完全依赖外部 LLM 的 logit，这里其实主要是个 Helper 类
        self.index2char = {i: chr(i+ord('A')) for i in range(25)}
        self.char2index = {chr(i+ord('A')):i for i in range(25)}

    def Parallel_Decoding_action(self, batch_generated_label, matrix):
            """
            实现论文中的 Parallel Jacobi 逻辑：
            一次性利用 Matrix 更新所有位置，而不是像 longest_greedy 那样被前缀锁死。
            """
            # matrix shape: [Batch, Seq_Len, Vocab_Size]
            if not torch.is_tensor(matrix):
                matrix = torch.tensor(matrix).cuda() 
            
            # --- FIX START ---
            # 动态获取当前矩阵的词表大小 (可能是 20, 100 等，不一定是 25)
            vocab_size = matrix.size(-1)
            
            # k 不能超过 vocab_size。如果 vocab_size < 25，就取 vocab_size 全量排序
            k = vocab_size 
            
            # topk 排序
            top_k_vals, top_k_indices = torch.topk(matrix, k=k, dim=-1)
            # --- FIX END ---
            
            # 转回 CPU 进行轻量级的去重分配
            top_k_indices_np = top_k_indices.cpu().numpy()
            
            action_list = []
            
            # 2. 对 Batch 中的每个样本进行去重处理
            for b_idx in range(len(batch_generated_label)):
                current_indices = top_k_indices_np[b_idx] # Shape: [Seq_Len, k]
                seq_len = len(batch_generated_label[b_idx])
                
                new_sequence = []
                used_items = set()
                
                for pos in range(seq_len):
                    # 获取该位置的候选列表
                    candidates = current_indices[pos]
                    
                    found = False
                    for cand_idx in candidates:
                        # 只有当索引在你的映射表中存在时才转换 (防止越界)
                        if cand_idx in self.index2char:
                            char = self.index2char[cand_idx]
                            if char not in used_items:
                                new_sequence.append(char)
                                used_items.add(char)
                                found = True
                                break
                    
                    # 兜底：如果没找到（比如所有候选都被用了），或者索引超出了 index2char 范围
                    if not found:
                        # 动态生成所有可能的字符集合
                        # 假设有效字符是 A, B, ... 到 vocab_size
                        valid_chars = {self.index2char[i] for i in range(vocab_size) if i in self.index2char}
                        remain = list(valid_chars - used_items)
                        if remain:
                            char = remain[0]
                            new_sequence.append(char)
                            used_items.add(char)
                        else:
                            # 极端情况：所有都用光了（理论不应发生），填个默认值防止崩溃
                            new_sequence.append('A') 

                action_list.append(new_sequence)
                
            return action_list

    
def evaluate_ranking_model(test_dataloader, llm, tokenizer, actor, step=3):
    
    ranking_save = []
    spearman_save = []
    footrule_save = []
    kemeny_save = []
    jaccard_at_10_save = []
    jaccard_at_15_save = []
    for index, batch in tqdm(enumerate(test_dataloader), desc="Evaluating"):
        prompts = list(batch[0])
        GT_label = [extract_from_string(label) for label in batch[1]]
        label_corpus = sorted(GT_label[0])

        # 初始预测
        _ = initial_step_logit(llm, tokenizer, prompts, label_corpus)
        generated_label = greedy_ranking_init(llm, tokenizer, prompts, label_corpus)

        # 多轮推理优化
        for _ in range(step - 1):
            prob_mat = ranking_prob(llm, tokenizer, prompts, generated_label)
            generated_label = actor.Parallel_Decoding_action(generated_label, prob_mat)

        # 评估指标计算
        for target_label, output_label in zip(GT_label, generated_label):
            ranking_save.append(ranking_distance(target_label, output_label))
            spearman_save.append(spearman_distance(target_label, output_label))
            footrule_save.append(footrule_distance(target_label, output_label))
            kemeny_save.append(kemeny_distance(target_label,output_label))
            jaccard_at_10_save.append(recall_at_k(target_label,output_label,10))
            jaccard_at_15_save.append(ndcg_at_k(target_label,output_label,10))
    return [np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(jaccard_at_10_save),np.mean(jaccard_at_15_save)]


def save_results(args, metrics):
    """Save results to result.log in current directory"""
    with open("result.log", "a") as f:
        # First line: model_name = blockwise, data, backbone, step
        f.write(f"model_name = Parallel_decoding, {args.data_name}, {args.backbone}, {args.n_step}\n")
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
    actor = Actor(args=args)
    actor = actor.bfloat16() if args.backbone != "Llama-3.2-3B-Instruct" else actor
    # 初始化训练状态
 
    # Evaluation
    actor.eval()    
    time_start = time.time()
    metrics = evaluate_ranking_model(test_dataloader, llm, tokenizer, actor, step=args.n_step)
    time_end = time.time()
    print(f"Time cost: {time_end - time_start} seconds")
    metrics.append((time_end - time_start)/len(test_dataloader))
    save_results(args, metrics)
    print(f"Results saved to result.log: {metrics}")

if __name__ == "__main__":
    main()