
import torch
#数据读取
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from utils import GenerationDataset, extract_from_string,load_datasets, initialize_models
# Custom imports
import time
from tqdm import tqdm
from eval_func import ranking_distance, spearman_distance, footrule_distance,kemeny_distance,recall_at_k,ndcg_at_k
from grop_func import greedy_ranking_init,ranking_prob
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
    parser.add_argument("--d_model", type=int, default=255)
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


def rejection_sampling(prob_mat, label_corpus, generated_label, start_index):
    """
    使用拒绝采样方式生成标签序列。
    
    参数:
    - prob_mat: Tensor，形状为 (seq_len, num_classes)，表示每个位置的概率分布。
    - label_corpus: List[str]，所有可能的标签。
    - generated_label: List[str]，当前已生成的标签序列。
    - start_index: int，从这个位置开始采样。

    返回:
    - generated_label: List[str]，更新后的生成标签序列。
    - start_index: int，更新后的起始索引。
    """
    length = prob_mat.size(0)
    if start_index == length-1:
        return generated_label, start_index


    # 将已有的标签映射为索引
    index_generation = torch.tensor([[ord(char) - ord('A')] for char in generated_label]).cuda()
    p_list = torch.gather(prob_mat, 1, index_generation).squeeze(1)
    q_list = prob_mat[start_index]

    # 拒绝采样
    r = torch.rand(length).cuda()
    n_accepted = 0
    for j, i in enumerate(range(start_index, min(length, start_index + 5))):
        pi = p_list[i]   # p_{i+1}(x_i)
        qi = q_list[i]   # q_i(x_i)
        if r[i] > (pi / qi):
            n_accepted = j
            break

    start_index += n_accepted + 1
    existing_ranking = generated_label[:start_index]

    # 如果已经是最后一个 token，就直接返回
    if start_index == length:
        return existing_ranking, start_index

    # 重新构造分布用于生成下一个 token
    logits = prob_mat[start_index].tolist()
    pairs = list(zip(logits, label_corpus))
    pairs_sorted = sorted(pairs, key=lambda x: x[0], reverse=True)
    sorted_items = [i for _, i in pairs_sorted]
    sorted_items_after_remove = [x for x in sorted_items if x not in existing_ranking]
    generated_label = existing_ranking + sorted_items_after_remove

    return generated_label, start_index



def save_results(args, metrics):
    """Save results to result.log in current directory"""
    with open("result.log", "a") as f:
        # First line: model_name = blockwise, data, backbone, step
        f.write(f"model_name = SD, {args.data_name}, {args.backbone}, {args.n_step}\n")
        # Second line: results
        f.write(f"{','.join(map(str, metrics))}\n") # 

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

    ranking_save = []
    spearman_save = []
    footrule_save = []
    kemeny_save = []
    jaccard_at_10_save = []
    jaccard_at_15_save = []
    time_start = time.time()
    for index, batch in tqdm(enumerate(test_dataloader), desc="Evaluating"):
        
        prompts = list(batch[0])
        GT_label = [extract_from_string(label) for label in batch[1]]
        label_corpus = sorted(GT_label[0])
        length = len(label_corpus)

        start_index = 0 
        # 初始预测
        generated_label = greedy_ranking_init(llm, tokenizer, prompts, label_corpus)[0]

        # 多轮推理优化
        for _ in range(args.n_step - 1):
            prob_mat = ranking_prob(llm, tokenizer, prompts, [generated_label])[0]
            generated_label, start_index = rejection_sampling(prob_mat, label_corpus, generated_label, start_index)
            # 评估指标计算
        for target_label, output_label in zip(GT_label, [generated_label]):
            ranking_save.append(ranking_distance(target_label, output_label))
            spearman_save.append(spearman_distance(target_label, output_label))
            footrule_save.append(footrule_distance(target_label, output_label))
            kemeny_save.append(kemeny_distance(target_label,output_label))
            jaccard_at_10_save.append(recall_at_k(target_label,output_label,10))
            jaccard_at_15_save.append(ndcg_at_k(target_label,output_label,10))
    time_end = time.time()
    metrics = [np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(jaccard_at_10_save),np.mean(jaccard_at_15_save)]
    metrics.append((time_end - time_start)/len(test_dataloader))
    save_results(args, metrics)
    print(f"Results saved to result.log: {metrics}")

if __name__ == "__main__":
    main()