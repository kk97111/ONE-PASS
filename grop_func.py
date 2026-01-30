# 将基于生成(one-step to all ranking)的prompt和结果链接起来~

import numpy as np
import torch
import gc
import torch.nn.functional as F
from eval_func import spearman_distance,footrule_distance
import torch.nn as nn
from itertools import product

def build_prefix_tree(lists):
    """
    输入：
    lists = [
        [1, 2, 3],
        [1, 2, 4],
        [1, 5], 
        [6]
    ]
    输出：
    tree_list = [-1, 0, 1, 1, 0, -1]     # 父节点索引
    values    = [1, 2, 3, 4, 5, 6]       # 每个节点的值
    search_path     = [
        [0, 1, 2],      # 对应 [1, 2, 3]
        [0, 1, 3],      # 对应 [1, 2, 4]
        [0, 4],         # 对应 [1, 5]
        [5]             # 对应 [6]
    ]
    """
    tree_list = []
    values = []
    depths = []
    node_map = {}  # key: (parent_idx, value), value: node_idx
    search_path = [] # 每个节点的搜索路径

    for lst in lists:
        path = []
        parent = -1
        depth = 0
        for val in lst:
            key = (parent, val)
            if key not in node_map:
                node_idx = len(tree_list)
                node_map[key] = node_idx
                tree_list.append(parent)
                values.append(val)
                depths.append(depth)
            else:
                node_idx = node_map[key]
            path.append(node_idx)
            parent = node_idx
            depth += 1
        search_path.append(path)

    return tree_list, values, depths, search_path
def greedy_ranking_init(model,tokenizer,prompts,label_corpus,add_stick=True):
    prob_np =  initial_step_logit(model,tokenizer,prompts,label_corpus,add_stick) #[batch,n_doc]
    # Get the sorted indices in descending order
    sorted_indices = prob_np.argsort()[:,::-1]
    # Sort corpus based on the indices
    sorted_corpus = [[label_corpus[i] for i in line] for line in sorted_indices]
    return sorted_corpus
def ranking_prob(llm,tokenizer,prompt,generate_label_list,return_hidden=False):
    # 这个函数输入prompt和ranking，已知位置i之前的ranking,输出位置i对应的生成的概率；P(X_i|X_1,...,X_{i-1});
    # generate_label_list是已经排序好的，要接着生成下一个位置的ranking
    #step:分别对prompt和ranking进行tokenize
    batch_size = len(generate_label_list)
    label_size = len(generate_label_list[0])
    corpus_id = sorted(tokenizer.convert_tokens_to_ids(generate_label_list[0])) #| 
    with torch.no_grad():
        generated_ranking_str = [' > '.join(f"|{element}|" for element in generate_label) for generate_label in generate_label_list]
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
        with torch.no_grad():
            len_prompt = prompt_tokenID.size()[1]
            combined_tokenID = torch.concat([prompt_tokenID,generated_ranking_tokenID],dim=-1).cuda()
            # 推理
            outputs =  llm(combined_tokenID, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 形状是 [batch_size, sequence_length, hidden_size]
            #找到对应的prob：
            start_indices = [i * 4 + len_prompt for i in range(label_size)] # 关键节点（下一个要预测char）的位置
            start_token_embedding = hidden_states[:, start_indices , :]  # 选择最后一个token的嵌入  [batch_size, start_indices, hidden_size]
            logits = llm.lm_head(start_token_embedding) #[batch_size, start_indices,all_tokne] 
            
            prob = torch.softmax(logits[:,:,corpus_id],dim=-1).detach() # 对应概率 [batch_size, start_indices,all_tokne]
    if return_hidden:
        return prob, start_token_embedding
    else:
        return prob

def spearman_footrule_distance(rank1, rank2):
    """
    计算 Spearman Footrule 距离
    :param rank1: 第一组排名（列表）
    :param rank2: 第二组排名（列表）
    :return: 距离值
    """
    # 创建元素到索引的映射
    rank1_dict = {item: i for i, item in enumerate(rank1)}
    rank2_dict = {item: i for i, item in enumerate(rank2)}

    # 计算所有元素的排名差的绝对值之和
    distance = np.mean([abs(rank1_dict[item] - rank2_dict[item]) for item in rank1])
    
    return distance


def get_label_ranking(generated_label):
    ranking_dict = {label: idx + 1 for idx, label in enumerate(generated_label)}
    sorted_labels = sorted(ranking_dict.keys())
    sorted_ranking = [ranking_dict[label] for label in sorted_labels]
    return np.array(sorted_ranking)

def our_rank_net_loss(y_pred, y_true, rewards = False, weighted=False, use_rank=True, weight_by_diff=False,
             weight_by_diff_powed=False):
    """
    RankNet loss for listwise ranking (optimizes predicted scores to match true ranks).
    
    Args:
        y_pred: Predicted scores, shape [batch_size, slate_length] (e.g., [batch_size, 26] for A-Z).
        y_true: True ranks (lower = better), shape [batch_size, slate_length] (e.g., [3, 1, 2] means B > C > A).
        weighted: If True, weight pairs by inverse rank sum (prioritize top ranks).
        use_rank: If True, convert y_true to relevance scores (1/rank).
        weight_by_diff: Weight pairs by true rank differences.
        weight_by_diff_powed: Weight pairs by squared true rank differences.
    
    Returns:
        Loss value (torch.Tensor).
    """
    if use_rank:
        # Convert ranks to relevance scores (1/rank, higher = better)
        y_true = 1.0 / (y_true.float() + 1)  # +1 to avoid division by zero
    
    # Generate all possible document pairs (i, j)
    slate_length = y_true.shape[1]
    document_pairs_candidates = list(product(range(slate_length), repeat=2))
    
    # Get true and predicted differences for all pairs
    pairs_true = y_true[:, document_pairs_candidates]  # shape [batch_size, n_pairs, 2]  (i,j)对应的真实得分
    pairs_pred = y_pred[:, document_pairs_candidates]  # shape [batch_size, n_pairs, 2]  （i，j）对应的真实位置
    
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]  # true_i - true_j :(i,j)对应的真实得分之差
    pred_diffs = pairs_pred[:, :, 0] - pairs_pred[:, :, 1]  # pred_i - pred_j :(i,j)对应的真实位置之差
    
    # Mask: Only keep pairs where true_i > true_j (i should rank higher than j)
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs)) #只保留得分前面大于后面的
    
    pred_diffs = pred_diffs[the_mask] #取出独影
    true_diffs_binary = (true_diffs > 0).float()[the_mask]  # 1 if i > j, else 0
    
    # Weighting strategies
    weight = None
    if weighted:
        # Weight by inverse rank sum (prioritize misranked top pairs)
        ranks = y_true.argsort(dim=1).argsort(dim=1) + 1  # Convert to ranks (1=best)
        rank_pairs = ranks[:, document_pairs_candidates]  # shape [batch_size, n_pairs, 2]
        rank_sum = rank_pairs.sum(dim=-1)  # rank_i + rank_j
        weight = 1.0 / rank_sum[the_mask]  # Higher weight for top-ranked pairs
    elif weight_by_diff:
        # Weight by absolute true rank difference
        weight = torch.abs(true_diffs)[the_mask]
    elif weight_by_diff_powed:
        # Weight by squared true rank difference
        weight = torch.pow(true_diffs, 2)[the_mask]
    
    # Compute pairwise BCE loss
    if rewards is False:
        loss = nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs_binary)
    else:
        rewards = rewards.unsqueeze(-1).repeat(1,y_true.shape[1]**2)
        if weighted:
            loss = torch.mean(nn.BCEWithLogitsLoss(reduction='none')(pred_diffs, true_diffs_binary) * weight * rewards[the_mask])
        else:
            loss = torch.mean(nn.BCEWithLogitsLoss(reduction='none')(pred_diffs, true_diffs_binary) * rewards[the_mask])
    return loss

def initial_step_logit(model,tokenizer,prompts,corpus,add_stick = True, return_hidden = False):
    # for lm_head
    if add_stick:
        text = [prompt + "|" for prompt in prompts]
    else:
        text = prompts#[prompt + "|" for prompt in prompts]
    temperature = 1.0
    input_encoding = tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
            )
    with torch.no_grad():
        input_ids = input_encoding['input_ids'].cuda()
            # 获取模型的输出
        outputs =  model(input_ids, output_hidden_states=True)
        # outputs是一个元组，其中包含多个元素
        # 第一个元素是最后的隐藏状态（也就是嵌入），
        hidden_states = outputs.hidden_states[-1]  # 形状是 [batch_size, sequence_length, hidden_size]
        # 如果你想查看最后一个token的嵌入
        last_token_embedding = hidden_states[:, -1, :]  # 选择最后一个token的嵌入
        token_corpus_id = tokenizer.convert_tokens_to_ids(corpus)
        wanted_weight = model.lm_head.weight[token_corpus_id, :]  # [num_wanted_tokens, hidden_size]
        logits = F.linear(last_token_embedding, wanted_weight)
        if temperature > 0:
            logits = logits / temperature  # 应用温度参数调整logits
        
        prob_corpus = torch.softmax(logits,dim=-1)
        prob_np = prob_corpus.float().cpu().detach().numpy()
    if return_hidden:
        return last_token_embedding
    else:
        return prob_np