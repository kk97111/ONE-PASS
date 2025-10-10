from scipy.stats import kendalltau,spearmanr
from tqdm import tqdm
import torch.nn.functional as F
from utils import extract_from_string
import numpy as np
from tqdm import tqdm
from typing import List

def recall_at_k(ground_truth,ranking2, k=10):
    ground_truth = ground_truth[:k]
    top_k = ranking2[:k]
    hits = sum([1 for item in top_k if item in ground_truth])
    return hits / len(ground_truth)

def ndcg_at_k(ground_truth,ranking2, k=10):
    ground_truth = ground_truth[:k]
    dcg = 0.0
    for i, item in enumerate(ranking2[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)  # log2(i+2) because index i starts from 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(ground_truth), k))])
    return dcg / idcg if idcg > 0 else 0.0

def jaccard_at_k(ranking1: List[str], ranking2: List[str], k: int) -> float:
    """
    Compute Jaccard@K between two ranking lists.
    
    :param ranking1: First ranked list (e.g., prediction)
    :param ranking2: Second ranked list (e.g., ground truth or another prediction)
    :param k: Cutoff rank k
    :return: Jaccard@K score between 0 and 1
    """
    set1 = set(ranking1[:k])
    set2 = set(ranking2[:k])
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    return len(intersection) / len(union) if union else 0.0



def kemeny_distance(r1, r2):
    inv1 = {x: i for i, x in enumerate(r1)}
    inv2 = {x: i for i, x in enumerate(r2)}
    n = len(r1)
    distance = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = r1[i], r1[j]
            if (inv1[a] < inv1[b]) != (inv2[a] < inv2[b]):
                distance += 1
    return distance

def ranking_distance(ranking1, ranking2):
    rank_map = {item: idx for idx, item in enumerate(ranking1)}
    rank_num1 = [rank_map[item] for item in ranking1]
    rank_num2 = [rank_map[item] for item in ranking2]
    
    tau, _ = kendalltau(rank_num1, rank_num2)
    return tau * 0.5 + 0.5

def spearman_distance(ranking1, ranking2):
    rank_map = {item: idx for idx, item in enumerate(ranking1)}
    rank_num1 = [rank_map[item] for item in ranking1]
    rank_num2 = [rank_map[item] for item in ranking2]

    rho, _ = spearmanr(rank_num1, rank_num2)
    return rho
def footrule_distance(ranking1, ranking2):
    rank_map = {item: idx for idx, item in enumerate(ranking1)}
    return sum(abs(rank_map[item] - idx) for idx, item in enumerate(ranking2))

def evaluate(test_dataloader,actor,mode ,heuristic ,eval_mode = 'fast'):
    
    ranking_save = []
    spearman_save = []
    footrule_save = []
    kemeny_save = []
    recall_at_k_save = []
    ndcg_at_k_save = []
    for index,batch in tqdm(enumerate(test_dataloader)):
        
        prompts = list(batch[0])#每个batch揪一个prompt
        GT_label = [extract_from_string(label) for label in batch[1]][0]
        # 计算每个token的每个候选词的概率
        _,prob_np,search_path_dict = actor.hidden_state_prob(prompts,mode = mode)
        generated_label_list = []
        for i in range(len(GT_label)):
            # print(generated_label_list)
            candidate_items = list(set(GT_label) - set(generated_label_list))
            generated_label_list = actor.generate(generated_label_list,candidate_items,prob_np,search_path_dict,mode,heuristic)

        
        target_label_list = GT_label
        output_label_list = generated_label_list
        #other error:
        target_label = target_label_list
        output_label = output_label_list
        ranking_save.append(ranking_distance(target_label,output_label))
        spearman_save.append(spearman_distance(target_label,output_label))
        footrule_save.append(footrule_distance(target_label,output_label))
        kemeny_save.append(kemeny_distance(target_label,output_label))
        recall_at_k_save.append(recall_at_k(target_label,output_label,10))
        ndcg_at_k_save.append(ndcg_at_k(target_label,output_label,10))
        
        if eval_mode=='fast' and index == 100:
            return np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(recall_at_k_save),np.mean(ndcg_at_k_save)
    return np.mean(ranking_save),np.mean(spearman_save),np.mean(footrule_save),np.mean(kemeny_save),np.mean(recall_at_k_save),np.mean(ndcg_at_k_save)
def evaluate_Blockwise(test_dataloader,llm,tokenizer,actor,step = 2,eval_mode = 'fast'):
    from grop_func import ranking_prob,initial_step_logit
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
        logit =  initial_step_logit(llm,tokenizer,prompts,label_corpus,return_hidden=True)[0]
        candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
        for j in range(step-1):
            prob_mat, hidden_state = ranking_prob(llm, tokenizer, prompts, [current_sequence + candidate_sequence],return_hidden=True) #第2/3次过LLMs
            prob_mat, hidden_state = prob_mat[0][len(current_sequence):].float().cpu().detach().numpy(), hidden_state[0].detach() 
            accepted_sequence = actor.verifying(prob_mat,candidate_sequence)
            current_sequence = current_sequence + accepted_sequence
            logit = hidden_state[len(current_sequence),:]
            candidate_sequence,logits_action = actor.drafting(current_sequence,logit)
            if len(current_sequence) == actor.vocab_size:
                break
        generated_label = current_sequence + candidate_sequence        #get the initial state
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


