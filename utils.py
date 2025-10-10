from torch.utils.data import Dataset,Sampler
from ftfy import fix_text
from typing import Dict, Sequence
import transformers
import copy
import torch
import numpy as np
from itertools import product
import torch.nn as nn
import re
import os
from datasets import load_from_disk
import logging
from datetime import datetime
from collections import defaultdict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

IGNORE_INDEX = -100


START_IDX = ord('A')
def get_Notice(label):
    label_len = len([char for char in label if char.isalpha()])
    corpus = [chr(i) for i in range(START_IDX, START_IDX+label_len)]
    Notice = f" The valid identifiers are: |{corpus[0]}| ~ |{corpus[-1]}| !!! Please ensure that only identifiers from this corpus are selected, and each identifier appears exactly once in the ranking results."
    return Notice
def get_label_ranking(generated_label):
    # 生成标签到排名的映射（排名 = 索引 + 1）
    ranking_dict = {label: idx + 1 for idx, label in enumerate(generated_label)}
    
    # 按字母顺序排序标签
    sorted_labels = sorted(ranking_dict.keys())
    
    # 生成对应的排名列表
    sorted_ranking = [ranking_dict[label] for label in sorted_labels]
    
    # 返回排序后的标签字典、标签列表、排名列表
    return sorted_ranking


def extract_from_string(string):
    letters = [char for char in string if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
    return letters

def setup_logging(args, resume=False):
    """配置日志记录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{args.data_name}/{args.backbone}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    os.makedirs(f"{log_dir}/logs", exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    
    log_file = f"{log_dir}/logs/training.log"
    
    # 清除已有handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # 仅保存到文件
        ]
    )
    if not resume:
        logging.info(f"Training started with config:\n{str(vars(args))}")
    return log_dir


def save_checkpoint(args, actor, optimizer, batch_idx, epoch, log_dir, loss_stats):
    """保存模型检查点"""
    checkpoint_dir = f"{log_dir}/checkpoints"
    
    checkpoint_path = f"{checkpoint_dir}/actor_batch_{batch_idx}_epoch_{epoch}.pt"
    save_dict = {k: v for k, v in actor.state_dict().items() if 'llm' not in k}
    torch.save({
        'batch_idx': batch_idx,
        'epoch': epoch,
        'actor_state_dict':save_dict, #不保存llm的参数
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'loss_stats': loss_stats,
    }, checkpoint_path)
    
    logging.info(f"Saved checkpoint at batch {batch_idx}, epoch {epoch} to {checkpoint_path}")
    return checkpoint_path

def find_latest_checkpoint(args, timestamp_dir=None):
    """查找指定时间戳目录下的最新检查点文件（优先选择epoch最大的，其次batch最大的）"""
    if timestamp_dir is None:
        # 如果没有指定时间戳目录，则查找最新的时间戳目录
        base_dir = f"./logs/{args.data_name}/{args.backbone}"
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Log directory not found: {base_dir}")
            
        timestamp_dirs = [d for d in os.listdir(base_dir) 
                          if os.path.isdir(os.path.join(base_dir, d))]
        if not timestamp_dirs:
            raise FileNotFoundError(f"No timestamp directories found in {base_dir}")
        
        timestamp_dirs.sort(reverse=True)
        timestamp_dir = timestamp_dirs[0]
    
    checkpoint_dir = f"./logs/{args.data_name}/{args.backbone}/{timestamp_dir}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # 匹配文件名中的 epoch 和 batch，例如 actor_batch_401_epoch_1.pt
    def extract_epoch_batch(filename):
        match = re.search(r'batch_(\d+)_epoch_(\d+)', filename)
        if match:
            batch = int(match.group(1))
            epoch = int(match.group(2))
            return (epoch, batch)
        else:
            return (0, 0)  # 如果解析失败，排到最后

    # 按照 epoch 降序，再 batch 降序排序
    checkpoints.sort(key=lambda x: extract_epoch_batch(x), reverse=True)

    return os.path.join(checkpoint_dir, checkpoints[0])

def load_checkpoint(checkpoint_path, args, actor, optimizer):
    """从检查点恢复训练"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    actor.load_state_dict(checkpoint['actor_state_dict'],strict=False)


def rank_net(y_pred, y_true, weighted=False, use_rank=False, weight_by_diff=False,
                 weight_by_diff_powed=False):
        """
        RankNet loss introduced in "Learning to Rank using Gradient Descent".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
        :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
        :return: loss value, a torch.Tensor
        """
        if use_rank is None:
            y_true = torch.tensor([[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]] * y_pred.size(0)).cuda()

        # generate every pair of indices from the range of document length in the batch
        document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

        pairs_true = y_true[:, document_pairs_candidates]
        selected_pred = y_pred[:, document_pairs_candidates]

        # calculate the relative true relevance of every candidate pair
        true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
        pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

        # filter just the pairs that are 'positive' and did not involve a padded instance
        # we can do that since in the candidate pairs we had symetric pairs so we can stick with
        # positive ones for a simpler loss function formulation
        the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

        pred_diffs = pred_diffs[the_mask]

        weight = None
        if weighted:
            values, indices = torch.sort(y_true, descending=True)
            ranks = torch.zeros_like(indices)
            ranks.scatter_(1, indices, torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices))
            pairs_ranks = ranks[:, document_pairs_candidates] 
            rank_sum = pairs_ranks.sum(-1)
            weight = 1/rank_sum[the_mask]     #Relevance Feedback
            # rank_prod=pairs_ranks[:, :, 0]*pairs_ranks[:, :, 1]
            # weight = rank_sum[the_mask]/rank_prod[the_mask]      
        else:
            if weight_by_diff:
                abs_diff = torch.abs(true_diffs)
                weight = abs_diff[the_mask]
            elif weight_by_diff_powed:
                true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
                abs_diff = torch.abs(true_pow_diffs)
                weight = abs_diff[the_mask]

        # 'binarize' true relevancy diffs since for a pairwise loss we just need to know
        # whether one document is better than the other and not about the actual difference in
        # their relevancy levels
        true_diffs = (true_diffs > 0).type(torch.float32)
        true_diffs = true_diffs[the_mask]

        return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

def sample_top_p(probs, top_p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # 对概率进行降序排序
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # 计算累积概率
    sorted_indices_to_remove = cumulative_probs > top_p  # 根据top_p确定要移除的索引
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    probs[indices_to_remove] = 0  # 将被移除的概率设置为0
    if probs.sum() > 0:
        next_token = torch.multinomial(probs, 1)  # 从非零概率中抽样
    else:
        next_token = probs.argmax(dim=-1, keepdim=True)  # 如果所有概率都为0，则选择最大的
    return next_token



class FixedLengthBatchSampler(Sampler):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.batches = self._create_batches()

    def _create_batches(self):
        length_dict = defaultdict(list)
        for idx, item in enumerate(self.data):
            length = len(item['conversations'][-1]['value'])
            length_dict[length].append(idx)

        batches = []
        for indices in length_dict.values():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])

        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


class GenerationDataset(Dataset):
    def __init__(self, raw_data, model_tokenizer, combined=False) -> None:
        self.raw_data = raw_data
        self.tokenizer = model_tokenizer
        self.combined = combined
        self.system_message_supported = "system" in self.tokenizer.chat_template
    
    def __getitem__(self, index):
        conversation = self.raw_data[index]["conversations"]
        sys_msg = conversation[0]['value']
        input_context = conversation[1]['value'] + get_Notice(conversation[2]["value"])
        label = conversation[2]["value"]
        label += self.tokenizer.eos_token
        
        if self.system_message_supported:
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": input_context}
            ]
        else:
            messages = [
                {"role": "user", "content": sys_msg + "\n " + input_context}
            ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = fix_text(prompt)
        if self.combined:
            label_map = {}
            label_rank = 0
            for token in conversation[2]["value"]:
                if token.isalpha():
                    label_map[token] = label_rank
                    label_rank += 1
            
            rank_label = [label_map[chr(c)] for c in range(START_IDX, START_IDX+len(label_map))]
            return prompt, label, rank_label
        else:
            return prompt, label
    
    def __len__(self):
        return len(self.raw_data)
    




def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return input_ids, labels, sources_tokenized["input_ids_lens"]

def combined_collate_fn(data, tokenizer):
    prompts, labels, rank_labels = list(zip(*data))
    tokenized_inputs, labels, source_lens = preprocess(prompts, labels, tokenizer)
    tokenized_inputs = torch.nn.utils.rnn.pad_sequence(
        tokenized_inputs, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    return tokenized_inputs, labels, rank_labels, source_lens


# 自定义 collate_fn 以进行左侧 padding
def collate_fn_test(batch):
    prompts, labels, rank_labels = list(zip(*data))
    
def initialize_models(args):
    """Initialize base model and tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    if args.backbone == "Llama-3.2-3B-Instruct":
        llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='cuda',

        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )
    return llm, tokenizer

def load_datasets(dataset_path):
    """Load and split datasets"""
    ds = load_from_disk(dataset_path)
    def func_1(example): # 去除有重复元素
        extracted = extract_from_string(example['conversations'][2]['value'])
        return len(extracted) >= 0 #保留ranking大于10的部分    
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