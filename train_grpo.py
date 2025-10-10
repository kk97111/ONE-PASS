import os
import re
import gc
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datasets import load_from_disk
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from eval_func import evaluate
from grop_func import ranking_prob,our_rank_net_loss
from utils import GenerationDataset, extract_from_string,setup_logging,save_checkpoint,find_latest_checkpoint,load_checkpoint,get_label_ranking




def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--backbone", type=str, required=True,
                      help="Base model name")
    parser.add_argument("--data_name", type=str, default=True,
                      help="data name")
    parser.add_argument("--model_path", type=str, 
                      default="/data/zoo/%(backbone)s",
                      help="Base model path")
    parser.add_argument("--dataset_path", type=str,
                      default="/data/yingpeng/efficient_inference/dataset/%(data_name)s/%(backbone)s/",
                      help="Path to preprocessed dataset")
    parser.add_argument("--group_batch_size", type=int, default=16,
                      help="Batch size for group training")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                      help="Learning rate for optimizer")
    parser.add_argument("--kl_weight", type=float, default=0.1,
                      help="Weight for KL divergence loss")
    parser.add_argument("--d_model", type=int, default=25)
    parser.add_argument("--nhead", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=5,
                      help="Save checkpoint every N batches")
    parser.add_argument("--n_step", type=int, default=3)
    parser.add_argument("--reward", type=str, default='none')
    parser.add_argument("--resume", action="store_true",
                      help="Resume training from latest checkpoint (True/False)")
    parser.add_argument("--checkpoint", type=str, default='None',
                      help="Specific checkpoint path to resume from")


    return parser.parse_args()


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.args.d_model, nhead=self.args.nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1).bfloat16().cuda()
        self.linear = nn.Linear(self.args.d_model, self.args.d_model).bfloat16().cuda()

        self.index2char = {i: chr(i+ord('A')) for i in range(25)}
        self.char2index = {chr(i+ord('A')):i for i in range(25)}
    def set_eval(self):
        self.transformer.eval()
    def set_train(self):
        self.transformer.train()
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




    def sampled_action(self, generated_label, matrix_action, matrix_llm_prob, sample = True, mode='RL'):
        # 针对一个generated_label和matrix去生成一个行为
        correct_list = self.longest_greedy(generated_label, matrix_llm_prob)
        if len(correct_list) == len(generated_label):
            return correct_list, len(correct_list) # 直接正确了，不用再进行排序了
        # 剩下的doc和对应其慨率
        remain_docs = sorted(list(set(generated_label) - set(correct_list)))
        if mode == 'heuristic':
            matrix_action = matrix_action[len(correct_list)]
        greedy_probs = matrix_action[[self.char2index[char] for char in remain_docs]]
        greedy_probs = greedy_probs/sum(greedy_probs)
        if sample:
            sampled_list = np.random.choice(
                remain_docs,
                size=len(remain_docs),  # 抽样次数 = 列表长度
                replace=False,          # 无放回抽样（确保不重复）
                p=greedy_probs                # 按概率加权
            ).tolist()
        else:
            sampled_list = [doc for doc, _ in sorted(zip(remain_docs, greedy_probs), key=lambda x: x[1], reverse=True)]        # 排序


        return correct_list + sampled_list, len(correct_list)
    def heuristic_action(self, batch_generated_label, matrix):
         # for this action, it receives a batch of
        #batch_matrices -->[batch,25,25]
        n_doc = len(batch_generated_label[0])
        matrix_llm_prob_np = matrix.float().cpu().detach().numpy() # 返回最后一个LLM对应的logit-prob
        actor_prob_np = matrix_llm_prob_np#[none or 1, 25]
        action_list = []
        for index in range(len(batch_generated_label)):
            # 行为抽样
            actions, _ = self.sampled_action(batch_generated_label[index], actor_prob_np[index], matrix_llm_prob_np[index], sample=False, mode ='heuristic')
            action_list.append(actions)
        return action_list

    def Parallel_Decoding_action(self, batch_generated_label, matrix):
        n_doc = len(batch_generated_label[0])
        matrix_llm_prob_np = matrix.float().cpu().detach().numpy() # 返回最后一个LLM对应的logit-prob
        actor_prob_np = matrix_llm_prob_np#[none or 1, 25]
        action_list = []
        for index in range(len(batch_generated_label)):
            # 行为抽样
            generated_label, matrix_action, matrix_llm_prob = batch_generated_label[index], actor_prob_np[index], matrix_llm_prob_np[index]
            correct_list = self.longest_greedy(generated_label, matrix_llm_prob)
            if len(correct_list) == len(generated_label):
                actions = correct_list
            # 剩下的doc和对应其慨率
            else:
                remain_docs = sorted(list(set(generated_label) - set(correct_list)))
                remain_indices = [self.char2index[char] for char in remain_docs]
                sampled_list = []
                for i in range(len(correct_list),len(generated_label)):
                    greedy_probs = matrix_action[i][remain_indices] 
                    normalized_probs = greedy_probs / np.sum(greedy_probs)
                    sampled_index = np.argmax(normalized_probs)
                    sampled_doc = self.index2char[remain_indices[sampled_index]]
                    sampled_list.append(sampled_doc)
                    remain_indices.pop(sampled_index)
                actions = correct_list + sampled_list
            
            action_list.append(actions)
        return action_list

               
    def RL_action(self, batch_generated_label, batch_matrices, sample=True):
        # for this action, it receives a batch of
        #batch_matrices -->[batch,25,25]
        n_doc = len(batch_generated_label[0])
        matrix_llm_prob_np = batch_matrices[:, -n_doc:, :].float().cpu().detach().numpy() # 返回最后一个LLM对应的logit-prob
        # 把前面的输入进去，得到actor的行为prob
        hidden_state = self.transformer(batch_matrices) # [none or 1, seq_len, A~Z]
        # actor_logit = hidden_state[:, -1, :]#self.linear(hidden_state[:, -n_doc:, :])  #[none or 1, 25]
        actor_logit = torch.mean(hidden_state,dim=1)#self.linear(hidden_state[:, -n_doc:, :])  #[none or 1, 25]
        actor_prob = torch.softmax(actor_logit[:, :n_doc],dim=-1)
        actor_prob_np = actor_prob.float().cpu().detach().numpy()
        action_list = []
        len_greedy_list = []
        for index in range(len(batch_generated_label)):
            # 行为抽样
            actions, len_greedy = self.sampled_action(batch_generated_label[index], actor_prob_np[index], matrix_llm_prob_np[index], sample)
            action_list.append(actions)
            len_greedy_list.append(len_greedy)
        return action_list, len_greedy_list, actor_prob, actor_logit
        
def initialize_models(args):
    """Initialize base model and tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'right'
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
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

def train_loop(args, llm, tokenizer, train_data,valid_data,test_data):
    """Main training loop"""
    # 初始化Actor和优化器
    pt = '/data/yingpeng/efficient_inference/logs/%s/%s/pretrained_actor_step2_NOweight.pt'%(args.data_name,args.backbone)
    checkpoint = torch.load(pt)

    actor = Actor(args)
    # load pretrained Actor:
    actor.load_state_dict(checkpoint)    
    # load reference model
    actor_reference = Actor(args)
    actor_reference.load_state_dict(checkpoint)    
    actor_reference.set_eval()


    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    
    # 初始化训练状态
    start_epoch = 0
    start_batch = 0
    loss_stats = {'epoch_loss': 0.0, 'epoch_kl_loss': 0.0, 'epoch_grpo_loss': 0.0}
    log_dir = None
    
    # 恢复训练逻辑
    if args.resume:
        print(args.resume)
        if args.checkpoint:
            try:
                checkpoint_path = find_latest_checkpoint(args, timestamp_dir=args.checkpoint)
                # 加载checkpoint
                start_epoch, start_batch, loss_stats, log_dir = load_checkpoint(
                    checkpoint_path, args, actor, optimizer)
                logging.info(f"Successfully resumed training from checkpoint: {checkpoint_path}")
            except FileNotFoundError as e:
                logging.error(f"Checkpoint file not found: {str(e)}")
                raise e  # 找不到checkpoint，直接报错终止
        else:
            logging.error("Checkpoint not specified. You must specify --checkpoint to resume training.")
            raise ValueError("Checkpoint not specified for resuming training.")
    
    # 全新训练（包括恢复失败的情况）
    if not args.resume:
        log_dir = setup_logging(args, resume=False)
        
        # 保存初始参数
        initial_params_path = f"{log_dir}/initial_params.txt"
        with open(initial_params_path, 'w') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")
        logging.info(f"Saved initial parameters to {initial_params_path}")
        
        # 保存初始检查点
        save_checkpoint(args, actor, optimizer, 0, 0, log_dir, loss_stats)
    
    # 初始化数据集和数据加载器
    train_dataset = GenerationDataset(train_data, tokenizer, combined=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1,shuffle=True,drop_last=True)
    valid_dataset = GenerationDataset(valid_data, tokenizer, combined=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)
    test_dataset = GenerationDataset(test_data, tokenizer, combined=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    
    # 训练循环
    total_batches = len(train_dataloader)
    start_time = time.time()
    
    # 创建训练统计文件
    stats_file = f"{log_dir}/logs/training.log"
    with open(stats_file, 'a') as f:
        f.write("epoch,batch_idx,loss,kl_loss,grpo_loss,time\n")
    ranking_save = - np.inf
    train_tolerance = 0
    for epoch in range(start_epoch, 1):
        epoch_loss = loss_stats['epoch_loss']
        epoch_kl_loss = loss_stats['epoch_kl_loss']
        epoch_grpo_loss = loss_stats['epoch_grpo_loss']
        
        # 初始化进度条
        pbar = tqdm(total=total_batches, desc=f"Epoch {epoch+1}", 
                    postfix={"loss": 0.0, "kl": 0.0, "grpo": 0.0},
                    disable=False)
        
        # 跳过已经处理过的batch
        dataloader_iter = enumerate(train_dataloader)
        # 跳过已处理的 batch（仅对恢复训练的第一个 epoch）
        if epoch == start_epoch and start_batch > 0:
            logging.info(f"Skipping to batch {start_batch}")
            pbar.update(start_batch)
            for _ in range(start_batch):
                next(dataloader_iter)
        
        for batch_idx, batch in dataloader_iter:
            if batch_idx < start_batch and epoch == start_epoch:
                continue
                
            batch_action_sequences = [] 
            batch_action_prob = []
            # sample for each query   
            prompt = batch[0][0]
            GT_label = extract_from_string(batch[1][0]) # [:-10] remove eos 
            label_corpus = sorted(GT_label)
            
            # THE FIRST inference and matrix
            generated_label = greedy_ranking_init(llm, tokenizer, [prompt], label_corpus) #第一次过LLMs 这个没有梯度计算
            prob_mat_only = ranking_prob(llm, tokenizer, [prompt], generated_label) #第二次过LLMs 这个没有梯度计算
            prob_mat = prob_mat_only.repeat(args.group_batch_size, 1, 1)
            padded_prob_mat_ts = F.pad(prob_mat, (0, 25 - prob_mat.shape[-1])) #[batch,n_doc,25]
            # batch for actor ranking generation
            generated_label_list, _, actor_prob,_ = actor.RL_action(generated_label * args.group_batch_size, padded_prob_mat_ts) #第二次得到 ranking 结果 这个有梯度
            batch_action_sequences.append(generated_label_list)
            batch_action_prob.append(actor_prob) #[batch,n_doc]
            #for reference model
            prob_mat_ref = prob_mat_only
            generated_label_ref, _, actor_prob_ref4KL,_ = actor_reference.RL_action(generated_label, padded_prob_mat_ts,sample=False) #第二次得到 ranking 结果;#这个可以和上一个RL_action共享同一个padded_prob_mat_ts；
            
            for step in range(args.n_step - 2):
                prob_matrices_all = ranking_prob(llm, tokenizer, [prompt for i in range(args.group_batch_size+1)], batch_action_sequences[-1] + generated_label_ref) #[batch+1,n_doc,n_doc]
                prob_matrices = prob_matrices_all[:args.group_batch_size] #[batch+1,n_doc,n_doc]
                prob_mat = torch.concat([prob_mat, prob_matrices], dim=1) #[batch+1,n_doc *(step -1),n_doc]
                padded_prob_mat_ts = F.pad(prob_mat, (0, 25 - prob_mat.shape[-1])) #[batch+1,n_doc *(step -1),25]
                generated_label_list, _, actor_prob_add,_ = actor.RL_action(batch_action_sequences[-1], padded_prob_mat_ts)
                batch_action_sequences.append(generated_label_list)
                batch_action_prob.append(actor_prob_add) #[batch,n_doc]
                _,   _,   actor_prob_add_ref4KL,   _      = actor_reference.RL_action(batch_action_sequences[-1], padded_prob_mat_ts)
                actor_prob_ref4KL = torch.concat([actor_prob_ref4KL, actor_prob_add_ref4KL], dim=0) #[batch,cum_doc,doc]
                #for reference model
                prob_matrices_ref = prob_matrices_all[-1].unsqueeze(0)
                prob_mat_ref = torch.concat([prob_mat_ref, prob_matrices_ref], dim=1)
                padded_prob_mat_ts_ref = F.pad(prob_mat_ref, (0, 25 - prob_mat.shape[-1]))
                generated_label_ref, _, _,_ = actor_reference.RL_action(generated_label_ref, padded_prob_mat_ts_ref,sample=False)


            #actor_prob --> [batch,cum_doc,doc]  where  cum_doc = group_batch_size * (step-1)
            #batch_action_sequences --> [step-1,group_batch_size,n_doc]
            # batch_action_prob --> [step-1,group_batch_size,n_doc]

            # 计算损失
            advantages,rewards,reward_ref = grpo_advantage(generated_label_list,generated_label_ref,GT_label,args.reward)
            # rewards_list.append(mean_rewards)
            print_reward = np.mean(rewards>reward_ref)



            batch_action_prob = torch.concat(batch_action_prob,dim=0) #[batch * (step - 1), n_doc]
            # for ranking
            batch_action_ranking = []
            for actions_batch in batch_action_sequences:
                batch_action_ranking.append(torch.tensor([get_label_ranking(actions) for actions in actions_batch]))
            batch_action_ranking = torch.concat(batch_action_ranking,dim=0).cuda()



            advantages = torch.concat([advantages for i in range(args.n_step-1)],dim=0)
            grpo_loss = our_rank_net_loss(batch_action_prob,batch_action_ranking, advantages,weighted=False)
            

            KL_loss = get_grpo_kl(batch_action_prob, actor_prob_ref4KL.detach())
            loss = grpo_loss + args.kl_weight * KL_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 记录损失
            batch_loss = loss.item()
            batch_kl_loss = KL_loss.item()
            batch_grpo_loss = grpo_loss.item()
            
            epoch_loss += batch_loss
            epoch_kl_loss += batch_kl_loss
            epoch_grpo_loss += batch_grpo_loss
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "Rewards_percentage": f"{print_reward:1.4f}",
                "kl": f"{batch_kl_loss:.4f}",
                "grpo": f"{batch_grpo_loss:.4f}"
            })
            pbar.update(1)
            
            # 记录到统计文件
            with open(stats_file, 'a') as f:
                f.write(f"{epoch},{print_reward},{batch_idx},{batch_loss},{batch_kl_loss},{batch_grpo_loss},{time.time()-start_time}\n")
            
            # 定期保存模型和日志
            if (batch_idx ) % args.save_interval == 0 or (batch_idx) == total_batches:
                actor.set_eval()
                ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate(valid_dataloader, llm, tokenizer, actor, args.n_step)
                # ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate(test_dataloader, llm, tokenizer, actor, args.n_step, eval_mode = 'complete')    
                print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15])
                actor.set_train()
                loss_stats = {
                    'epoch_loss': epoch_loss,
                    'epoch_kl_loss': epoch_kl_loss,
                    'epoch_grpo_loss': epoch_grpo_loss
                }
                if ranking_save < ranking:
                    train_tolerance = 0
                    ranking_save = ranking
                    cur_ck = save_checkpoint(args, actor, optimizer, batch_idx + 1, epoch, log_dir, loss_stats)
                    # ck = torch.load(cur_ck)
                    # actor_reference.load_state_dict(ck['actor_state_dict'])    
                    # actor_reference.eval()

                else:
                    train_tolerance = train_tolerance + 1
                if train_tolerance > 4:
                    break

                
                # 记录训练进度
                avg_loss = epoch_loss / (batch_idx + 1 - (start_batch if epoch == start_epoch else 0))
                avg_kl = epoch_kl_loss / (batch_idx + 1 - (start_batch if epoch == start_epoch else 0))
                avg_grpo = epoch_grpo_loss / (batch_idx + 1 - (start_batch if epoch == start_epoch else 0))
                
                logging.info(
                    f"Epoch {epoch+1} Batch {batch_idx+1}/{total_batches} | "
                    f"Outperform Ref Ration {print_reward:1.4f}|"
                    f"Loss: {avg_loss:.4f} (KL: {avg_kl:.4f}, GRPO: {avg_grpo:.4f}) | "
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
    start_epoch, start_batch, loss_stats, log_dir = load_checkpoint(
        checkpoint_path, args, actor, optimizer)
    actor.set_eval()
    ranking, spearman, footrule, kemeny,jaccard_at_10,jaccard_at_15 = evaluate(test_dataloader, llm, tokenizer, actor, args.n_step, eval_mode = 'complete')    
    print([ranking, spearman, footrule,kemeny,jaccard_at_10,jaccard_at_15])
    with open(stats_file, 'a') as f:
            f.write(f"TEST evaluation,Ranking:{ranking},Spearman:{spearman},Footrule:{footrule},Kemeny:{kemeny},Jaccard_at_10:{jaccard_at_10},Jaccard_at_15:{jaccard_at_15}\n")



if __name__ == "__main__":
    args = parse_args()
    
    # 路径格式化
    args.model_path = args.model_path % {"backbone": args.backbone}
    args.dataset_path = args.dataset_path % {"backbone": args.backbone,"data_name":args.data_name}
    
    # 初始化模型和数据集
    llm, tokenizer = initialize_models(args)
    train_data, valid_data, test_data= load_datasets(args.dataset_path)
    
    # 开始训练
    train_loop(args, llm, tokenizer, train_data, valid_data,test_data)