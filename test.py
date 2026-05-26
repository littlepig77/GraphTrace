# from verl import DataProto
# import torch
# # @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
# def compute_causal_mask(data: DataProto):
#     B = data.batch.batch_size[0]
#     responses = data.batch['responses']      # (B, 2048), right-pad
#     old_log_probs = data.batch['old_log_probs']  # (B, 2048)
#     attention_mask = data.batch['attention_mask']  # (B, 4096)
#     response_length = responses.size(1)  # 2048
#     response_mask = attention_mask[:, -response_length:]  # (B, 2048), 1=valid, 0=pad
#
#     device = torch.cuda.current_device()
#
#     extra_info_list = data.non_tensor_batch['extra_info']
#     all_cf_prompts = [pt for info in extra_info_list for pt in info['cf_prompts']]  # len = B * 5
#
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained("/home/zhudianqi/GTE-R1/DeepSeek_R1_Distill_Qwen_1.5B",
#                                               trust_remote_code=True)
#
#     # tokenize + left-pad to 2048
#     encoded = tokenizer(
#         all_cf_prompts,
#         add_special_tokens=False,
#         truncation=True,
#         max_length=2048,
#         padding='max_length',
#         return_tensors='pt'
#     ).input_ids  # (B*5, 2048), right-padded
#
#     pad_token_id = tokenizer.pad_token_id
#     batch_size, max_len = encoded.shape
#     actual_lengths = (encoded != pad_token_id).sum(dim=1)  # (B*5,)
#     positions = torch.arange(max_len, device=encoded.device).unsqueeze(0).repeat(batch_size, 1)
#     valid_positions = positions >= (max_len - actual_lengths).unsqueeze(1)
#     left_padded = torch.full_like(encoded, pad_token_id)
#     left_padded[valid_positions] = encoded[encoded != pad_token_id]
#
#     cf_prompts_tensor = left_padded.to(device)    # (B*5, 2048)
#
#     N_CF = 2
#     N_total = B * N_CF
#
#     # --- Repeat responses and masks for each CF ---
#     resp_repeated = responses.unsqueeze(1).expand(-1, N_CF, -1).reshape(N_total, -1).to(device)  # (B*5, 2048)
#     resp_mask_repeated = response_mask.unsqueeze(1).expand(-1, N_CF, -1).reshape(N_total, -1).to(device)  # (B*5, 2048)
#
#     # --- Concatenate prompt + response ---
#     cf_input_ids = torch.cat([cf_prompts_tensor, resp_repeated], dim=1)  # (B*5, 4096)
#     prompt_mask = (cf_prompts_tensor != pad_token_id).long()  # (B*5, 2048)
#     cf_attention_mask = torch.cat([prompt_mask, resp_mask_repeated], dim=1)  # (B*5, 4096)
#     arange = torch.arange(cf_attention_mask.size(1), device=cf_attention_mask.device)
#     cf_position_ids = (arange - cf_attention_mask.argmax(dim=1, keepdim=True) + 1).clamp(min=0)
#
#     # batch dict
#     select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
#     cf_data = data.select(batch_keys=select_keys).batch
#     cf_data['input_ids'] = cf_input_ids
#     cf_data['attention_mask'] = cf_attention_mask
#     cf_data['position_ids'] = cf_position_ids
#     cf_data['responses'] = resp_repeated
#     # cf_data.meta_info = {
#     #     'micro_batch_size': self.config.rollout.log_prob_micro_batch_size,
#     #     'max_token_len': self.config.rollout.log_prob_max_token_len_per_gpu,
#     #     'use_dynamic_bsz': self.config.rollout.log_prob_use_dynamic_bsz,
#     #     'temperature': self.config.rollout.temperature,
#     # }
#
#     # compute_log_prob
#     # with self.ulysses_sharding_manager:
#     #     cf_data = self.ulysses_sharding_manager.preprocess_data(cf_data)
#     #     cf_log_probs = self.actor.compute_log_prob(data=cf_data)
#     #     cf_data.batch['cf_log_probs'] = cf_log_probs
#     #     cf_data = self.ulysses_sharding_manager.postprocess_data(cf_data)
#
#     cf_log_probs = torch.randn_like(cf_data['responses'], dtype=torch.float32).flatten()
#     cf_data['cf_log_probs'] = cf_log_probs
#     # if self._is_offload_param:
#     #     offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
#     # clear kv cache
#     # torch.cuda.empty_cache()
#     # log_gpu_memory_usage('After recompute log prob', logger=logger)
#
#     cf_log_probs = cf_data['cf_log_probs'].view(B, N_CF, -1)  # (B, 5, 2048)
#     avg_logp = cf_log_probs.mean(dim=1)
#
#     ##  20%
#     abs_diff = (avg_logp - old_log_probs.to(device)).abs()  # (B, 2048)
#     abs_diff_masked = abs_diff.masked_fill(~response_mask.bool().to(device), -float('inf'))  # (B, 2048)
#     sorted_values, sorted_indices = torch.sort(abs_diff_masked, dim=1, descending=True)  # (B, 2048)
#     valid_counts = response_mask.sum(dim=1)  # (B,)
#     k_per_sample = torch.clamp((valid_counts.float() * 0.2).long(), min=1)  # (B,)
#     arange = torch.arange(response_length, device=abs_diff.device).unsqueeze(0)  # (1, 2048)
#     topk_positions = arange < k_per_sample.to(device).unsqueeze(1)  # (B, 2048)
#     causal_mask = torch.zeros_like(response_mask, dtype=torch.bool).to(device)  # (B, 2048)
#     causal_mask.scatter_(dim=1, index=sorted_indices.to(device), src=topk_positions)
#
#     return causal_mask
#
# # data.batch['causal_mask'] = causal_mask
#
# B = 1
#
# # Base prompt that can be split
# base_prompt = """\
# Intro.
# Node0: A
# Node1: B
# The connection relationship among the nodes mentioned: A-B.
# Consider: output the graph.
# """
#
# # CF prompts (2 of them)
# cf_prompts = [
#     "Node0: X\nNode1: Y\nThe connection relationship among the nodes mentioned: X-Y.",
#     "Node0: P\nNode1: Q\nThe connection relationship among the nodes mentioned: P-Q."
# ]
#
# # responses: last token is pad
# responses = torch.tensor([[10, 20, 30, 40, 0]])  # (1,5)
# old_log_probs = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -10.0]])  # (1,5)
# attention_mask = torch.tensor([[1,1,1,1,1,1,1,1,1,0]])  # total len=10, so prompt=5, response=5; last response token padded
#
#
# class FakeBatch:
#     def __init__(self, **kwargs):
#         # 保存所有字段为属性（可选）
#         for k, v in kwargs.items():
#             setattr(self, k, v)
#         # 同时保存一份用于 __getitem__
#         self._data = kwargs
#         # 必须提供 batch_size（list/tuple，len=1）
#         self.batch_size = [kwargs['attention_mask'].shape[0]]
#
#     def __setitem__(self, key, value):
#         self._data[key] = value
#         # 同步为属性（可选，方便 .key 访问）
#         setattr(self, key, value)
#
#     def __getitem__(self, key):
#         return self._data[key]
#
#     def __contains__(self, key):
#         return key in self._data
#
#     def keys(self):
#         return self._data.keys()
#
# class FakeData:
#     def __init__(self, batch_dict):
#         self.batch = FakeBatch(**batch_dict)
#         self.non_tensor_batch = {}
#
#     def select(self, batch_keys):
#         # 从当前 batch 中筛选出指定 keys
#         selected_batch_dict = {
#             k: self.batch[k] for k in batch_keys if k in self.batch
#         }
#         # 创建新的 FakeData 实例
#         new_data = FakeData(selected_batch_dict)
#         # 保留 non_tensor_batch（如果需要）
#         new_data.non_tensor_batch = self.non_tensor_batch
#         return new_data
#
# # 构造 fake data
# data = FakeData({
#     'responses': responses,
#     'old_log_probs': old_log_probs,
#     'attention_mask': attention_mask,
# })
#
# data.non_tensor_batch = {
#     'extra_info': [{'cf_prompts': cf_prompts}]
# }
#
# # ===== Run test =====
# # causal_mask = compute_causal_mask(data)
# # print("causal_mask:", causal_mask)
#
# def get_per_sample_entropy_top_mask(entropy, response_mask, top_ratio=0.2):
#     """
#     For each sample in the batch, select the top `top_ratio` high-entropy tokens
#     among its own response tokens — fully vectorized, no for-loop.
#
#     Args:
#         entropy: [B, S] tensor of token entropies.
#         response_mask: [B, S] tensor (1 = response token, 0 = non-response).
#         top_ratio: fraction of response tokens to keep per sample (e.g. 0.2 = top 20%).
#
#     Returns:
#         entropy_top_mask: [B, S] binary mask (dtype=torch.long), 1 = selected.
#     """
#     B, S = entropy.shape
#     device = entropy.device
#
#     # Ensure boolean mask
#     response_mask = response_mask.bool()  # [B, S]
#
#     # Number of response tokens per sample
#     response_lengths = response_mask.sum(dim=1)  # [B]
#
#     # Compute k_i = ceil(L_i * top_ratio), but at least 1 if L_i > 0
#     # Use (x + 1 - eps).floor() == ceil(x) trick, or directly use torch.ceil
#     k_per_sample = (response_lengths.float() * top_ratio).ceil().long()  # [B]
#     k_per_sample = torch.clamp(k_per_sample, min=0, max=S)  # safety
#
#     # Mask out non-response positions with -inf so they won't be selected
#     masked_entropy = entropy.masked_fill(~response_mask, -float('inf'))  # [B, S]
#
#     # Find the maximum k across the batch to do a single topk
#     max_k = k_per_sample.max().item()
#
#     # Get top `max_k` indices for every sample
#     _, topk_indices = torch.topk(masked_entropy, k=max_k, dim=1)  # [B, max_k]
#
#     # Create a mask over the topk results: which of the top max_k are actually within k_i?
#     arange = torch.arange(max_k, device=device).unsqueeze(0)  # [1, max_k]
#     valid_in_topk = arange < k_per_sample.unsqueeze(1)  # [B, max_k], bool
#
#     # Scatter valid selections back into full-sequence mask
#     output_mask = torch.zeros(B, S, dtype=torch.long, device=device)  # [B, S]
#     output_mask.scatter_(1, topk_indices, valid_in_topk.long())
#
#     return output_mask
#
# # entropy = torch.tensor([[0.1, 0.9, 0.8, 0.2],
# #                         [0.5, 0.6, 0.9, 0.4]], device='cuda')
# # response_mask = torch.tensor([[1, 1, 1, 0],
# #                               [1, 1, 0, 0]], device='cuda')
# #
# # mask = get_per_sample_entropy_top_mask(entropy, response_mask, top_ratio=0.25)
# # print(mask)
#
# import torch
# from verl import DataProto  # 假设 DataProto 在这个路径，按你项目调整
#
# # 模拟构造 counterfactual 数据（小规模）
# B, N_CF = 2, 3
# N_total = B * N_CF  # = 6
# S = 16
# resp_len = 8
# pad_token_id = 0
#
# # 创建模拟张量（确保 attention_mask 至少有一个 1，避免 seqlen=0）
# cf_input_ids = torch.randint(1, 20, (N_total, S))          # 非 pad token
# cf_attention_mask = torch.ones(N_total, S, dtype=torch.long)
# cf_position_ids = torch.arange(S).repeat(N_total, 1)
# resp_repeated = torch.randint(1, 20, (N_total, resp_len))
#
# # 构造 cf_dict
# cf_dict = {
#     "input_ids": cf_input_ids,
#     "attention_mask": cf_attention_mask,
#     "position_ids": cf_position_ids,
#     "responses": resp_repeated,
# }
#
# # ✅ 关键：用 DataProto.from_single_dict 创建
# cf_data = DataProto.from_single_dict(cf_dict)
#
# # 设置 meta_info（按你配置）
# cf_data.meta_info = {
#     'micro_batch_size': 4,
#     'max_token_len': 128,
#     'use_dynamic_bsz': False,
#     'temperature': 1.0,
# }
# print(cf_data)


import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# 配置
MODEL_PATH = "/home/zhudianqi/GTE-R1/DeepSeek_R1_Distill_Qwen_1.5B"
INPUT = "/home/zhudianqi/GTE-R1/GRPO-normal/graph/data/train_4592_cf.parquet"
MAX_LEN = 2000

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 读数据
df = pd.read_parquet(INPUT)
print(f"Original: {len(df)}")

# 提取 content 并计算长度
def get_len(x):
    try:
        text = x[0]["content"]  # 直接取第一个 message 的 content
        return len(tokenizer.encode(text, add_special_tokens=True))
    except Exception:
        return 10**6  # 出错就设为超长，会被过滤

df["len"] = df["prompt"].apply(get_len)
filtered = df[df["len"] <= MAX_LEN].drop(columns=["len"])

# 保存
out_path = f"/home/zhudianqi/GTE-R1/GRPO-normal/graph/data/train_{len(filtered)}_cf.parquet"
filtered.to_parquet(out_path, index=False)
print(f"Saved {len(filtered)} samples to {out_path}")