# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re

dataset2task = {
    'arxiv': ['lp', 'nc'],
    'chemblpre': ['gc'],
    'chemhiv': ['gc'],
    'chempcba': ['gc'],
    'children': ['lp', 'nc'],
    'citeseer': ['nc'],
    'computer': ['lp', 'nc'],
    'cora': ['lp', 'nc'],
    'cora_simplified': ['lp', 'nc'],
    'fb15k_237': ['lc'],
    'history': ['lp', 'nc'],
    'instagram': ['nc'],
    'photo': ['lp', 'nc'],
    'products': ['lp', 'nc'],
    'pubmed': ['lp', 'nc'],
    'reddit': ['nc'],
    'sports': ['lp', 'nc'],
    'wikics': ['nc'],
    'wn18rr': ['lc'],
    'cora_link': ['lp'],
    'cora_node': ['nc'],
    'expla_graph': ['gc'],
    'protein_hs': ['lp'],
    'scene_graph': ['gc'],
    'fb15k237': ['lc'],

}

train_dataset_names = ['arxiv', 'citeseer', 'pubmed',
                       'instagram',
                       'children', 'computer', 'photo', 'sports',
                       'chemblpre', 'chempcba',
                       'wn18rr']

test_dataset_names = ['cora', 'cora_simplified',
                      'reddit',
                      'history', 'products',
                      'chemhiv',
                      'fb15k_237',
                      'wikics']


def compute_score(split):
    if split == 'train':
        return compute_score_norm
    elif split == 'validation':
        return compute_score_val


def compute_score_val(solution_str, ground_truth):
    try:
        response_part = solution_str.split('<｜Assistant｜>')[-1]
        reasoning_part, answer_part = response_part.split('</think>', 1)
        final_answer = answer_part.split('Answer:', 1)[-1].split('Brief_reasoning:', 1)[0].strip()
        ground_truth = str(ground_truth).lower().strip()
        if str(final_answer).lower() == ground_truth:
            return 1.
        else:
            return 0.
    except:
        return 0.


def compute_score_norm(solution_str, ground_truth):
    try:
        response_part = solution_str.split('<｜Assistant｜>')[-1]
        reasoning_part, answer_part = response_part.split('</think>', 1)
        reasoning = reasoning_part.strip()
        final_answer_part, brief_reasoning_part = answer_part.split('Answer:', 1)[-1].split('Brief_reasoning:', 1)
        final_answer = final_answer_part.strip()
        brief_reasoning = brief_reasoning_part.strip()
        ground_truth = str(ground_truth).lower().strip()

        if str(final_answer).lower() == ground_truth:
            return 1.
        elif ground_truth in str(brief_reasoning).lower():
            return 0.2
        else:
            return 0.01
    except:
        return 0


def compute_score_rethink(solution_str, ground_truth):
    try:
        response_part = solution_str.split('<｜Assistant｜>')[-1]
        reasoning_part, answer_part = response_part.split('</think>', 1)
        reasoning = reasoning_part.strip()
        if not all(i in reasoning for i in
                   ['<structure>', '</structure>', '<semantic>', '</semantic>', '<comprehensive>',
                    '</comprehensive>', '<rethink>', '</rethink>']):
            return 0
        candidate_answer = reasoning.split('<comprehensive>', 1)[-1].split('</comprehensive>')[0].strip()
        final_answer_part, brief_reasoning_part = answer_part.split('Answer:', 1)[-1].split('Brief_reasoning:', 1)
        final_answer = final_answer_part.strip()
        brief_reasoning = brief_reasoning_part.strip()
        ground_truth = str(ground_truth).lower().strip()

        if str(final_answer).lower() == ground_truth:
            return 1.
        elif ground_truth in str(brief_reasoning).lower():
            return 0.5
        elif ground_truth in str(candidate_answer).lower():
            return 0.3
        elif ground_truth in str(reasoning).lower():
            return 0.1
        else:
            return 0.05
    except:
        return 0


def _select_rm_score_fn(data_source, split):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif data_source in dataset2task.keys():
        return compute_score(split)
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, split) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.split = split

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        count = 0

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, self.split)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            # if data_source not in already_print_data_sources:
            #     already_print_data_sources[data_source] = 0

            # if already_print_data_sources[data_source] < self.num_examine:
            #     already_print_data_sources[data_source] += 1
            #     print(sequences_str)

            if count < 1:
                count += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, split='train')

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, split='validation')

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
