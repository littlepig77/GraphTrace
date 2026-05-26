import re
import os
from datasets import Dataset, load_dataset
from typing import List, Tuple
from tqdm import tqdm
import sys

sys.path.append('../../')
from verl.utils.hdfs_io import copy, makedirs
import argparse
import textwrap
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from collections import defaultdict

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
    'wn18rr': ['lc']
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


def make_template(input_text, tokenizer):
    messages = [
        {"role": "user", "content": input_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        return_tensors=None
    )

    return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/wuyicong/wyc/graph_reasoning/GRPO-zero/TinyZero/graph/data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=8422)
    parser.add_argument('--test_size', type=int, default=0)
    parser.add_argument('--template_type', type=str, default='base')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-m', '--mode', type=str, default='all', choices=['train', 'test'])

    args = parser.parse_args()

    # data_source = 'graph'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    seed = args.seed
    mode = args.mode

    data_path = f'/home/wuyicong/wyc/graph_reasoning/data/train_data/grpo_train_data_8422_rethink_filter2000_new.csv'
    raw_dataset = load_dataset('csv', data_files=data_path)['train'].shuffle(seed=seed)

    assert len(raw_dataset) >= TRAIN_SIZE + TEST_SIZE, 'data nums lack'
    train_dataset = raw_dataset.select(range(TRAIN_SIZE)) if mode != 'test' else None
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE)) if mode != 'train' else None

    model_path = '/home/wuyicong/wyc/graph_reasoning/model/DeepSeek_R1_Distill_Qwen_14B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # df = raw_dataset.to_pandas()
    # sample_dict = defaultdict(dict)
    # for (dataset_name, task_name), group in df.groupby(['dataset', 'task']):
    #     sample_answer = group['ground_truth'].sample(n=1, random_state=seed).iloc[0]
    #     sample_dict[dataset_name][task_name] = sample_answer
    # sample_dict = dict(sample_dict)

    def make_map_fn(split, tokenizer):
        def process_fn(row, idx):
            prompt = row['prompt']
            prompt_with_chat_template = make_template(prompt, tokenizer)
            data = {
                "data_source": row['dataset'],
                "prompt": [{
                    "role": "user",
                    "content": prompt_with_chat_template,
                }],
                "task": row['task'],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": row['ground_truth']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train', tokenizer), with_indices=True) if train_dataset else None
    test_dataset = test_dataset.map(function=make_map_fn('test', tokenizer), with_indices=True) if test_dataset else None

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'train_{len(train_dataset)}_rethink.parquet')) if train_dataset else None
    test_dataset.to_parquet(os.path.join(local_dir, f'test_{len(test_dataset)}_rethink.parquet')) if test_dataset else None

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
