from verl import DataProto
import torch
import json
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.math_verify_reward import reward_fn_math_verify, reward_fn_math_verify_no_think

def _select_rm_score_fn(data_source, reward_impl_version):
    if reward_impl_version == 1: # think
        return reward_fn_math_verify
    elif reward_impl_version == 2: # no_think
        return reward_fn_math_verify_no_think
    else:
        raise NotImplementedError

class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_impl_version = reward_impl_version

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
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
            # if not "no_think" in self.reward_impl_version:
            from deepscaler.globals import THOUGHT_DELIMITER_START
            # sequences_str = [THOUGHT_DELIMITER_START + seq.strip() for seq in sequences_str]
            # if self.reward_impl_version != 2: # if think
            #     sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
        results = list(process_item(args[i]) for i in range(len(args)))

        # # Process items in parallel using ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=48) as executor:
        #     args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
        #     results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)
    
data = load_json('/hpc2hdd/home/zyang398/yangzhch6/projs/reasoning_baselines/LUFFY/luffy/verl/checkpoints/reasoning_baselines/qwen25_math_7b_luffy_on_policy_3k_openr1_nothink/val_generations/500.json')

# print(data[0])


source2score = {}
for line in data:
    our_score = []
    for response in line["output"]:
        whole_response = line["input"] + response
        if len(whole_response.split("\nassistant\n")) > 2:
            print("###### Warning: more than one assistant response detected, using the last one")
        whole_response = "\nassistant\n".join(whole_response.split("\nassistant\n")[1:])
        
        score = reward_fn_math_verify_no_think(solution_str=whole_response, ground_truth=line["answer"])
        our_score.append(score)
    
    if line["data_source"] not in source2score:
        source2score[line["data_source"]] = []
    source2score[line["data_source"]] += our_score


for key in source2score:
    print(f"## {key}: {len(source2score[key])} scores")
    print(f"Average score for {key}: {sum(source2score[key]) / len(source2score[key])}")