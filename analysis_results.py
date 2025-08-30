import json
import argparse
import numpy as np
from scipy.special import comb

def unbiased_pass_at_k(correct_counts, num_samples=128, max_k=128):
    num_problems = len(correct_counts)
    pass_at_k = []
    for k in range(1, max_k + 1):
        estimates = []
        for c in correct_counts:
            if k <= num_samples - c:
                estimate = 1 - comb(num_samples - c, k) / comb(num_samples, k)
            else:
                estimate = 1.0
            estimates.append(estimate)
        pass_at_k.append(np.sum(estimates))
    return pass_at_k


def compute_pass_k_avg(data):
    pass_k_list = []
    correct_counts = []
    for line in data:
        correct_count = sum(line['score'])
        num_samples = len(line['score'])
        correct_counts.append(correct_count)
    
    return unbiased_pass_at_k(correct_counts, num_samples=128, max_k=128)

def load_json(file_path):
    """
    Load a JSON file and return its content.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def coverage(K, data):
    pass_k = 0
    for item in data:
        acc_item = sum(item['score'])
        acc_item = acc_item / len(item['score'])

        pass_k += 1 - (1 - acc_item) ** K

    return pass_k


def coverage_(data):
    pass_k = 0
    for item in data:
        if sum(item['score']) > 0:
            pass_k += 1

    return pass_k


def split_data(data):
    train_data = []
    tes_data = []
    for item in data:
        if item["data_source"] == "train":
            train_data.append(item)
        else:
            tes_data.append(item)
    return train_data, tes_data

def compute_acc_micro(data):
    acc_count = 0
    all_count = 0
    for line in data:
        acc_count += sum(line["score"])
        all_count += len(line["score"])
    return acc_count / all_count

def compute_acc_source(data):
    acc_count_source = {}
    all_count_source = {}
    for line in data:
        source = line["data_source"]
        if source not in acc_count_source:
            acc_count_source[source] = 0
            all_count_source[source] = 0

        line_acc_count = sum(line["score"])
        line_all_count = len(line["score"])
        acc_count_source[source] += line_acc_count
        all_count_source[source] += line_all_count
    return {source: acc_count_source[source] / all_count_source[source] for source in acc_count_source}


if __name__ == "__main__":
    # 如果愿意，可以通过命令行参数覆盖默认路径与仓库名
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="")
    args = parser.parse_args()

    print("## Load data:", args.data_path)
    initial_test = load_json(args.data_path)

    source_2_acc = compute_acc_source(initial_test)
    print(source_2_acc)

    macro_acc_list = []
    for source in source_2_acc:
        if source == "train":
            continue
        macro_acc_list.append(source_2_acc[source])

    
    print("## Micro Accuracy:", compute_acc_micro(initial_test))
    print("## Macro Accuracy:", sum(macro_acc_list) / len(macro_acc_list))


    # compute pass 16/32/48/64/80/96/112/128
    pass_k_test = compute_pass_k_avg(initial_test)

    pass_k_test = [item / len(initial_test) for item in pass_k_test]
    print("##  Pass16:", pass_k_test[15])
    print("##  Pass32:", pass_k_test[31])
    print("##  Pass48:", pass_k_test[47])
    print("##  Pass64:", pass_k_test[63])
    print("##  Pass80:", pass_k_test[79])
    print("##  Pass96:", pass_k_test[95])
    print("##  Pass112:", pass_k_test[111])
    print("##  Pass128:", pass_k_test[127])
