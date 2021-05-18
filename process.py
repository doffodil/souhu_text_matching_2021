import json
import os
from typing import List, Tuple
import jieba
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

jieba.initialize()


def get_filepath(path: str, exclude_path=None):
    file_list = []
    for root, dirs, files in os.walk(path):
        if root in exclude_path:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


def stopwords_loader(stopwords_dir: str) -> set:
    stopwords = set()
    for file in get_filepath(stopwords_dir, []):
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                stopwords.add(line.strip())
    return stopwords


def sample_exchange(sample):
    index, source_text, target_text, label_id = sample
    source = target_text
    target = source_text
    new_sample = (index, source, target, label_id)
    return new_sample


class Dataset():
    def __init__(self, _sample_list: List[Tuple[int, str, str, any]]):
        self.sample_list = _sample_list

    @property
    def size(self):
        return len(self.sample_list)

    @staticmethod
    def dataset_loader(file_name: str, kind=0):
        dataset = []
        with open(file_name, 'r', encoding='utf-8') as fr:
            lines = fr.readlines()
            for _item in lines:
                item = json.loads(_item)
                if "labelA" in item:
                    dataset.append((kind, item["source"], item["target"], int(item["labelA"])))
                elif "labelB" in item:
                    dataset.append((kind, item["source"], item["target"], int(item["labelB"])))
                else:
                    dataset.append((kind, item["source"], item["target"], item["id"]))
        return Dataset(dataset)

    def merge_dataset(self, _other: 'Dataset'):
        self.sample_list += _other.sample_list

    def clean_dataset(self, stopwords_dir='datasets/stopwords'):
        stopwords = stopwords_loader(stopwords_dir)
        for i, (index, source_text, target_text, label_id) in enumerate(tqdm(self.sample_list)):
            source = " ".join(jieba.cut(source_text, HMM=False))
            target = " ".join(jieba.cut(target_text, HMM=False))
            source_word_list = [str(word) for word in source.split(' ') if word not in stopwords]
            target_word_list = [str(word) for word in target.split(' ') if word not in stopwords]
            source = "".join(source_word_list)
            target = "".join(target_word_list)
            self.sample_list[i] = (index, source, target, label_id)
        return

    def enhance_dataset(self):
        dataset = []
        for i, (index, source_text, target_text, label_id) in enumerate(tqdm(self.sample_list)):
            source = target_text
            target = source_text
            dataset.append((int(index + self.size), source, target, label_id))
        self.sample_list += dataset

    def pool_enhance_dataset(self):
        with Pool(cpu_count()) as pool:
            datasets = pool.map(sample_exchange, self.sample_list)
        self.sample_list += datasets
