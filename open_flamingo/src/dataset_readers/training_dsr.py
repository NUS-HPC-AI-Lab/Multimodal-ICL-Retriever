import torch
from transformers import AutoTokenizer
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
import collections
from copy import deepcopy
import requests
from io import BytesIO
import os
from urllib.request import urlopen
from PIL import Image
import open_clip
from transformers import CLIPImageProcessor, CLIPTokenizer
from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor

InputSample = collections.namedtuple(
    "InputSample",
    [
        "question_image_ids",
        "question_text_ids",
        "ctxs_candidates"
    ]
)


def encode_field(example, **kwargs):
    field_getter = kwargs['field_getter']
    image_processor = kwargs['image_processor']
    image_name, text = field_getter(example)
    task_name = kwargs['task_name']

    # coco
    if task_name == 'coco':
        image_path = os.path.join(os.path.join('coco/train2014', image_name))
        image = Image.open(image_path)

    assert image is not None
    question_text_ids, question_image_ids = None, None

    if image_name:
        question_image_ids = image_processor(image)
    return {
        "question_image_ids": question_image_ids.pixel_values,
        "question_text_ids": question_text_ids,
        "ctxs_candidates": example['ctxs_candidates']
    }


class TrainingDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, field, dataset_path, model_name, dataset_split=None, ds_size=None):
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        dataset_wrapper = get_dataset_wrapper(task_name, dataset_path=dataset_path, dataset_split=dataset_split, ds_size=ds_size)
        self.encoded_dataset = self.encode_field(dataset_wrapper, field, task_name)

    def encode_field(self, dataset_wrapper, field, task_name):
        remove_columns = [col for col in dataset_wrapper.dataset.column_names]
        encoded_dataset = dataset_wrapper.dataset.map(
            encode_field,
            load_from_cache_file=False,
            remove_columns=remove_columns,
            fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field],
                       'tokenizer': self.tokenizer, 'image_processor': self.image_processor,
                       'task_name': task_name}
        )
        return encoded_dataset

    def __getitem__(self, index) -> InputSample:
        return InputSample(**self.encoded_dataset[index])

    def __len__(self):
        return len(self.encoded_dataset)

    def split_dataset(self, test_size=0.1, seed=42):
        dataset = self.encoded_dataset.train_test_split(test_size=test_size, seed=seed)
        train_dataset, eval_dataset = dataset['train'], dataset['test']

        cache_self = {k: self.__dict__[k] for k in self.__dict__.keys()}
        for k in self.__dict__.keys():
            self.__dict__[k] = None

        trainset_cls = deepcopy(self)
        for k, v in cache_self.items():
            trainset_cls.__dict__[k] = v
        trainset_cls.encoded_dataset = train_dataset

        evalset_cls = deepcopy(self)
        for k, v in cache_self.items():
            evalset_cls.__dict__[k] = v
        evalset_cls.encoded_dataset = eval_dataset

        self.__dict__ = cache_self
        return trainset_cls, evalset_cls