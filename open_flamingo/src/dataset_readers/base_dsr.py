import torch
import logging
from open_flamingo.src.dataset_readers.dataset_wrappers import get_dataset_wrapper
import open_clip
from typing import Any, Dict, Iterable
import requests
from io import BytesIO
import os
from urllib.request import urlopen
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

logger = logging.getLogger(__name__)


def _encode_field(example, idx, **kwargs):
    field_getter = kwargs['field_getter']
    image_processor = kwargs['image_processor']
    tokenizer = kwargs['tokenizer']
    truncation = kwargs['truncation']
    task_name = kwargs['task_name']

    tokenized_inputs, image_inputs, image, text = None, None, None, None
    if task_name == 'coco':
        image_name, text = field_getter(example)
        image = Image.open(os.path.join('coco/train2014', image_name))
        assert image is not None
        if text:
            tokenized_inputs = tokenizer(text, padding='max_length', max_length=77, truncation=truncation)
        image_inputs = image_processor(image)
    return {
        'input_ids': tokenized_inputs.input_ids if tokenized_inputs is not None else None,
        'attention_mask': tokenized_inputs.attention_mask if tokenized_inputs is not None else None,
        'image_inputs': image_inputs.pixel_values,
        'metadata': {'id': idx,
                     'text': text, 'image': image_name}
    }


def encode_field(tokenizer, image_processor, dataset_wrapper, field, task_name, truncation):
    remove_columns = [col for col in dataset_wrapper.dataset.column_names]
    encoded_dataset = dataset_wrapper.dataset.map(
        _encode_field,
        load_from_cache_file=False,
        with_indices=True,
        remove_columns=remove_columns,
        fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field],
                   'image_processor': image_processor, 'tokenizer': tokenizer, 'truncation': truncation,
                   'task_name': task_name}
    )
    return encoded_dataset


class BaseDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, field, model_name, dataset_path=None, dataset_split=None,
                 ds_size=None) -> None:
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        self.task_name = task_name
        self.dataset_split = dataset_split
        self.field = field
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size)

    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size,
                     truncation=True):
        self.dataset_wrapper = get_dataset_wrapper(task_name,
                                                   dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)

        self.encoded_dataset = encode_field(self.tokenizer, self.image_processor, self.dataset_wrapper, field,
                                            task_name, truncation)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)
