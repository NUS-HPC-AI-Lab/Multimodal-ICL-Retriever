import torch
import logging
from open_flamingo.src.dataset_readers.dataset_wrappers import get_dataset_wrapper
import open_clip
from typing import Any, Dict, Iterable
import requests
from io import BytesIO
from PIL import Image
from transformers import CLIPImageProcessor, CLIPTokenizer

logger = logging.getLogger(__name__)

def _encode_field(example, idx, **kwargs):
    field_getter = kwargs['field_getter']
    file_name, text = field_getter(example)
    return {
        'metadata': {'id': idx,
                     'text': text, 'image': file_name}
    }


def encode_field(dataset_wrapper, field):
    remove_columns = [col for col in dataset_wrapper.dataset.column_names]
    encoded_dataset = dataset_wrapper.dataset.map(
        _encode_field,
        load_from_cache_file=False,
        with_indices=True,
        remove_columns=remove_columns,
        fn_kwargs={'field_getter': dataset_wrapper.field_getter.functions[field]}
    )
    return encoded_dataset


class BaseScoreDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, field, model_name, dataset_path=None, dataset_split=None,
                 ds_size=None) -> None:
        self.field = field
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size)

    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size,
                     truncation=True):
        self.dataset_wrapper = get_dataset_wrapper(task_name,
                                                   dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        self.encoded_dataset = encode_field(self.dataset_wrapper, field)

    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)
