from src.util.misc import App
from .base_dsw import *
import logging

logger = logging.getLogger(__name__)
field_getter = App()


@field_getter.add("q")
def get_q(entry):
    return entry["filename"], None


@field_getter.add("qa")
def get_qa(entry):
    image = entry["filename"]
    text = entry['sentences'][0]
    return image, text


@field_getter.add("a")
def get_a(entry):
    caption = entry['sentences'][0]
    return None, caption


@field_getter.add("gen_a")
def get_gen_a(entry):
    return entry["filename"], None


class DatasetWrapper(ABC):
    task_name = "coco"
    ice_separator = "\n"
    question_field = ["image"]
    answer_field = "caption"
    hf_dataset = "yerevann/coco-karpathy"
    hf_dataset_name = None
    field_getter = field_getter
