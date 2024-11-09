import pandas as pd
from datasets import Dataset
from copy import deepcopy
from PIL import Image

from open_flamingo.src.dataset_readers.base_scorer_dsr import encode_field
from open_flamingo.src.dataset_readers.inference_dsr import InferenceDatasetReader
from open_flamingo.src.dataset_readers.dataset_wrappers import get_dataset_wrapper


class ScorerDatasetReader(InferenceDatasetReader):

    def init_dataset(self, task_name, field, dataset_path, dataset_split,
                 ds_size, truncation=False):
        def get_instance(idx, entry):
            # todo, note here we may overwrite original idx field (if exists)
            entry['idx'] = idx  # unique id of original instances, used for grouping instances
            ctxs_candidates = entry.pop("ctxs_candidates")
            for exp in ctxs_candidates:
                example = deepcopy(entry)
                example['ctxs'] = exp
                yield example
        
        def get_dataset(data):
            for idx, entry in enumerate(data):
                yield from get_instance(idx, entry)
        self.dataset_wrapper = get_dataset_wrapper(task_name,
                                                   dataset_path=dataset_path,
                                                   dataset_split=dataset_split,
                                                   ds_size=ds_size)
        self.task_name = task_name
        df = pd.DataFrame(list(get_dataset(self.dataset_wrapper.dataset)))
        self.dataset_wrapper.dataset = Dataset.from_pandas(df)
        self.encoded_dataset = encode_field(self.dataset_wrapper, field)

    def __getitem__(self, index):
        entry = self.dataset_wrapper[index]
        prompt = self.encoded_dataset[index]['metadata']['image']

        _, answer = self.dataset_wrapper.get_field(entry=entry, field="a")
        answer_with_prompt = self.get_caption_A_prompt(answer)
        tokenized_labels = self.tokenizer.encode_plus(answer, truncation=False, add_special_tokens=False,
                                                      return_tensors='pt')
        # coco
        context_text, context_images = self.get_ice_prompt(entry)

        context_images.append(prompt)
        context_text += answer_with_prompt

        batch_images = self._prepare_images(context_images, self.task_name)
        input_ids, attention_mask = self._prepare_text(context_text)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': tokenized_labels.attention_mask[0],
            'images': batch_images,
            'metadata': entry
        }