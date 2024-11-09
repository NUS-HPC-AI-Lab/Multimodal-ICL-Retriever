import logging
import pandas as pd
import os
from open_flamingo.src.dataset_readers.base_dsr import BaseDatasetReader, encode_field
from open_flamingo.src.dataset_readers.base_scorer_dsr import BaseScoreDatasetReader
from open_flamingo.src.util.misc import save_json

logger = logging.getLogger(__name__)


def deduplicate(dataset_wrapper, encoded_dataset):
    """deduplication """
    df = pd.DataFrame(encoded_dataset)
    df['uid'] = df['image_inputs'].astype(str)
    is_dup = df.duplicated(subset=['uid'], keep='first')
    keep_idx = is_dup[~is_dup].index.values

    dataset_wrapper.dataset = dataset_wrapper.dataset.select(keep_idx)
    encoded_dataset = encoded_dataset.select(keep_idx)

    encoded_dataset = encoded_dataset.map(reassign_idx, load_from_cache_file=False, with_indices=True)
    logger.info(f"Keeping {len(keep_idx)}/{len(df)} instances after deduplicating")
    return dataset_wrapper, encoded_dataset


def reassign_idx(example, index):
    example['metadata']['id'] = index
    return example


class IndexDatasetReader(BaseDatasetReader):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        dataset_path = kwargs['dataset_path']
        # if not create index file, we create it by deduplication q field
        if dataset_path is None or not os.path.exists(dataset_path):
            self.encoded_dataset = encode_field(self.tokenizer, self.image_processor, self.dataset_wrapper, kwargs['field'], self.task_name, truncation=True)
            if dataset_path is not None:
                save_json(dataset_path, list(self.dataset_wrapper))
                logger.info(f"index dataset has been saved to {dataset_path}")
