import json
import pickle
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from src.util.misc import parallel_run, partial
from src.util.collators import DataCollatorWithPaddingAndCuda
from src.dataset_readers.base_dsr import BaseDatasetReader
from src.dataset_readers.index_dsr import IndexDatasetReader
import open_clip

from src.models.biencoder import BiEncoder

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(self, cfg) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model_config = hu.instantiate(cfg.model_config)
        self.model = BiEncoder(model_config)

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice

        self.is_train = cfg.dataset_reader.dataset_split == "train"

        if os.path.exists(cfg.faiss_index):
            logger.info(f"Loading faiss index from {cfg.faiss_index}")
            self.index = faiss.read_index(cfg.faiss_index)
        else:
            self.index = self.create_index(cfg)

    def create_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        res_list = self.forward(dataloader, encode_ctx=True)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add_with_ids(embed_list, id_list)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index

    def forward(self, dataloader, encode_ctx=False):
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop('metadata')
                if encode_ctx:
                    image_inputs = entry['image_inputs'].squeeze(1)
                    text_inputs, attention_mask = entry['input_ids'], entry['attention_mask']
                    text_inputs = entry['input_ids'].squeeze(1)
                    # coco
                    res = self.model.encode(image_inputs, text_inputs, attention_mask, encode_ctx=encode_ctx)
                else:
                    image_inputs = entry['image_inputs'].squeeze(1)
                    # coco
                    res = self.model.encode(image_inputs, None, None, encode_ctx=encode_ctx)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def find(self):
        res_list = self.forward(self.dataloader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]
        func = partial(knn, num_candidates=self.num_candidates, num_ice=self.num_ice)
        data = parallel_run(func=func, args_list=res_list, initializer=set_global_object,
                            initargs=(self.index, self.is_train))
        with open(self.output_file, "w") as f:
            json.dump(data, f)


def set_global_object(index, is_train):
    global index_global, is_train_global
    index_global = index
    is_train_global = is_train


def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice) + 1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    return entry


@hydra.main(config_path="open_flamingo/config", config_name="dense_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    dense_retriever = DenseRetriever(cfg)
    dense_retriever.find()


if __name__ == "__main__":
    main()
