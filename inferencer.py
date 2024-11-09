import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
from collections import defaultdict
from src.metrics.coco.evaluator import compute_cider, postprocess_captioning_generation
from src.util.collators import DataCollatorWithPaddingAndCuda
from open_flamingo import create_model_and_transforms
from src.util.misc import parallel_run, save_json

from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")

logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, cfg, accelerator=None) -> None:
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        self.gen_field = cfg.dataset_reader.field

        self.accelerator = accelerator
        self.output_file = cfg.output_file
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        self.dataset_reader.shard(self.accelerator)

        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model, _, _ = create_model_and_transforms(
            clip_vision_encoder_path=cfg.vision_encoder_path,
            clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
            lang_encoder_path=cfg.lang_encoder_path,
            tokenizer_path=cfg.tokenizer_path,
            cross_attn_every_n_layers=1
        )
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        model.eval()
        model = self.accelerator.prepare(model)

        if hasattr(model, "module"):
            model = model.module

        return model, dataloader

