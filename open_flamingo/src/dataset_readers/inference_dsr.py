import numpy as np
from open_flamingo.src.dataset_readers.base_scorer_dsr import BaseScoreDatasetReader
from open_flamingo import create_model_and_transforms
import logging
import requests
from io import BytesIO
from PIL import Image
from typing import List
import os
from transformers import CLIPImageProcessor, CLIPTokenizer

import torch


logger = logging.getLogger(__name__)


class InferenceDatasetReader(BaseScoreDatasetReader):

    def __init__(self, task_name, index_reader, n_ics, field, dataset_path=None, vision_encoder_path='ViT-L-14',
                 vision_encoder_pretrained='openai',
                 lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
                 tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b", dataset_split=None,
                 ds_size=None):

        _, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=vision_encoder_path,
            clip_vision_encoder_pretrained=vision_encoder_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=1
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.index_reader = index_reader
        self.n_ics_in_prompt = n_ics
        # set truncation to false so that metadata['len'] corresponds to metadata['text']
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size)

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_caption_A_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}"

    def get_vqa_prompt(self, text=None) -> str:
        return f"<image>{text if text is not None else ''}{'<|endofchunk|>' if text is not None else ''}"

    def get_ice_prompt(self, entry):
        
        if 'ctxs' in entry:
            ctx = [self.index_reader[i] for i in entry['ctxs'][:self.n_ics_in_prompt]]

            context_text = "".join(
                [
                    self.get_caption_prompt(caption=i['metadata']['text'].strip()) + "\n"
                    for i in ctx
                ]
            )

            context_images = [i['metadata']['image'] for i in ctx]
        else:
            context_text, context_images = [], []

        return context_text, context_images

    def shard(self, accelerator):
        self.dataset_wrapper.dataset = self.dataset_wrapper.dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )
        self.encoded_dataset = self.encoded_dataset.shard(
            num_shards=accelerator.num_processes,
            index=accelerator.process_index
        )
        assert len(self.dataset_wrapper.dataset) == len(self.encoded_dataset)


    def _prepare_images(self, item: List[Image.Image], task_name) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = len(item)
        batch_images = None
        for iimage, image_name in enumerate(item):
            # coco
            if task_name == 'coco':
                image_path = os.path.join(os.path.join('coco/train2014', image_name))
                image = Image.open(image_path)

            assert image is not None

            preprocessed = self.image_processor(image)
            if batch_images is None:
                batch_images = torch.zeros(
                    (images_per_example, 1) + preprocessed.shape,
                    dtype=preprocessed.dtype,
                )
            batch_images[iimage, 0] = preprocessed
        if batch_images is not None:
            batch_images = batch_images.to(
                self.device, 
                # dtype=self.cast_dtype, 
                non_blocking=True
            )
        return batch_images

    def _prepare_text(
            self,
            item: List[str],
            padding="max_length",
            truncation=True,
            max_length=77,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (T_txt)
            attention_mask (tensor)
                shape (T_txt)
        """
        encodings = self.tokenizer(
            item,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"].squeeze(0), encodings["attention_mask"].squeeze(0)
        input_ids = input_ids.to(self.device, 
                                 # dtype=self.cast_dtype, 
                                 non_blocking=True)
        attention_mask = attention_mask.to(
            self.device, 
            # dtype=self.cast_dtype, 
            non_blocking=True
        )
        return input_ids, attention_mask.bool()

    
