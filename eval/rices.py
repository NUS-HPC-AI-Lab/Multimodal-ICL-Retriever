import open_clip
import torch
from tqdm import tqdm
import torch
import hydra.utils as hu
import numpy as np
from eval.utils import custom_collate_fn
from eval.models.biencoder import BiEncoder
from transformers import CLIPImageProcessor, CLIPTokenizer
import random


class RICES:
    def __init__(
            self,
            dataset,
            device,
            batch_size,
            model_config,
            pretrained_model_path=None,
            model_name=None,
            cached_features=None,
    ):
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size

        # Load the model and processor
        # clip
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        model_config = hu.instantiate(model_config)
        if pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(pretrained_model_path, config=model_config).to(self.device)
        else:
            self.model = BiEncoder(model_config).to(self.device)

        if cached_features is None:
            self.features = self._precompute_features()
        else:
            self.features = cached_features

    def _precompute_features(self):
        features = []

        # Switch to evaluation mode
        self.model.eval()

        # Set up loader
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=int(self.batch_size),
            collate_fn=custom_collate_fn,
        )

        with torch.no_grad():
            for batch in tqdm(
                    loader,
                    desc="Precomputing features for MSIER",
            ):
                # coco
                images, texts = batch["image"], batch["caption"]

                # coco clip/align
                image_inputs = torch.stack(
                    [torch.from_numpy(np.array(self.image_processor(image).pixel_values)) for image in images]
                ).to(self.device).squeeze(1)

                text_inputs = torch.stack(
                    [torch.from_numpy(np.array(self.tokenizer(text, padding='max_length', max_length=77, truncation=True).input_ids)) for text in
                     texts]
                ).to(self.device).squeeze(1)
                attention_mask = torch.stack(
                    [torch.from_numpy(np.array(self.tokenizer(text, padding='max_length', max_length=77, truncation=True).attention_mask)) for text in
                     texts]
                ).to(self.device).squeeze(1)

                # coco
                image_text_features = self.model.encode(image_inputs, text_inputs, attention_mask, encode_ctx=True)

                features.append(image_text_features.detach())
        features = torch.cat(features)

        return features

    def find(self, batch, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """
        # Switch to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # coco
            inputs = torch.stack(
                [torch.from_numpy(np.array(self.image_processor(image).pixel_values)) for image in batch]).squeeze(
                1).to(
                self.device
            )
            # coco
            query_feature = self.model.encode(inputs, None, None, encode_ctx=False)
            query_feature = query_feature.detach().cpu()

            if query_feature.ndim == 1:
                query_feature = query_feature.unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            # Get the indices of the 'num_examples' most similar images
            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [[self.dataset[i] for i in reversed(row)] for row in indices]

    def mask_random_words(self, sentence, num_words_to_mask):
        words = sentence.split()
        num_words = len(words)

        # Ensure the number of words to mask is not greater than the sentence length
        num_words_to_mask = min(num_words_to_mask, num_words)

        # Randomly pick indices to mask
        indices_to_mask = random.sample(range(num_words), num_words_to_mask)

        # Replace selected words with a mask token
        for i in indices_to_mask:
            words[i] = '[MASK]'

        # Reconstruct the sentence
        masked_sentence = ' '.join(words)
        return masked_sentence
