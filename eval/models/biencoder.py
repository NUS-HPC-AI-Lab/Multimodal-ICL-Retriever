
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch import Tensor as T
import open_clip

from transformers import CLIPProcessor, CLIPModel, CLIPConfig, PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class BiEncoderConfig(PretrainedConfig):
    model_type = "BiEncoder"

    def __init__(
            self,
            q_model_name=None,
            ctx_model_name=None,
            ctx_no_grad=True,
            pair_wise=False,
            norm_embed=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_model_name = q_model_name
        self.ctx_model_name = ctx_model_name
        self.ctx_no_grad = ctx_no_grad
        self.pair_wise = pair_wise
        self.norm_embed = norm_embed


class BiEncoder(PreTrainedModel):
    config_class = BiEncoderConfig

    def __init__(self, config):
        super(BiEncoder, self).__init__(config)
        assert config.q_model_name is not None or config.ctx_model_name is not None

        if config.q_model_name is not None:
            self.question_model = CLIPModel.from_pretrained(config.q_model_name)
        else:
            self.question_model = None

        if config.ctx_model_name is not None:
            self.ctx_model = CLIPModel.from_pretrained(config.ctx_model_name)
        else:
            self.ctx_model = None

        # share q and ctx model if one of them is None
        if self.question_model is None and self.ctx_model is not None:
            self.question_model = self.ctx_model
            logging.info("Sharing ctx_model with question_model")
        if self.question_model is not None and self.ctx_model is None:
            self.ctx_model = self.question_model
            logging.info("Sharing question_model with ctx_model")

        self.ctx_no_grad = config.ctx_no_grad
        self.norm_embed = config.norm_embed
        self.loss_func = self.calc_nll_loss
        self.pair_wise = config.pair_wise

    def encode(self, image, text, attention_mask, encode_ctx=False):

        def get_features(model, img, txt, att):
            img_feat = model.get_image_features(img) if img is not None else None
            txt_feat = model.get_text_features(txt, attention_mask=att) if txt is not None else None
            return img_feat, txt_feat

        def normalize(feat):
            return feat / feat.norm(dim=-1, keepdim=True) if self.norm_embed and feat is not None else feat

        if encode_ctx:
            if self.ctx_no_grad:
                with torch.no_grad():
                    image_features, text_features = get_features(self.ctx_model, image, text, attention_mask)
                    image_features, text_features = normalize(image_features), normalize(text_features)
                    if image_features is not None and text_features is not None:
                        enc_emb = (image_features + text_features) / 2
                    elif image_features is not None:
                        enc_emb = image_features
                    elif text_features is not None:
                        enc_emb = text_features
                    else:
                        enc_emb = None
            else:
                image_features, text_features = get_features(self.ctx_model, image, text, attention_mask)
                image_features, text_features = normalize(image_features), normalize(text_features)
                if image_features is not None and text_features is not None:
                    enc_emb = (image_features + text_features) / 2
                elif image_features is not None:
                    enc_emb = image_features
                elif text_features is not None:
                    enc_emb = text_features

        else:
            image_features, text_features = get_features(self.question_model, image, text, attention_mask)
            image_features, text_features = normalize(image_features), normalize(text_features)
            if image_features is not None and text_features is not None:
                enc_emb = (image_features + text_features) / 2
            elif image_features is not None:
                enc_emb = image_features
            elif text_features is not None:
                enc_emb = text_features
        return enc_emb

    def forward(
            self,
            questions_image_tensor: T,
            ctxs_image_tensor: T,
            ctxs_text_tensor: T,
            ctxs_attn_mask: T,
            ctx_indices: T,
            labels
    ) -> Dict:
        ctx_pooled_out = self.encode(ctxs_image_tensor, ctxs_text_tensor, ctxs_attn_mask, encode_ctx=True)
        q_pooled_out = self.encode(questions_image_tensor, None, None, encode_ctx=False)
        return self.loss_func(q_pooled_out, ctx_pooled_out, ctx_indices, labels)

    def calc_nll_loss(
            self,
            q_vectors: T,
            ctx_vectors: T,
            ctx_indices: T,
            labels: Optional[T],
    ) -> Dict:
        assert ctx_indices.shape[1] == 1, "In-context number != 1, set dpp_training to true!"
        scores = torch.matmul(q_vectors, ctx_vectors.T)

        if not self.pair_wise:
            # directly get pos_idx in ctx_vectors
            labels = ctx_indices.squeeze(1)[labels]
            softmax_scores = F.log_softmax(scores, dim=1)

            loss = F.nll_loss(
                softmax_scores,
                labels,
                reduction="mean",
            )
        else:
            batch_size = scores.shape[0]
            # batch, num_hard_pos_neg
            ctx_indices = ctx_indices.reshape(-1).reshape(batch_size, -1)
            hard_pos_neg_num = ctx_indices.shape[1]
            in_batch_neg_num = hard_pos_neg_num
            full_ctx_indices = []
            for i in range(batch_size):
                neg_ctx_indices = torch.cat([ctx_indices[:i], ctx_indices[i + 1:]], dim=0).reshape(-1)
                rand_indx = torch.randperm(len(neg_ctx_indices))
                neg_ctx_indices = neg_ctx_indices[rand_indx][:in_batch_neg_num]
                per_sample_ctx_indices = torch.cat([ctx_indices[i], neg_ctx_indices], dim=0)
                full_ctx_indices.append(per_sample_ctx_indices)

            full_ctx_indices = torch.stack(full_ctx_indices, dim=0)
            scores = scores.gather(-1, full_ctx_indices)

            loss = ranking_loss(scores[:, :hard_pos_neg_num], margin=self.margin)
        return {'loss': loss, 'logits': scores}