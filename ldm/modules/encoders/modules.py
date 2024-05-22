import torch
import torch.nn as nn
from transformers import  CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

import numpy as np
from PIL import Image

def _expand_mask(mask, dtype, tgt_len = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _build_causal_attention_mask(bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77,
                 freeze=True, layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)

        self.processor = CLIPProcessor.from_pretrained(version)
        self.image_encoder = CLIPModel.from_pretrained(version)

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

        def embedding_forward(
                self,
                input_ids=None,
                position_ids=None,
                inputs_embeds=None,
                embedding_manager=None,
                image_embeds=None,
        ) -> torch.Tensor:

            seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

            if position_ids is None:
                position_ids = self.position_ids[:, :seq_length]

            if inputs_embeds is None:
                inputs_embeds = self.token_embedding(input_ids)

            if embedding_manager is not None:
                inputs_embeds = embedding_manager(input_ids, inputs_embeds, image_embeds=image_embeds)

            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings

            return embeddings

        self.transformer.text_model.embeddings.forward = embedding_forward.__get__(
            self.transformer.text_model.embeddings)

        def encoder_forward(
                self,
                inputs_embeds,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            hidden_states = inputs_embeds
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            return hidden_states

        self.transformer.text_model.encoder.forward = encoder_forward.__get__(self.transformer.text_model.encoder)

        def text_encoder_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                embedding_manager=None,
                image_embeds=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify either input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                            embedding_manager=embedding_manager, image_embeds=image_embeds)

            bsz, seq_len = input_shape
            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
                hidden_states.device
            )

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            last_hidden_state = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = self.final_layer_norm(last_hidden_state)

            return last_hidden_state

        self.transformer.text_model.forward = text_encoder_forward.__get__(self.transformer.text_model)

        def transformer_forward(
                self,
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                embedding_manager=None,
                image_embeds=None,
        ):
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                embedding_manager=embedding_manager,
                image_embeds=image_embeds,
            )

        self.transformer.forward = transformer_forward.__get__(self.transformer)

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, input_img, **kwargs):
        if input_img is None:
            print('input_img is None')
            input_img = torch.rand(size=(1, 3, 512, 512)).to(self.device)

        img = input_img.permute(0, 2, 3, 1)
        img = img.cpu().numpy().astype(np.uint8)
        image = [Image.fromarray(i) for i in img]
        img = self.processor(text=["a" for _ in range(len(image))], images=image, return_tensors="pt", padding=True)
        image_embeds = self.image_encoder(**img.to(self.device)).image_embeds

        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        z = self.transformer(input_ids=tokens, image_embeds=image_embeds, **kwargs)

        return z

    def encode(self, text, input_img, **kwargs):
        return self(text, input_img=input_img, **kwargs)