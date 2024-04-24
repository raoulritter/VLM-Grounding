import glob
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from language_torch import LanguageModel, TextConfig  # Ensure these are adapted for PyTorch
from vision_torch import VisionConfig, VisionModel    # Ensure these are adapted for PyTorch

@dataclass
class LlaVAConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    ignore_index: int = -100
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000

    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlaVAConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

class LlavaModel(nn.Module):
    def __init__(self, config: LlaVAConfig):
        super().__init__()
        self.config = config
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.vision_feature_layer = config.vision_feature_layer
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

    def get_input_embeddings(self, input_ids: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None):
        if pixel_values is None:
            return self.language_model(input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)
        _, _, hidden_states = self.vision_tower(pixel_values.permute(0, 3, 1, 2), output_hidden_states=True)
        selected_image_feature = hidden_states[self.vision_feature_layer]

        if self.vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        image_features = self.multi_modal_projector(selected_image_feature)
        final_inputs_embeds = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids)
        return final_inputs_embeds

    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        image_token_index = self.config.image_token_index
        image_positions = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0].tolist()

        text_segments = []
        start_idx = 0
        for position in image_positions:
            text_segments.append(inputs_embeds[:, start_idx:position])
            start_idx = position + 1

        image_embeddings = torch.split(image_features, 1, dim=0)
        final_embeddings = [v.squeeze(0) for pair in zip(text_segments, image_embeddings) for v in pair]
        final_embeddings += [inputs_embeds[:, start_idx:]]

        return torch.cat(final_embeddings, dim=1)

    def forward(self, input_ids, pixel_values):
        inputs_embeds = self.get_input_embeddings(input_ids, pixel_values)
        logits = self.language_model(input_ids, inputs_embeds=inputs_embeds)
        return logits

    @staticmethod
    def from_pretrained(path_or_hf_repo: str):
        path = Path(path_or_hf_repo)
        if not path.exists():
            path = Path(snapshot_download(repo_id=path_or_hf_repo, allow_patterns=["*.json", "*.bin", "*.py", "tokenizer.model", "*.token"]))
        with open(path / "config.json", "r") as f:
            model_config = json.load(f)
        model_config = LlaVAConfig.from_dict(model_config)
        model_config.vision_config = VisionConfig.from_dict(model_config.vision_config)
        model_config.text_config = TextConfig.from_dict(model_config.text_config)
        model = LlavaModel(model_config)
        model.load_state_dict(torch.load(next(glob.iglob(str(path / "*.bin")))))
        return model
