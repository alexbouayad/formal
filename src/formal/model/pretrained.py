import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Final, Self, cast

import safetensors.torch
import torch
from huggingface_hub import CommitInfo, HfApi, hf_hub_download  # type: ignore
from peft import AutoPeftModelForCausalLM, PeftModel
from torch import nn

from formal.config import FormalizationConfig


class PreTrainedFormalModel(nn.Module):
    text_model: Final[PeftModel]
    formalization_config: Final[FormalizationConfig]

    formal_mask_embedding: nn.Linear
    formal_input_embedding: nn.Embedding
    formal_output_embedding: nn.Linear

    @property
    def dtype(self) -> torch.dtype:
        return cast(torch.dtype, self.text_model.dtype)

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self.text_model.device)

    @property
    def text_input_embedding(self) -> nn.Embedding:
        return cast(nn.Embedding, self.text_model.get_input_embeddings())  # type: ignore

    @property
    def text_output_embedding(self) -> nn.Linear:
        return cast(nn.Linear, self.text_model.get_output_embeddings())  # type: ignore

    def __init__(self, text_model: PeftModel, formalization_config: FormalizationConfig) -> None:
        super().__init__()

        self.text_model = text_model
        self.formalization_config = formalization_config

    def push_to_hub(
        self: Self,
        repo_id: str,
        *,
        config_name: str = "default",
        revision: str = "main",
        private: bool = True,
    ) -> CommitInfo:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir, config_name=config_name)

            api = HfApi()
            api.create_repo(repo_id, private=private, exist_ok=True)

            commit_info = api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                commit_message=f"Upload adapter '{config_name}' for revision '{revision}'",
                revision=revision,
            )

        return commit_info

    def save_pretrained(self, save_directory: str, *, config_name: str = "default") -> None:
        formalization_config_path = Path(save_directory) / config_name / "formalization_config.json"
        formal_safetensors_path = Path(save_directory) / config_name / "formal_embeddings.safetensors"

        formalization_config_json = json.dumps(asdict(self.formalization_config), indent=2)

        formal_embeddings_state_dict = {
            "formal_mask_embedding.weight": self.formal_mask_embedding.weight.detach().contiguous(),
            "formal_input_embedding.weight": self.formal_input_embedding.weight.detach().contiguous(),
            "formal_output_embedding.weight": self.formal_output_embedding.weight.detach().contiguous(),
        }

        self.text_model.save_pretrained(save_directory, adapter_name=config_name)
        formalization_config_path.write_text(formalization_config_json)
        safetensors.torch.save_file(formal_embeddings_state_dict, formal_safetensors_path)  # type: ignore

    @classmethod
    def from_pretrained(cls, path: str, *, config_name: str = "default", revision: str = "main") -> Self:
        text_model = AutoPeftModelForCausalLM.from_pretrained(  # type: ignore
            pretrained_model_name_or_path=path,
            adapter_name=config_name,
            revision=revision,
        )

        text_model = cast(PeftModel, text_model)

        if path.startswith("hf://"):
            repo_id = str(path).removeprefix("hf://")

            formalization_config_path = hf_hub_download(repo_id, filename=f"{config_name}/formalization_config.json")
            formal_safetensors_path = hf_hub_download(repo_id, filename=f"{config_name}/formal_embeddings.safetensors")

        else:
            formalization_config_path = Path(path) / config_name / "formalization_config.json"
            formal_safetensors_path = Path(path) / config_name / "formal_embeddings.safetensors"

        with open(formalization_config_path, "r") as f:
            formalization_config_dict = json.load(f)

        formalization_config = FormalizationConfig(**formalization_config_dict)
        formal_embeddings_state_dict = safetensors.torch.load_file(formal_safetensors_path)  # type: ignore

        formal_model = cls(text_model, formalization_config)
        formal_model.load_state_dict(formal_embeddings_state_dict, strict=False)

        return formal_model
