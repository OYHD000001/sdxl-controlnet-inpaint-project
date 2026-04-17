from __future__ import annotations

from huggingface_hub import hf_hub_download


REPO_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer/tokenizer_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/vocab.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]


def main() -> None:
    for filename in FILES:
        print(f"DOWNLOADING {filename}", flush=True)
        path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="model")
        print(f"OK {filename} -> {path}", flush=True)


if __name__ == "__main__":
    main()
