# Model Train Sample

Small study project for testing LoRA fine-tuning of a language model with Unsloth.

## Goal

This project was made only for learning and experimentation.
The main idea is to:

- load a base Qwen model
- fine-tune it with LoRA
- use a small subset of the Alpaca dataset
- save the adapter
- test the fine-tuned model in a simple chat script

## Technologies Used

- Python
- PyTorch
- Transformers
- TRL
- Unsloth
- Hugging Face Datasets
- Qwen3-4B
- LoRA / PEFT
- 4-bit loading
- 8-bit optimizer

## Project Files

- [train.py](c:/Projects/model-train-sample/train.py): fine-tuning script
- [chat.py](c:/Projects/model-train-sample/chat.py): simple chat script for inference
- [gpu_check.py](c:/Projects/model-train-sample/gpu_check.py): GPU and CUDA check
- [lora_trained](c:/Projects/model-train-sample/lora_trained): saved fine-tuned adapter
- [qwen3-lora](c:/Projects/model-train-sample/qwen3-lora): training output folder

## Training Setup

The training script uses:

- base model: `Qwen/Qwen3-4B`
- dataset: `tatsu-lab/alpaca`
- subset: first 500 samples
- max sequence length: `512` 
- LoRA rank: `16`
- precision: `bf16`
- optimizer: `adamw_8bit`

## What This Project Does

1. Loads the base Qwen model with Unsloth
2. Applies LoRA adapters 
3. Formats Alpaca samples into chat-style messages
4. Runs supervised fine-tuning with `SFTTrainer`
5. Saves the trained adapter and tokenizer
6. Loads the result in a local chat script

## Run
To run it you only need to run train.py then, once finished, chat.py
