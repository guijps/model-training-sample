from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

ds_full = load_dataset("tatsu-lab/alpaca")
ds = ds_full["train"].select(range(500))

max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-4B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

#load_dataset(
#    "json",
#    data_files="dataset.jsonl",
#    split="train")

def format_examples(example):

    user_prompt = example["instruction"]

    if example["input"]:
        user_prompt += "\n\n" + example["input"]

    messages = [
        {
            "role": "user",
            "content": user_prompt,
        },
        {
            "role": "assistant",
            "content": example["output"],
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = ds.map(format_examples)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        output_dir="qwen3-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        fp16=False,
        bf16=True,
    ),
)

trainer.train()

model.save_pretrained("lora_trained")
tokenizer.save_pretrained("lora_trained")