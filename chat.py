from unsloth import FastLanguageModel
import torch

max_seq_length = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_trained",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

history = []

def generate_reply(history, max_new_tokens=256):
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return reply

print("Chat ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})
    reply = generate_reply(history)

    print(f"Model: {reply}\n")
    history.append({"role": "assistant", "content": reply})
