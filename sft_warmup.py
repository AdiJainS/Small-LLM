!pip install unsloth
!pip install --no-cache-dir -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft accelerate bitsandbytes

import json
import uuid
import torch
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

# 1. LOAD DATA

with open("sft_warmup_final.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Loaded {len(raw_data)} total samples.")

# 2. ACTIVE TOOLS

ACTIVE_TOOLS = {
    "get_reservation_details",
    "cancel_reservation",
    "get_user_details",
    "search_direct_flight",
    "book_reservation",
    "think"
}

# 3. FORMAT TO HUGGING FACE TOOL-CALLING STANDARD

processed_data = {
    "messages": [],
    "tools": []
}

for row in raw_data:

    original_tools = json.loads(row["tools"])

    pruned_tools = [
        t for t in original_tools
        if t["name"] in ACTIVE_TOOLS
    ]

    hf_tools = [
        {
            "type": "function",
            "function": t
        }
        for t in pruned_tools
    ]

    messages = [
        {
            "role": "system",
            "content": row["system"]
        }
    ]

    last_tc_id = None
    last_tc_name = None

    for turn in row["conversations"]:

        # -------------------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------------------
        if turn["from"] == "human":

            messages.append({
                "role": "user",
                "content": turn["value"]
            })

        # -------------------------------------------------------------
        # ASSISTANT MESSAGE
        # -------------------------------------------------------------
        elif turn["from"] == "gpt":

            if turn["value"].strip():

                messages.append({
                    "role": "assistant",
                    "content": turn["value"]
                })

        # -------------------------------------------------------------
        # TOOL CALL
        # -------------------------------------------------------------
        elif turn["from"] == "function_call":

            call_data = json.loads(turn["value"])

            last_tc_id = f"call_{uuid.uuid4().hex[:8]}"
            last_tc_name = call_data["name"]

            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": last_tc_id,
                        "type": "function",
                        "function": {
                            "name": last_tc_name,
                            "arguments": json.dumps(
                                call_data["arguments"]
                            )
                        }
                    }
                ]
            })

        # TOOL OBSERVATION
        elif turn["from"] == "observation":

            messages.append({
                "role": "tool",
                "tool_call_id": last_tc_id,
                "name": last_tc_name,
                "content": turn["value"]
            })

    processed_data["messages"].append(messages)
    processed_data["tools"].append(hf_tools)


dataset = Dataset.from_dict(processed_data)

print(f"Formatted dataset size: {len(dataset)}")

# 4. LOAD QWEN3-1.7B

max_seq_length = 3072

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-1.7B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen-2.5"
)

# 5. APPLY CHAT TEMPLATE WITH NATIVE TOOLS

def apply_template(examples):

    texts = []

    for msgs, tools in zip(
        examples["messages"],
        examples["tools"]
    ):

        text = tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False
        )

        texts.append(text)

    return {
        "text": texts
    }


dataset = dataset.map(
    apply_template,
    batched=True,
    remove_columns=["messages", "tools"]
)

print("Chat template applied successfully.")

# 6. LORA ADAPTERS

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
        "down_proj"
    ],

    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# 7. SFT TRAINER

trainer = SFTTrainer(

    model=model,

    tokenizer=tokenizer,

    train_dataset=dataset,

    dataset_text_field="text",

    max_seq_length=max_seq_length,

    dataset_num_proc=2,

    args=SFTConfig(

        # BATCHING
        per_device_train_batch_size=1,

        gradient_accumulation_steps=8,

        # TRAINING
        warmup_ratio=0.1,

        num_train_epochs=1.5,

        learning_rate=1.5e-5,

        fp16=not is_bfloat16_supported(),

        bf16=is_bfloat16_supported(),

        # OPTIMIZER
        optim="adamw_8bit",

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        # REPRODUCIBILITY
        seed=3407,

        # OUTPUT
        output_dir="qwen3_airline_sft",

        # NEFTUNE
        neftune_noise_alpha=5,

        # LOGGING
        logging_steps=1,
    ),
)

# 8. STRICT LOSS MASKING

trainer = train_on_responses_only(
    trainer,

    instruction_part="<|im_start|>user\n",

    response_part="<|im_start|>assistant\n",
)

# 9. TRAIN

print("\nStarting Qwen3-1.7B SFT Warmup...")

trainer.train()

# 10. SAVE MODEL

model.save_pretrained(
    "qwen3-1.7b-airline-sft-lora"
)

tokenizer.save_pretrained(
    "qwen3-1.7b-airline-sft-lora"
)

print(
    "\nTraining completed! "
    "LoRA adapters successfully saved to "
    "'qwen3-1.7b-airline-sft-lora'."
)
