# =====================================================================
# 0. IMPORTS
# =====================================================================
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import re
import json
from collections import Counter
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# =====================================================================
# 1. CONFIG
# =====================================================================
SFT_ADAPTER_PATH = "qwen3-1.7b-airline-sft-lora"
SFT_DATA_PATH = "sft_warmup_final.json"

MAX_SEQ_LENGTH = 2048
MAX_PROMPT_TOKENS = 1536
MAX_COMPLETION_LENGTH = 300
LORA_RANK = 16

TARGET_TOOLS = {
    "get_user_details",
    "get_reservation_details",
    "search_direct_flight",
    "book_reservation",
    "cancel_reservation",
}

# =====================================================================
# 2. LOAD MODEL (Already Adapted)
# =====================================================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_ADAPTER_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=False, # Changed to False to avoid vLLM related CUDA library issues
    max_lora_rank=LORA_RANK,
    gpu_memory_utilization=0.55,
)

# =====================================================================
# 3. DATASET PREPARATION
# =====================================================================
def build_decision_points(raw_samples):
    examples = []
    for sample in raw_samples:
        tools_schema = json.loads(sample["tools"]) if isinstance(sample["tools"], str) else sample["tools"]
        pruned_tools = [t for t in tools_schema if t["name"] in TARGET_TOOLS]

        context = []
        for turn in sample["conversations"]:
            role = turn["from"]
            if role == "function_call":
                try:
                    fc = json.loads(turn["value"])
                except json.JSONDecodeError:
                    continue
                fn_name = fc.get("name")

                if fn_name == "think":
                    continue

                if fn_name in TARGET_TOOLS:
                    examples.append({
                        "system": sample["system"],
                        "tools": pruned_tools,
                        "context": list(context),
                        "target_name": fn_name,
                        "target_arguments": fc.get("arguments", {}),
                    })
                context.append(turn)
            else:
                context.append(turn)
    return examples

def render_prompt(example, tokenizer):
    messages = [{"role": "system", "content": example["system"]}]
    for turn in example["context"]:
        if turn["from"] == "human":
            messages.append({"role": "user", "content": turn["value"]})
        elif turn["from"] == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})
        elif turn["from"] == "function_call":
            messages.append({"role": "assistant", "content": turn["value"]})
        elif turn["from"] == "observation":
            messages.append({"role": "tool", "content": turn["value"]})

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=example["tools"],
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt

with open(SFT_DATA_PATH) as f:
    raw_samples = json.load(f)

decision_points = build_decision_points(raw_samples)
print(f"Built {len(decision_points)} GRPO decision points.")

grpo_rows = []
for ex in decision_points:
    prompt = render_prompt(ex, tokenizer)
    n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if n_tokens > MAX_PROMPT_TOKENS:
        continue
    grpo_rows.append({
        "prompt": prompt,
        "target_name": ex["target_name"],
        "target_arguments": json.dumps(ex["target_arguments"]),
    })

train_dataset = Dataset.from_list(grpo_rows)

# =====================================================================
# 4. REWARD FUNCTIONS
# =====================================================================
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOLCALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def _get_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)

def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        text = _get_text(comp)
        score = 0.0
        think_match = THINK_RE.search(text)
        tool_match = TOOLCALL_RE.search(text)

        if think_match and think_match.group(1).strip():
            score += 0.5
        if tool_match:
            if think_match is None or tool_match.start() > think_match.start():
                if len(TOOLCALL_RE.findall(text)) == 1:
                    score += 0.5
        rewards.append(score)
    return rewards

def json_validity_reward_func(prompts, completions, **kwargs):
    rewards = []
    for comp in completions:
        text = _get_text(comp)
        match = TOOLCALL_RE.search(text)
        if not match:
            rewards.append(0.0)
            continue
        try:
            payload = json.loads(match.group(1).strip())
            if isinstance(payload, dict) and "name" in payload and "arguments" in payload and isinstance(payload["arguments"], dict):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except json.JSONDecodeError:
            rewards.append(0.0)
    return rewards

def action_correctness_reward_func(prompts, completions, target_name, **kwargs):
    rewards = []
    for comp, gt_name in zip(completions, target_name):
        text = _get_text(comp)
        match = TOOLCALL_RE.search(text)
        if not match:
            rewards.append(0.0)
            continue
        try:
            payload = json.loads(match.group(1).strip())
            rewards.append(1.0 if payload.get("name") == gt_name else 0.0)
        except json.JSONDecodeError:
            rewards.append(0.0)
    return rewards

# =====================================================================
# 5. GRPO CONFIG & TRAIN
# =====================================================================
training_args = GRPOConfig(
    output_dir="outputs/grpo_airline",
    num_generations=8,
    max_completion_length=MAX_COMPLETION_LENGTH,
    temperature=1.0,
    top_p=1.0,
    use_vllm=False, # Changed to False to fix vLLM ImportError with GRPOTrainer
    vllm_gpu_memory_utilization=0.3,
    vllm_max_model_length=MAX_PROMPT_TOKENS + MAX_COMPLETION_LENGTH + 64,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-6,
    num_train_epochs=1,
    max_grad_norm=0.1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    bf16=is_bfloat16_supported(),
    fp16=not is_bfloat16_supported(),
    beta=0.0,
    loss_type="dapo",
    mask_truncated_completions=True,
    reward_weights=[1.0, 1.0, 2.0],
    logging_steps=1,
    log_completions=True,
    report_to="none",
    save_steps=25,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        json_validity_reward_func,
        action_correctness_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

model.save_lora("grpo_airline_lora")
print("Saved GRPO-trained LoRA adapter to grpo_airline_lora")
