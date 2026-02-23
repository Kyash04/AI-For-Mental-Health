import os
import re
import torch
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

os.environ["WANDB_DISABLED"] = "true"

# ── System prompt (single source of truth) ────────────────────────────
SYSTEM_PROMPT = (
    "You are HealthMate, a compassionate mental health support assistant. "
    "You are NOT a doctor or therapist. You provide empathetic, supportive responses. "
    "Do NOT introduce yourself as anyone other than HealthMate. "
    "Keep responses warm and focused, between 2-3 sentences."
)

# ── Text cleaning ──────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # FIX 7: Remove encoding artifacts baked into your Kaggle CSV
    text = text.replace("Â", "").replace("\xa0", " ")
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def truncate_at_sentence(text: str, max_words: int = 100) -> str:
    """FIX 6: Cap response length so model learns consistent output size"""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = ' '.join(words[:max_words])
    last_stop = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
    if last_stop > len(truncated) * 0.5:
        return truncated[:last_stop + 1]
    return truncated + "."

# ── Dataset formatter ──────────────────────────────────────────────────
def format_prompt(row: dict) -> dict:
    """
    FIX 1: Every turn must end with </s>.
    Without this, TinyLlama never learned when a speaker's turn ends,
    which is the root cause of the hallucinated nurse midwife response.
    """
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{row['context']}</s>\n"
        f"<|assistant|>\n{row['response']}</s>"   # ← was missing in old train.py
    )
    return {"text": text}

# ── Data loading + cleaning ────────────────────────────────────────────
def load_and_prepare_dataset(data_file_path: str, val_split: float = 0.1):
    print(f"Loading dataset from: {data_file_path}")
    df = pd.read_csv(data_file_path, encoding='utf-8', on_bad_lines='skip')

    # Normalise column names (handles Context/context/Question etc.)
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for c in df.columns:
        if any(k in c for k in ['context', 'question', 'input']):
            col_map[c] = 'context'
        elif any(k in c for k in ['response', 'answer', 'output']):
            col_map[c] = 'response'
    df = df.rename(columns=col_map)

    if 'context' not in df.columns or 'response' not in df.columns:
        raise ValueError(
            f"Could not find context/response columns. Found: {df.columns.tolist()}"
        )

    df = df[['context', 'response']].copy()
    df['context']  = df['context'].apply(clean_text)
    df['response'] = df['response'].apply(clean_text)

    # ── Quality filters ──────────────────────────────────────────────
    before = len(df)
    df = df[df['context'].str.len() > 30]                           # Drop truncated contexts
    df = df[df['response'].str.len() > 20]                          # Drop near-empty responses
    df = df[~df['context'].str.strip().str.lower()                  # Drop "I'm going through" rows
              .str.startswith("i'm going through")]
    df = df[~df['response'].str.contains(r'Â|\xa0', na=False)]     # Remove remaining artifacts
    df = df.drop_duplicates(subset='context')                       # Remove duplicate prompts
    df['response'] = df['response'].apply(truncate_at_sentence)     # Cap response length
    df = df.reset_index(drop=True)

    print(f"Dataset: {before} raw rows → {len(df)} clean rows")
    if len(df) < 200:
        print("⚠️  WARNING: Very few rows after cleaning. Consider a larger/cleaner dataset.")

    # ── Format + FIX 4: train/val split ─────────────────────────────
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_prompt, remove_columns=['context', 'response'])
    split   = dataset.train_test_split(test_size=val_split, seed=42)

    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    print("\n── Sample formatted entry ──")
    print(split['train'][0]['text'][:400])
    return split['train'], split['test']

# ── Main training function ─────────────────────────────────────────────
def main(args):
    # ── FIX 3 & 5: 4-bit quantization, T4-compatible dtype ──────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # FIX 5: float16 not bfloat16 for T4
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── FIX 2: r=8 instead of r=64 ──────────────────────────────────
    # r=64 on a small mental health dataset = severe overfitting.
    # r=8 gives the model just enough capacity to learn new patterns
    # without forgetting its base instruction-following ability.
    peft_config = LoraConfig(
        r=8,                            # WAS 64 — this was a major bug
        lora_alpha=16,                  # Rule of thumb: alpha = 2 * r
        lora_dropout=0.05,              # WAS 0.1 — lower is fine with small r
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    train_ds, val_ds = load_and_prepare_dataset(args.data_file)

    training_arguments = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=2,                     # 2 is safer than 1 with small r
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,          # Effective batch = 16

        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.001,
        max_grad_norm=0.3,

        # ── Eval & checkpointing ──────────────
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,            # Saves best, not last checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # ── Hardware ─────────────────────────
        fp16=True,                              # FIX 5: T4 supports fp16, not bf16
        bf16=False,
        optim="paged_adamw_8bit",               # FIX 8: less VRAM than 32bit
        gradient_checkpointing=True,
        group_by_length=True,

        logging_steps=10,
        report_to="none",
    )

    # ── FIX 3: DataCollatorForCompletionOnlyLM ───────────────────────
    # This is the second most important fix after the </s> tokens.
    # It ensures the model ONLY learns to predict the assistant's reply,
    # not the system prompt or user message. Without this, the model
    # gets confused about its own identity (hence the nurse midwife bug).
    response_template = "<|assistant|>"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,            # FIX 4: validation set wired in
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
        data_collator=collator,         # FIX 3: completion-only collator
        packing=False,
    )

    print("\n🚀 Starting training...")
    print("   Watch eval/loss — should decrease each checkpoint.")
    print("   If it goes UP after epoch 1, that's overfitting. Stop early.\n")
    trainer.train()

    # Save adapter only (small file, loads fast in app)
    print(f"\nSaving adapter to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ Training complete!\n")

    # ── Quick sanity test ────────────────────────────────────────────
    print("── Running post-training sanity test ──")
    test_cases = ["Hello!", "I feel really anxious.", "Who are you?"]
    bad_patterns = ["my name is", "i am a certified", "i work at",
                    "as a nurse", "as a doctor", "midwife"]

    for prompt in test_cases:
        formatted = (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{prompt}</s>\n"
            f"<|assistant|>\n"
        )
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.65,
                top_p=0.88,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        ).strip()
        hallucinated = any(p in reply.lower() for p in bad_patterns)
        status = "⚠️  HALLUCINATION" if hallucinated else "✅"
        print(f"\n{status}  User    : {prompt}")
        print(f"      Model   : {reply[:200]}")

    print("\n── Done. Download the adapter folder from Google Drive. ──")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama for HealthMate")
    parser.add_argument("--base_model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_file",  type=str, required=True,
                        help="Path to train.csv")
    parser.add_argument("--output_dir", type=str,
                        default="./my_finetuned_healthmate_model")
    args = parser.parse_args()
    main(args)