import torch
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

os.environ["WANDB_DISABLED"] = "true"

def format_prompt(row):
    system_prompt = ( "You are HealthMate, a friendly and supportive mental health assistant. " "Respond with empathy, understanding, and provide helpful, safe advice. "
                      "Your goal is to make the user feel heard and supported." )
    user_prompt = row['Context']
    model_response = row['Response']

    text = f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n{model_response}"
    return {"text":text}

def load_prep_dataset(data_file_path):
    print(f"Loading dataset from: {data_file_path}")
    df = pd.read_csv(data_file_path)
    df = df.dropna(subset=['Context','Response'])
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_prompt)

    print("Dataset formatted successfully")
    return dataset

def main(args):
    #Loading the base model
    base_model = args.base_model

    # # Configuring 4-bit quantization (to save memory)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=False,
    # )

    #Loading model with quantization
    print(f"Loading Model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto" #Automatically uses the Colab GPU
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token #Setting padding token
    tokenizer.padding_side = "right"

    # Configuring LoRA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_prep_dataset(args.data_file)

    # Setting up Training Arguments
    training_arguments = TrainingArguments(
        output_dir = args.output_dir,
        num_train_epochs=1, #Start with 1 epoch
        per_device_train_batch_size=4, #Batch size
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=100, #Save a checkpoint every 50 steps
        logging_steps=10, # Log progress every 10 steps
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True, #Use bf16 for speed on new GPU
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none"
    )

    #Initializing the SFTTrainer (Supervised Fine-Tuning Trainer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        #max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    #Start training
    print("Starting training...")
    trainer.train()

    #Saving the final model
    print(f"Training complete. Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model using a Kaggle dataset.")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="The name of the base model to fine-tune.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the .csv training data file(from Kaggle).")
    parser.add_argument("--output_dir", type=str, default="./my-healthmate-model", help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args)