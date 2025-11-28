import os
import random
import json
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(cfg, model_name, trust_remote_code=True, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False, cache_dir =cfg.get("model", {}).get("cache_dir", None))
    # For Qwen-like models, AutoModelForCausalLM often works; adjust if model requires custom class.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        cache_dir=cfg.get("model", {}).get("cache_dir", None),
    )
    return tokenizer, model

def load_dataset_from_config(cfg):
    ds_cfg = cfg.get("dataset", {})
    name = ds_cfg.get("name", "wmt14")
    src_key =  ds_cfg.get("src_lang", "de")
    tgt_key = ds_cfg.get("tgt_lang", "en")
    max_examples = ds_cfg.get("max_examples", None)
    cache_dir = ds_cfg.get("cache_dir", None)

    # load dataset
    try:
        ds = load_dataset(name, ds_cfg.get('pair'), cache_dir=cache_dir)
    except Exception as e:
        raise RuntimeError(f"加载数据集失败: {name}/{ds_cfg.get('pair')}: {e}")

    # map to src_text / tgt_text
    def _extract(example):
        # 优先处理 translation 字段（wmt 风格）
        if "translation" in example and isinstance(example["translation"], dict):
            src = example["translation"].get(src_key)
            tgt = example["translation"].get(tgt_key)
        else:
            # 退回到常见字段名
            src = example.get("source") or example.get("src") or example.get("sentence") or example.get("text")
            tgt = example.get("target") or example.get("tgt") or example.get("translation_text") or example.get("translation")
            # 如果 translation 是字符串且未在上面处理，则尝试直接使用
            if isinstance(tgt, dict):
                tgt = None
        return {"src_text": src, "tgt_text": tgt}

    ds = ds.map(_extract)

    # 过滤空样本
    ds = ds.filter(lambda x: x["src_text"] is not None and x["src_text"] != "" and x["tgt_text"] is not None and x["tgt_text"] != "")

    return ds

def apply_peft_lora(model, cfg):
    lora_cfg = cfg.get("lora", None)

    if lora_cfg:
        peft_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            bias=lora_cfg.get("bias", "none"),
            task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    # ensure caching disabled for training with Trainer
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model

def prepare_decoder_map_fn(cfg, tokenizer):
    template = cfg.get("prompt", {}).get("template", "{text}")
    max_length = cfg.get("tokenization", {}).get("max_length", 1024)

    def _map_batch(batch):
        input_ids = []
        attention_mask = []
        labels = []
        srcs = batch["src_text"]
        tgts = batch["tgt_text"]
        for src, tgt in zip(srcs, tgts):
            prompt = template.format(src_lang=cfg.get("src_lang", {}).get("src_lang", ""), tgt_lang=cfg.get("tgt_lang", {}).get("tgt_lang", ""), text=src)
            full = prompt + " " + tgt
            tok_full = tokenizer(full, truncation=True, max_length=max_length, padding=False)
            tok_prompt = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
            lbls = tok_full["input_ids"].copy()
            prompt_len = len(tok_prompt["input_ids"])
            if prompt_len > 0:
                lbls[:prompt_len] = [-100] * prompt_len
            input_ids.append(tok_full["input_ids"])
            attention_mask.append(tok_full.get("attention_mask", [1] * len(tok_full["input_ids"])))
            labels.append(lbls)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _map_batch

def train_from_ds(cfg, tokenizer, model, ds):
    set_seed(cfg.get("train", {}).get("seed", 42))
    model = apply_peft_lora(model, cfg)

    map_fn = prepare_decoder_map_fn(cfg, tokenizer)
    tokenized = ds.map(map_fn, batched=True)

    train_cfg = cfg.get("train", cfg.get("training", {}))
    output_dir = train_cfg.get("output_dir", "outputs/lora")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", train_cfg.get("batch_size", 4)),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("epochs", train_cfg.get("num_train_epochs", 1)),
        learning_rate=train_cfg.get("learning_rate", train_cfg.get("lr", 2e-5)),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 100),
        save_steps=train_cfg.get("save_steps", None),
        eval_strategy=train_cfg.get("eval_strategy", "no"),
        eval_steps=train_cfg.get("eval_steps", None),
        remove_unused_columns=False,
    )

    data_collator = lambda features: tokenizer.pad(features, padding="longest", return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    # 保存 LoRA 权重与 tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    # parser = arg_parser()
    # args = parser.parse_args()

    config = load_yaml('config.yaml')
    ds = load_dataset_from_config(config)
    tokenizer, model = load_model_and_tokenizer(config, config['model']['name'])
    train_from_ds(config, tokenizer, model, ds)