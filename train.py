import argparse
import torch
import yaml

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def arg_parser():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--continue', type=bool, required=True, help='Number of training epochs')
    return parser


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model_and_tokenizer(model_name, trust_remote_code=True, device="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=False)
    # For Qwen-like models, AutoModelForCausalLM often works; adjust if model requires custom class.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
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

    # 限制样本数（可选）
    if max_examples is not None:
        n = min(len(ds), int(max_examples))
        ds = ds.select(range(n))

    return ds


if __name__ == '__main__':
    # parser = arg_parser()
    # args = parser.parse_args()

    config = load_yaml('config.yaml')
    ds = load_dataset_from_config(config)
    tokenizer, model = load_model_and_tokenizer(config['model']['name'])
