import os
import torch
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig



def get_model(device):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32
    )
    repo_id = 'microsoft/Phi-3-mini-4k-instruct'
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, device_map=device, quantization_config=bnb_config
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        # the rank of the adapter, the lower the fewer parameters you'll need to train
        r=8,                   
        lora_alpha=16, # multiplier, usually 2*r
        bias="none",           
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        # Newer models, such as Phi-3 at time of writing, may require 
        # manually setting target modules
        target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', 'down_proj'],
    )
    model = get_peft_model(model, peft_config)

    return model


def get_tokenizer():
    repo_id = 'microsoft/Phi-3-mini-4k-instruct'
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    return tokenizer




