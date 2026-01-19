from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from model import get_model

def get_trainer():
    sft_config = SFTConfig(
        ## GROUP 1: Memory usage
        # These arguments will squeeze the most out of your GPU's RAM
        # Checkpointing
        gradient_checkpointing=True,    # this saves a LOT of memory
        # Set this to avoid exceptions in newer versions of PyTorch
        gradient_checkpointing_kwargs={'use_reentrant': False}, 
        # Gradient Accumulation / Batch size
        # Actual batch (for updating) is same (1x) as micro-batch size
        gradient_accumulation_steps=1,  
        # The initial (micro) batch size to start off with
        per_device_train_batch_size=16, 
        # If batch size would cause OOM, halves its size until it works
        auto_find_batch_size=True,

        ## GROUP 2: Dataset-related
        max_length=64, # renamed in v0.20
        # Dataset
        # packing a dataset means no padding is needed
        packing=True,
        packing_strategy='wrapped', # added to approximate original packing behavior

        ## GROUP 3: These are typical training parameters
        num_train_epochs=10,
        learning_rate=3e-4,
        # Optimizer
        # 8-bit Adam optimizer - doesn't help much if you're using LoRA!
        optim='paged_adamw_8bit',       
        
        ## GROUP 4: Logging parameters
        logging_steps=10,
        logging_dir='./logs',
        output_dir='./phi3-mini-yoda-adapter',
        report_to='none'.

        # ensures bf16 (the new default) is only used when it is actually available
        bf16=torch.cuda.is_bf16_supported(including_emulation=False)
    )
    model = get_model(device='cuda')
    
    train_dataset = load_dataset(
        "json",
        data_files="chat_finetune_train.json",
        split="train",
    )

    eval_dataset = load_dataset(
        "json",
        data_files="chat_finetune_val.json",
        split="train",
    )

    # Sanity check
    assert "messages" in train_dataset.column_names
    assert "messages" in eval_dataset.column_names

    tokenizer = model.tokenizer
    config = model.peft_config

    trainer = SFTTrainer(
        model=model, # the underlying Phi-3 model
        peft_config=config,  # added to fix issue in TRL>=0.20
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    return trainer