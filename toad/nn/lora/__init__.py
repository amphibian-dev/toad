def get_lora_model(model, config = None):
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    model = get_peft_model(model, config)
    
    return model
