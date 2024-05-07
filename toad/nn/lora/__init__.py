def get_lora_model(model, config = None):
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    peft_config = LoraConfig(
        # task_type=TaskType.SEQ_2_SEQ_LM,
        # task_type=TaskType.FEATURE_EXTRACTION,
        target_modules = ['linear_1'],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    
    return model
