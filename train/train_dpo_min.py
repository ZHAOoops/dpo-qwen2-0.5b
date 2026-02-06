from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)


train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:2000]")
train_dataset = train_dataset.map(lambda x: {"prompt": x["chosen"][0]["content"], "chosen": x["chosen"][1]["content"], "rejected": x["rejected"][1]["content"]}, remove_columns=train_dataset.column_names)


training_args = DPOConfig(
    output_dir="/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-min",
    logging_steps=10,
    save_steps=200,
    report_to=["tensorboard"],
    run_name="qwen2_0.5b_dpo_min",
    max_steps=60,                 
    per_device_train_batch_size=1 
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

trainer.train()
