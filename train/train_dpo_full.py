from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# 本地缓存里已有模型/数据，不需要学术加速
model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 全量数据（62k）
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# 数据是 list[{"role","content"}, ...]，摊平成 TRL DPO 需要的 prompt/chosen/rejected 三列（全是 str）
train_dataset = train_dataset.map(
    lambda x: {
        "prompt": x["chosen"][0]["content"],
        "chosen": x["chosen"][1]["content"],
        "rejected": x["rejected"][1]["content"],
    },
    remove_columns=train_dataset.column_names,
)

# 全量训练的“保守稳健”配置：小 batch + 梯度累积 + bf16 + checkpointing
training_args = DPOConfig(
    output_dir="/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full",
    run_name="qwen2_0.5b_dpo_full",
    report_to=["tensorboard"],

    # 训练长度：跑满 1 个 epoch（全量数据）
    num_train_epochs=1,

    # 显存/稳定性：单卡 4090 24GB 更稳的设置
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # 有效 batch=8
    bf16=True,                       # 4090 支持 bf16，通常比 fp16 更稳
    gradient_checkpointing=True,     # 降激活显存
    remove_unused_columns=False,

    # DPO 关键参数
    beta=0.1,

    # 序列长度：给个明确值，避免默认值未来变动
    max_length=512,
    max_prompt_length=128,

    # 学习率：保守起步（你可以后面做对照实验再调）
    learning_rate=1e-6,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",

    # 日志/保存
    logging_steps=10,
    save_steps=1000,
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

trainer.train()
