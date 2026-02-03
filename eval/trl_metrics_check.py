import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

MODEL_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-4000"
BASE_ID  = "Qwen/Qwen2-0.5B-Instruct"
IDX = [41905, 7296, 1639, 48598, 18024]

device = "cpu"

# 1) 载入模型 / tokenizer（用同一个 tokenizer，避免模板差异）
tokenizer = AutoTokenizer.from_pretrained(BASE_ID)

policy = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device).eval()

# 2) 取 5 条原始样本（保持它原来的 list[dict] 格式，让 TRL 自己提取 prompt）
ds_all = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
small = ds_all.select(IDX)

# 3) DPOConfig：尽量贴近你训练时默认（512/128），并且强制不生成（只算loss/metrics）
args = DPOConfig(
    output_dir="/tmp/trl_metrics_check",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    max_length=512,
    max_prompt_length=128,
    remove_unused_columns=False,
    report_to=[],
)

trainer = DPOTrainer(
    model=policy,
    ref_model=None,          # 关键：让 TRL 按它的默认方式处理 ref（内部会拷贝/冻结）
    args=args,
    processing_class=tokenizer,   # 你这版trl支持这个
    train_dataset=small,
)

# 4) 从 trainer 拿一个 batch（经过它的 collator/tokenize）
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
# batch里一般有：prompt_input_ids / chosen_input_ids / rejected_input_ids ... 等

# 5) 用 TRL 内部函数算 metrics（这就是你 TensorBoard 里那套口径）
with torch.no_grad():
    loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")

print("loss:", float(loss))
# 把关键信息打印出来
keys = [k for k in metrics.keys() if any(s in k for s in ["rewards/", "logps/", "logits/"])]
for k in sorted(keys):
    v = metrics[k]
    if hasattr(v, "item"):
        v = v.item()
    print(f"{k}: {v}")
