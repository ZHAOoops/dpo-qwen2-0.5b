import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

POLICY_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-4000"
REF_ID    = "Qwen/Qwen2-0.5B-Instruct"
IDX = [41905, 7296, 1639, 48598, 18024]

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(REF_ID)

policy = AutoModelForCausalLM.from_pretrained(POLICY_ID).to(device).eval()
ref    = AutoModelForCausalLM.from_pretrained(REF_ID).to(device).eval()
for p in ref.parameters():
    p.requires_grad_(False)

ds_all = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
small = ds_all.select(IDX)

args = DPOConfig(
    output_dir="/tmp/trl_metrics_check_ref",
    per_device_train_batch_size=2,   # 用2条避免单样本边界
    gradient_accumulation_steps=1,
    max_length=512,
    max_prompt_length=128,
    remove_unused_columns=False,
    report_to=[],
)

trainer = DPOTrainer(
    model=policy,
    ref_model=ref,
    args=args,
    processing_class=tokenizer,
    train_dataset=small,
)

dl = trainer.get_train_dataloader()
batch = next(iter(dl))

with torch.no_grad():
    loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")

print("loss:", float(loss))
for k in ["rewards/chosen","rewards/rejected","rewards/margins","rewards/accuracies",
          "logps/chosen","logps/rejected","logits/chosen","logits/rejected"]:
    if k in metrics:
        v = metrics[k]
        if hasattr(v, "item"):
            v = v.item()
        print(f"{k}: {v}")
    else:
        print(f"{k}: <missing>")
