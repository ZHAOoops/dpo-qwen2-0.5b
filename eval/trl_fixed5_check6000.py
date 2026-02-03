import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

POLICY_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-6000"
REF_ID    = "Qwen/Qwen2-0.5B-Instruct"
IDX = [41905, 7296, 1639, 48598, 18024]

device = "cpu"
ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
small = ds.select(IDX)

tokenizer = AutoTokenizer.from_pretrained(REF_ID)
policy = AutoModelForCausalLM.from_pretrained(POLICY_ID).to(device).eval()
ref    = AutoModelForCausalLM.from_pretrained(REF_ID).to(device).eval()
for p in ref.parameters():
    p.requires_grad_(False)

args = DPOConfig(
    output_dir="/tmp/trl_fixed5_check",
    per_device_train_batch_size=1,
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
with torch.no_grad():
    for i, batch in enumerate(dl):
        loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")
        m = float(metrics["rewards/margins"])
        a = float(metrics["rewards/accuracies"])
        print(f"idx={IDX[i]}  margin={m:+.4f}  acc={a:.3f}  loss={float(loss):.4f}")
