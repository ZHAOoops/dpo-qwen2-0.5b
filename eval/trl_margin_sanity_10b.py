import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

POLICY_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-6000"
REF_ID    = "Qwen/Qwen2-0.5B-Instruct"

# 尽量贴近训练默认截断（你训练时也提示默认 512/128）
MAX_LEN = 512
MAX_PROMPT = 128

# 为了让统计更接近“训练时的一个优化 step”，这里用 batch=8（CPU 会慢一点，但只取 10 个 batch）
BATCH = 8
N_BATCH = 10
SEED = 0

device = "cpu"

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
small = ds.shuffle(seed=SEED).select(range(256))

tokenizer = AutoTokenizer.from_pretrained(REF_ID)

policy = AutoModelForCausalLM.from_pretrained(POLICY_ID).to(device).eval()
ref    = AutoModelForCausalLM.from_pretrained(REF_ID).to(device).eval()
for p in ref.parameters():
    p.requires_grad_(False)

args = DPOConfig(
    output_dir="/tmp/trl_margin_sanity",
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=1,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT,
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

margins = []
accs = []
with torch.no_grad():
    for bi, batch in enumerate(dl):
        loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")
        m = metrics.get("rewards/margins", None)
        a = metrics.get("rewards/accuracies", None)
        m = float(m.item()) if hasattr(m, "item") else float(m)
        a = float(a.item()) if hasattr(a, "item") else float(a)
        margins.append(m)
        accs.append(a)
        print(f"[batch {bi:02d}] loss={float(loss):.4f}  margin={m:+.4f}  acc={a:.3f}")
        if bi + 1 >= N_BATCH:
            break

import numpy as np
margins = np.array(margins, dtype=np.float64)
accs = np.array(accs, dtype=np.float64)
print("\n===== summary over", len(margins), "batches =====")
print("mean(margin):", float(margins.mean()))
print("frac(margin>0):", float((margins > 0).mean()))
print("mean(acc):", float(accs.mean()))
