import argparse
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", type=str, required=True, help="local path to TRL checkpoint dir")
    ap.add_argument("--ref_model", type=str, required=True, help="HF repo id for reference model/tokenizer")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, use_fast=True)

    policy = AutoModelForCausalLM.from_pretrained(args.policy_ckpt).to(device).eval()
    ref    = AutoModelForCausalLM.from_pretrained(args.ref_model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    ds_all = load_dataset(args.dataset, split=args.split)

    # 随机抽样（可复现）
    n = len(ds_all)
    k = min(args.num_samples, n)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(n), k)
    small = ds_all.select(idxs)

    dpo_args = DPOConfig(
        output_dir="/tmp/trl_metrics_check_ref",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=small,
    )

    dl = trainer.get_train_dataloader()

    # 聚合统计：mean(margin), mean(acc), mean(loss)
    loss_sum = 0.0
    margin_sum = 0.0
    acc_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")
            loss_sum += float(loss)
            margin_sum += float(metrics["rewards/margins"])
            acc_sum += float(metrics["rewards/accuracies"])
            n_batches += 1

    print(f"device: {device}")
    print(f"samples: {k}  batches: {n_batches}  batch_size: {args.batch_size}")
    print(f"mean_loss: {loss_sum / n_batches:.6f}")
    print(f"mean_reward_margin: {margin_sum / n_batches:.6f}")
    print(f"mean_reward_acc: {acc_sum / n_batches:.6f}")

if __name__ == "__main__":
    main()
