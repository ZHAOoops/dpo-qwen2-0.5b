import argparse
import random
from collections import defaultdict

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
    ap.add_argument("--num_samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=128)
    ap.add_argument("--cpu", action="store_true")

    # bucketing
    ap.add_argument("--bucket_by", type=str, default="prompt",
                    choices=["prompt", "chosen", "rejected", "pair_total"],
                    help="which length to bucket on")
    ap.add_argument("--bucket_edges", type=str, default="0,32,64,96,128,99999",
                    help="comma-separated bucket edges, e.g. 0,32,64,96,128,99999")
    return ap.parse_args()


def to_edges(s: str):
    edges = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(edges) < 2:
        raise ValueError("bucket_edges needs at least 2 numbers")
    return edges


def bucket_id(x: int, edges):
    # edges like [0,32,64,...,INF]
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    return len(edges) - 2


def bucket_name(i: int, edges):
    return f"[{edges[i]},{edges[i+1]})"


@torch.no_grad()
def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, use_fast=True)
    # 很多 causal LM 没有 pad_token；DPO collator 会用 padding，所以这里补齐
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.policy_ckpt).to(device).eval()
    ref    = AutoModelForCausalLM.from_pretrained(args.ref_model).to(device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    ds_all = load_dataset(args.dataset, split=args.split)
    n = len(ds_all)
    k = min(args.num_samples, n)
    rng = random.Random(args.seed)
    idxs = rng.sample(range(n), k)
    small = ds_all.select(idxs)

    # 关键：batch_size=1，才能把每条样本落到桶里（不用拆 batch）
    dpo_args = DPOConfig(
        output_dir="/tmp/trl_metrics_lenbuckets",
        per_device_train_batch_size=1,
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
    pad_id = tokenizer.pad_token_id
    edges = to_edges(args.bucket_edges)

    # overall accum
    overall = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}

    # per-bucket accum
    buckets = defaultdict(lambda: {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0})

    for batch in dl:
        # move to device
        for k2, v2 in batch.items():
            if torch.is_tensor(v2):
                batch[k2] = v2.to(device)

        loss, metrics = trainer.get_batch_loss_metrics(policy, batch, train_eval="train")

        # lengths (batch_size=1)
        def nonpad_len(t):
            # t: [1, L]
            return int((t[0] != pad_id).sum().item())

        # TRL DPO collator keys (usually exist):
        # prompt_input_ids, chosen_input_ids, rejected_input_ids
        plen = nonpad_len(batch["prompt_input_ids"]) if "prompt_input_ids" in batch else None
        clen = nonpad_len(batch["chosen_input_ids"]) if "chosen_input_ids" in batch else None
        rlen = nonpad_len(batch["rejected_input_ids"]) if "rejected_input_ids" in batch else None

        if args.bucket_by == "prompt":
            keylen = plen
        elif args.bucket_by == "chosen":
            keylen = clen
        elif args.bucket_by == "rejected":
            keylen = rlen
        else:
            keylen = (clen or 0) + (rlen or 0)

        bid = bucket_id(int(keylen), edges)
        bname = bucket_name(bid, edges)

        l = float(loss)
        m = float(metrics["rewards/margins"])
        a = float(metrics["rewards/accuracies"])

        overall["loss"] += l
        overall["margin"] += m
        overall["acc"] += a
        overall["n"] += 1

        buckets[bname]["loss"] += l
        buckets[bname]["margin"] += m
        buckets[bname]["acc"] += a
        buckets[bname]["n"] += 1

    def mean(x): 
        return x["loss"]/x["n"], x["margin"]/x["n"], x["acc"]/x["n"]

    oloss, omargin, oacc = mean(overall)
    print(f"device: {device}")
    print(f"samples: {k}  eval_batches(batch_size=1): {overall['n']}")
    print(f"overall_mean_loss: {oloss:.6f}")
    print(f"overall_mean_reward_margin: {omargin:.6f}")
    print(f"overall_mean_reward_acc: {oacc:.6f}")
    print()
    print(f"bucket_by: {args.bucket_by}  edges: {edges}")
    print("bucket\tN\tmean_loss\tmean_margin\tmean_acc")
    for b in sorted(buckets.keys(), key=lambda s: int(s.split(",")[0].strip("["))):
        x = buckets[b]
        ml, mm, ma = mean(x)
        print(f"{b}\t{x['n']}\t{ml:.6f}\t{mm:.6f}\t{ma:.6f}")


if __name__ == "__main__":
    main()
