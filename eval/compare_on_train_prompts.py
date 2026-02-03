import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_ID = "Qwen/Qwen2-0.5B-Instruct"
CKPT_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-4000"

# 为了不干扰你正在训练：默认用CPU（你也可以改成 "cuda" 但会和训练抢GPU）
DEVICE = "cpu"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.bfloat16

N = 5
SEED = 42
MAX_NEW_TOKENS = 160

def build_prompt_text(tok, prompt: str) -> str:
    # 用 chat template 构造“用户提问 + assistant 开始”形式
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

def avg_logprob_of_answer(model, tok, prompt_text: str, answer: str) -> float:
    # 计算 log p(answer | prompt) 的“逐token平均”
    full_text = prompt_text + answer
    enc = tok(full_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)

    # prompt 部分长度（用于只取 answer token 的 logprob）
    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    if input_ids.shape[1] <= prompt_len:
        return float("nan")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [B, T, V]
        logprobs = torch.log_softmax(logits, dim=-1)

    # 目标token是 input_ids[:, 1:]，对应预测位置是 logits[:, :-1]
    target = input_ids[:, 1:]
    lp = logprobs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    # 只取 answer 对应的 token 区间（从 prompt_len 到末尾-1 的预测）
    start = prompt_len - 1  # 因为 target 从原序列第2个token开始对齐
    ans_lp = lp[:, start:]
    ans_target = target[:, start:]

    # 排除 padding（虽然这里基本没有pad，但保险）
    ans_mask = (ans_target != tok.pad_token_id) if tok.pad_token_id is not None else torch.ones_like(ans_target, dtype=torch.bool)
    ans_lp = ans_lp[ans_mask]

    if ans_lp.numel() == 0:
        return float("nan")
    return ans_lp.mean().item()

def generate(model, tok, prompt_text: str) -> str:
    enc = tok(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    # 粗暴截取：取 prompt 后面的部分
    return decoded.split(prompt_text, 1)[-1].strip()

def load_pair_dataset():
    ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")  # 走本地cache
    return ds

def to_pair(ex):
    # ex["chosen"] / ex["rejected"] 是 list[{"role","content"}, ...]
    prompt = ex["chosen"][0]["content"]
    chosen = ex["chosen"][1]["content"]
    rejected = ex["rejected"][1]["content"]
    return prompt, chosen, rejected

def main():
    random.seed(SEED)
    ds = load_pair_dataset()
    idxs = random.sample(range(len(ds)), N)

    tok_base = AutoTokenizer.from_pretrained(BASE_ID)
    tok_ckpt = AutoTokenizer.from_pretrained(CKPT_ID)

    # 用各自tokenizer的chat template（Qwen通常一致，但稳妥）
    model_base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    model_ckpt = AutoModelForCausalLM.from_pretrained(CKPT_ID, torch_dtype=DTYPE).to(DEVICE).eval()

    for k, i in enumerate(idxs, 1):
        prompt, chosen, rejected = to_pair(ds[i])

        ptxt_base = build_prompt_text(tok_base, prompt)
        ptxt_ckpt = build_prompt_text(tok_ckpt, prompt)

        # 科学对比：平均logprob（越大越好；chosen 应该 > rejected）
        base_lp_c = avg_logprob_of_answer(model_base, tok_base, ptxt_base, chosen)
        base_lp_r = avg_logprob_of_answer(model_base, tok_base, ptxt_base, rejected)
        ckpt_lp_c = avg_logprob_of_answer(model_ckpt, tok_ckpt, ptxt_ckpt, chosen)
        ckpt_lp_r = avg_logprob_of_answer(model_ckpt, tok_ckpt, ptxt_ckpt, rejected)

        # 生成对比（同一prompt）
        base_gen = generate(model_base, tok_base, ptxt_base)
        ckpt_gen = generate(model_ckpt, tok_ckpt, ptxt_ckpt)

        print("\n" + "="*90)
        print(f"[Sample {k}] dataset_index={i}")
        print("- Prompt:\n", prompt)

        print("\n[Teacher-forced avg logprob]")
        print(f"  BASE  chosen={base_lp_c:.4f}  rejected={base_lp_r:.4f}  (margin={base_lp_c-base_lp_r:.4f})")
        print(f"  DPO   chosen={ckpt_lp_c:.4f}  rejected={ckpt_lp_r:.4f}  (margin={ckpt_lp_c-ckpt_lp_r:.4f})")

        print("\n[Greedy generation]")
        print(f"  BASE len(chars)={len(base_gen)}:\n{base_gen[:900]}")
        print(f"\n  DPO  len(chars)={len(ckpt_gen)}:\n{ckpt_gen[:900]}")

if __name__ == "__main__":
    main()
