import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

REF_ID = "Qwen/Qwen2-0.5B-Instruct"
POLICY_ID = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-7766"
IDX = [48598, 18024]

tokenizer = AutoTokenizer.from_pretrained(REF_ID)
policy = AutoModelForCausalLM.from_pretrained(POLICY_ID).eval()
ref    = AutoModelForCausalLM.from_pretrained(REF_ID).eval()
device="cpu"
policy.to(device); ref.to(device)

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

def split_prompt_and_answer(chat_list):
    last_asst = None; last_i=None
    for i in range(len(chat_list)-1, -1, -1):
        if chat_list[i].get("role") == "assistant":
            last_asst = chat_list[i].get("content","")
            last_i=i; break
    prompt_msgs = chat_list[:last_i]
    return prompt_msgs, last_asst

@torch.no_grad()
def sum_and_mean_logp(model, prompt_msgs, answer_text):
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + answer_text

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    # labels = input_ids shifted; 只在 answer 部分算logp：我们用 token count 来切分
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = prompt_ids.shape[1]
    # 预测位置从 1..T-1，对应 label 是 input_ids[:,1:]
    logits = model(input_ids).logits  # [1,T,V]
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # [1,T-1,V]
    labels = input_ids[:, 1:]  # [1,T-1]

    # 取每个位置的 logp
    lp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [1,T-1]

    # completion 的 label 区间：从 prompt_len-1 开始（因为 shift 了1）
    start = max(prompt_len-1, 0)
    comp_lp = lp[:, start:]
    sum_lp = float(comp_lp.sum())
    mean_lp = float(comp_lp.mean())
    tok = int(comp_lp.numel())
    return tok, sum_lp, mean_lp

for idx in IDX:
    ex = ds[int(idx)]
    prompt_msgs, chosen_ans = split_prompt_and_answer(ex["chosen"])
    prompt_msgs2, rejected_ans = split_prompt_and_answer(ex["rejected"])

    # sanity
    assert prompt_msgs == prompt_msgs2

    c_tok, c_sum_pi, c_mean_pi = sum_and_mean_logp(policy, prompt_msgs, chosen_ans)
    r_tok, r_sum_pi, r_mean_pi = sum_and_mean_logp(policy, prompt_msgs, rejected_ans)

    c_tok2, c_sum_ref, c_mean_ref = sum_and_mean_logp(ref, prompt_msgs, chosen_ans)
    r_tok2, r_sum_ref, r_mean_ref = sum_and_mean_logp(ref, prompt_msgs, rejected_ans)

    d_c = c_sum_pi - c_sum_ref
    d_r = r_sum_pi - r_sum_ref
    margin = d_c - d_r

    print("\n"+"="*90)
    print(f"IDX={idx}")
    print(f"chosen_tok={c_tok} rejected_tok={r_tok}")
    print(f"policy  sum_logp: chosen={c_sum_pi:.2f} rejected={r_sum_pi:.2f}")
    print(f"ref     sum_logp: chosen={c_sum_ref:.2f} rejected={r_sum_ref:.2f}")
    print(f"Δ(policy-ref)    chosen={d_c:.2f} rejected={d_r:.2f}  margin={margin:.2f}")
    print(f"policy mean_logp: chosen={c_mean_pi:.4f} rejected={r_mean_pi:.4f}")
    print(f"ref    mean_logp: chosen={c_mean_ref:.4f} rejected={r_mean_ref:.4f}")
