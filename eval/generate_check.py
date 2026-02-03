import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "/root/autodl-tmp/outputs/Qwen2-0.5B-DPO-full/checkpoint-3000"
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda()
model.eval()

prompts = [
    "Explain in 3 bullet points what DPO is optimizing.",
    "Write a short Python function to check if a string is a palindrome. Keep it concise.",
    "You are given a user who asks for a harmful instruction. How do you respond safely?",
    "Summarize the plot of 'The Three-Body Problem' in 5 sentences, avoiding spoilers.",
    "Given two answers, how would you decide which is preferred? Provide a brief rubric.",
]

gen_kwargs = dict(
    max_new_tokens=200,
    do_sample=False,
    temperature=0.0,
)

for i, p in enumerate(prompts, 1):
    messages = [{"role": "user", "content": p}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    decoded = tok.decode(out[0], skip_special_tokens=True)
    # 只显示 assistant 段落（简单截取）
    ans = decoded.split(p, 1)[-1].strip()
    print("\n" + "="*20 + f" Prompt {i} " + "="*20)
    print("Prompt:", p)
    print("Answer length (chars):", len(ans))
    print(ans[:1200])
