from datasets import load_dataset
from transformers import AutoTokenizer

REF_ID = "Qwen/Qwen2-0.5B-Instruct"
IDX = [48598, 18024]

MAX_LEN = 512
MAX_PROMPT = 128

ds = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
tokenizer = AutoTokenizer.from_pretrained(REF_ID)

def split_prompt_and_answer(chat_list):
    # chat_list: list of {"role": ..., "content": ...}
    # 约定：最后一个 assistant 作为答案；其前面的全部当 prompt（含 system/user/assistant 多轮）
    # 这个切分与 TRL 的“提取 prompt + 套模板”逻辑接近（但我们主要是为了把文本可视化出来）
    last_asst = None
    last_asst_i = None
    for i in range(len(chat_list)-1, -1, -1):
        if chat_list[i].get("role") == "assistant":
            last_asst = chat_list[i].get("content", "")
            last_asst_i = i
            break
    if last_asst is None:
        return chat_list, ""
    prompt_msgs = chat_list[:last_asst_i]  # 不含最后的 assistant
    return prompt_msgs, last_asst

def show(idx):
    ex = ds[int(idx)]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    prompt_msgs, chosen_ans = split_prompt_and_answer(chosen)
    prompt_msgs2, rejected_ans = split_prompt_and_answer(rejected)

    # prompt_msgs 和 prompt_msgs2 理论上应一致（同一个 prompt），不一致也要暴露出来
    prompt_same = (prompt_msgs == prompt_msgs2)

    # 用 tokenizer 的 chat template 渲染“模型实际看到的 prompt 文本”
    # add_generation_prompt=True：会在末尾加 assistant 的起始标记，接下来才生成答案
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    # 检查 prompt 截断：只对 prompt 本身做 tokenization，看看是否超过 MAX_PROMPT
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    prompt_truncated = len(prompt_ids) > MAX_PROMPT

    # 检查 completion 截断：把 prompt+answer 拼起来（近似），看是否超过 MAX_LEN
    # 注意：真实训练里会有更精细的 mask/labels，但这里用长度做诊断足够了
    full_chosen = prompt_text + chosen_ans
    full_rejected = prompt_text + rejected_ans
    full_chosen_ids = tokenizer(full_chosen, add_special_tokens=False).input_ids
    full_rejected_ids = tokenizer(full_rejected, add_special_tokens=False).input_ids
    chosen_truncated = len(full_chosen_ids) > MAX_LEN
    rejected_truncated = len(full_rejected_ids) > MAX_LEN

    print("\n" + "="*110)
    print(f"[IDX={idx}]  prompt_same(chosen vs rejected prompt msgs) = {prompt_same}")
    print(f"prompt_tokens={len(prompt_ids)}  (MAX_PROMPT={MAX_PROMPT})  prompt_truncated={prompt_truncated}")
    print(f"full_chosen_tokens={len(full_chosen_ids)}  (MAX_LEN={MAX_LEN})  chosen_truncated={chosen_truncated}")
    print(f"full_rejected_tokens={len(full_rejected_ids)}  (MAX_LEN={MAX_LEN})  rejected_truncated={rejected_truncated}")

    print("\n----- PROMPT (after chat template, what model sees) -----\n")
    print(prompt_text)

    print("\n----- CHOSEN assistant content -----\n")
    print(chosen_ans)

    print("\n----- REJECTED assistant content -----\n")
    print(rejected_ans)

for idx in IDX:
    show(idx)
