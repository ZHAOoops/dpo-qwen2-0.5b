# Qwen2-0.5B + TRL DPO on ultrafeedback_binarized (AutoDL 单卡4090)

这是我用一个尽量小的模型把 DPO（Direct Preference Optimization）完整跑通的一次记录。目标不是“刷指标”，而是把训练流程跑顺，并能在 TensorBoard 里看到关键曲线（reward margin / reward accuracy / chosen & rejected 的 logprob 等），方便理解 DPO 的行为。

我在AutoDL做的过程里遇到过比较典型的版本依赖坑、数据字段格式坑、以及一些看起来像训练失败的评测误用坑。我把这些都写进这个仓库，方便以后自己和别人复现时少走弯路。

---

## 0. 我实际跑通的最短路线

1. 4090 单卡（24GB）能跑：先用 `train/train_dpo_min.py` 跑 60 steps 验证流程 + TensorBoard 曲线。
2. 跑通后再用 `train/train_dpo_full.py` 跑全量训练（62.1k pairs）。
3. 训练曲线我主要看 5 条：`loss`, `rewards/margins`, `rewards/accuracies`, `logps/chosen`, `logps/rejected`。
4. 我中途遇到的坑都是真实踩过的：transformers/trl/torch/numpy 版本不匹配、数据集字段结构不是字符串、离线评测脚本把 margin 算反、以及显存问题。

---

## 1. 我实际用的实验设置

### 模型
- Base model: `Qwen/Qwen2-0.5B-Instruct`

### 数据集
- HF dataset: `trl-lib/ultrafeedback_binarized`
- train split:  `62135` 行

注意：这个数据集里的 `chosen/rejected` 不是纯字符串，而是 **chat messages list**（每条是 `{role, content}`）。如果你自己写 tokenize，很容易因为格式不对直接报错。TRL 内置的 DPOTrainer 会做 prompt 抽取、套 chat template、再 tokenize。

### 机器
- AutoDL 单卡 RTX 4090（24GB 显存）
- CPU 16 核、内存 120GB
- 数据盘：`/root/autodl-tmp`（HF cache / 输出都放这里）

---

## 2. 目录结构

```
train/
  train_dpo_min.py        # 60 steps 快速跑通，验证能训练+能出曲线
  train_dpo_full.py       # 全量训练脚本（实际跑 full 的用它）
eval/
  trl_metrics_check.py
  trl_metrics_check_ref.py
  trl_margin_sanity_10b.py
  trl_fixed5_check6000.py
  compare_on_train_prompts.py
  analyze_len_bias_two.py
  dump_two_negatives.py
  generate_check.py
requirements/
  requirements_freeze.txt  # 我训练环境的 pip freeze 结果，用于复现
notes/
  file_manifest.txt
```

---

## 3. 国内环境：下载时开“学术加速”，训练时关闭

我在国内训练时：
- **下载模型/数据集**：开学术加速
- **开始训练**：关闭加速

### 开启学术加速（需要访问 HF 时）
```bash
source /etc/network_turbo
```

### 关闭学术加速（不需要访问 HF 时）
```bash
unset http_proxy && unset https_proxy
```

---

## 4. HF Cache & 输出路径

把 cache 放到数据盘，避免系统盘爆掉：

```bash
mkdir -p /root/autodl-tmp/hf_cache/{transformers,datasets}
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

transformers 会提示 `TRANSFORMERS_CACHE` 未来弃用，但不影响使用。

---

## 5. 我最后跑通的一套依赖版本

完整内容在：
- `requirements/requirements_freeze.txt`

建议复现时直接对照 freeze

---

## 6. 第一步：先跑通 60 steps

我先用小步数验证：
- 能正常 tokenize 数据
- 能正常 forward/backward
- TensorBoard 能看到曲线

运行：

```bash
cd /root/dpo-qwen2-0.5b-ultrafeedback
accelerate launch train/train_dpo_min.py
```

成功时，你会在日志里看到类似这些字段（每隔若干 step 打印一次）：

- `loss`
- `rewards/chosen`
- `rewards/rejected`
- `rewards/margins`
- `rewards/accuracies`
- `logps/chosen`
- `logps/rejected`

---

## 7. TensorBoard 可视化

```bash
tensorboard --logdir /root/autodl-tmp/outputs --port 6006 --bind_all
```

我用过的观察点：
- smoothing 我截图时用过 `0.8` 和 `0.6` 两个版本（都能帮助看趋势，但会改变视觉感受，所以我同时留了两张截图）
- 某些图会挤出屏幕，这属于 TensorBoard UI 体验问题，不影响训练

### 曲线截图（请我自己粘贴）
训练中途（smoothing=0.8）：
- \[ ] `docs/img/tb_mid_smooth_0p8.png`

训练完成（smoothing=0.8）：
- \[ ] `docs/img/tb_final_smooth_0p8.png`

训练完成（smoothing=0.6）：
- \[ ] `docs/img/tb_final_smooth_0p6.png`

（我后续会把这些图片放进 `docs/img/` 并在这里用 markdown 引用。）

---

## 8. 第二步：跑全量训练

在 60 steps 跑通后，我跑了 full：

```bash
accelerate launch train/train_dpo_full.py
```

我会定期保存 checkpoint（例如：checkpoint-1000 / 2000 / 3000 / ...），用于中途对比生成效果和离线评测。

---

## 9. 我遇到过的坑

### 9.1 最折磨人的版本依赖问题
我遇到过：
- transformers/trl 的 API 变动导致参数名不一致（例如 `processing_class` vs `tokenizer`）
- 一些组合会触发 import error（例如某些 symbols 在不同 transformers 版本里才存在）
- numpy 2.x 和某些二进制编译包（尤其 torch/torchvision）会提示“不兼容 / 可能崩溃”

我最终以能跑通为准把版本固定下来，并写进 `requirements_freeze.txt`，避免以后复现时再掉坑里。


### 9.2 数据集字段不是字符串：`chosen/rejected` 是 list[dict]
我一开始用的最小脚本会报类似错误：
- `KeyError: 'prompt'`
- `ValueError: chosen should be an str but got <class 'list'>`

后来我改用 TRL 的标准处理流程：从 message list 抽 prompt，套 chat template，再 tokenize，这样不会手写错。


### 9.3 离线评测算出来全是负 margin也许不等于训练失败
我曾经写过一个离线脚本，用 teacher-forced logprob 去算 margin，结果一度全是负的，看起来像训崩了。但后面用 TRL 内部逻辑重新算，就能得到合理的正 margin（至少大部分样本是这样）。

结论：**评测脚本非常容易写错（尤其是 prompt/answer 的拼接、mask、截断、sum vs mean、以及 ref 的对齐方式）**。我把相关脚本留在 `eval/` 里，并尽量用 TRL 的方法去验证。

---

## 10. 评测（我做过的）

我主要做两类：

1) 生成对比：固定 prompts，对 base 和 checkpoint 生成，人工 spot-check。
2) teacher-forced 对比：对一批训练样本计算 chosen vs rejected 的 margin（注意 sum / mean 的差异，注意截断）。

对应脚本在 `eval/` 目录。

---



