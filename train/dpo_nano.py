import torch
import torch.nn.functional as F

def dpo_nano(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0, reference_free=False):
    # logits = (π(yw)-π(yl)) - (πref(yw)-πref(yl)) ; ref-free 时第二项为 0
    logratios = chosen_logps - rejected_logps
    ref_logratios = torch.zeros_like(logratios) if reference_free else (ref_chosen_logps - ref_rejected_logps)
    logits = logratios - ref_logratios

    # DPO sigmoid loss 
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    loss = losses.mean()

    # rewards = β * (log π(y) - log πref(y))，并 detach，不回传梯度
    chosen_rewards = (beta * (chosen_logps - (0 if reference_free else ref_chosen_logps))).detach()
    rejected_rewards = (beta * (rejected_logps - (0 if reference_free else ref_rejected_logps))).detach()

    metrics = dict(
        rewards_chosen=chosen_rewards.mean().item(),
        rewards_rejected=rejected_rewards.mean().item(),
        rewards_margins=(chosen_rewards - rejected_rewards).mean().item(),
        rewards_accuracies=(chosen_rewards > rejected_rewards).float().mean().item(),
        logps_chosen=chosen_logps.detach().mean().item(),
        logps_rejected=rejected_logps.detach().mean().item(),
    )
    return loss, metrics