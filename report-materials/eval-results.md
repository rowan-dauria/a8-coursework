

# K=2, trained to 5864 steps

[https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/keh7f5es](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/keh7f5es)
\=========================================================================
fine-tuned LoRA \- run k2\_baseline
\=========================================================================
source jsonl    : /home/funmilooi-somoye/tpu-2026/analysis/k2\_baseline\_lora.jsonl
n\_samples       : 1319
n\_iter          : 10000
seed            : 42

metric   point             95% CI                     boot mean   std err
\-------------------------------------------------------------------------
accuracy  0.08% (0.0008) \[ 0.00%,  0.23%\]              0.08%      0.07%
partial   0.38% (0.0038) \[ 0.08%,  0.76%\]              0.38%      0.17%
format    2.81% (0.0281) \[ 1.90%,  3.71%\]              2.81%      0.46%

# K=8, trained to 5864 steps

[https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/8k-baseline-6516-steps-rd](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/8k-baseline-6516-steps-rd)
\========================================================================
fine-tuned LoRA — run k8-6516-steps
\========================================================================
  source jsonl    : /home/ext\_rowandauria1\_gmail\_com/tpu-2026/analysis/k8-6516-steps\_lora.jsonl
  n\_samples       : 1319
  n\_iter          : 10000
  seed            : 42

  metric   point             95% CI                      boot mean   std err
  \---------------------------------------------------------------------------
  accuracy 56.03% (0.5603) \[53.45%, 58.76%\]            56.05%      1.36%
  partial  58.98% (0.5898) \[56.41%, 61.64%\]            58.99%      1.35%
  format   94.84% (0.9484) \[93.63%, 96.06%\]            94.85%      0.61%

# K=8, new reward, trained to 5864 steps

[https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/k-8-new-reward](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/k-8-new-reward)
\========================================================================
fine-tuned LoRA — run k8-new-reward
\========================================================================
  source jsonl    : /home/ext\_rowandauria1\_gmail\_com/tpu-2026/analysis/k8-new-reward\_lora.jsonl
  n\_samples       : 1319
  n\_iter          : 10000
  seed            : 42

  metric   point             95% CI                      boot mean   std err
  \---------------------------------------------------------------------------
  accuracy 58.53% (0.5853) \[55.88%, 61.18%\]            58.53%      1.35%
  partial  60.80% (0.6080) \[58.15%, 63.46%\]            60.80%      1.34%
  format   94.31% (0.9431) \[93.03%, 95.53%\]            94.32%      0.64%

# K=16, trained for 2000 steps (probs not needed for report?)

[https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/xqbl406c](https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/xqbl406c)
\========================================================================
fine-tuned LoRA — run k16-2000-steps
\========================================================================
  source jsonl    : /home/ext\_rowandauria1\_gmail\_com/tpu-2026/analysis/k16-2000-steps\_lora.jsonl
  n\_samples       : 1319
  n\_iter          : 10000
  seed            : 42

  metric   point             95% CI                      boot mean   std err
  \---------------------------------------------------------------------------
  accuracy 56.56% (0.5656) \[53.90%, 59.29%\]            56.58%      1.38%
  partial  59.29% (0.5929) \[56.63%, 61.94%\]            59.30%      1.36%
  format   94.69% (0.9469) \[93.48%, 95.91%\]            94.70%      0.62%

---

# K sweep evals (64 test pts)

| k=2 | LoRA: correct=2/64 acc=3.12% partial=6.25% format=12.50% | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/jgs4c6kl |
| :---- | :---- | :---- |
| K \= 4 | (evaluate.py) FINAL: correct=35/64 acc=54.69% partial=59.38% format=89.06% | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/x4j7yhdp |
| K=8 | LoRA: correct=38/64 acc=59.38% partial=62.50% format=95.31% | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/m3xp6k97 |
| K \=16 | (2500 steps )FINAL: correct=36/64 acc=56.25% partial=59.38% format=92.19% | https://wandb.ai/felsomoye-university-of-cambridge/tunix/runs/xqbl406c |

