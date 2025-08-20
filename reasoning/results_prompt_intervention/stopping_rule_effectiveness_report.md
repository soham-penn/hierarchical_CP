# Stopping Rule Effectiveness Report

---
## RL-Tuned Models

**Average Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 128.19 / 1738.21 | 43.88 / 3931.07 | 431.80 / 773.35 | 138.57 / 104.72 | 77.54 / 240.72 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 228.12 / 2034.60 | 222.51 / 3950.49 | 1851.35 / 3075.80 | 140.35 / 297.25 | 119.46 / 273.72 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 196.89 / 3466.04 | 234.88 / 7573.84 | 742.25 / 1591.30 | 497.55 / 1115.78 | 1332.53 / 1550.53 |
| Skywork/Skywork-OR1-7B | 7B | 534.88 / 4053.98 | 125.01 / 6415.61 | 259.40 / 2277.30 | 0.00 / 85.49 | 99.50 / 58.26 |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00 / 7552.11 | 591.21 / 10112.34 | 5616.75 / 11762.00 | 311.26 / 588.64 | 325.54 / 2234.74 |
| nvidia/AceReason-Nemotron-14B | 14B | 134.09 / 2212.51 | 124.66 / 4083.30 | 954.33 / 1266.22 | 68.91 / 148.39 | 162.35 / 220.44 |
| Qwen/QwQ-32B | 32B | 164.46 / 2414.05 | 98.43 / 4389.52 | 266.95 / 1143.20 | 54.99 / 81.77 | 48.96 / 297.34 |
| Qwen/Qwen3-32B | 32B | 49.30 / 857.97 | 67.54 / 2551.37 | 508.55 / 2059.35 | 71.13 / 101.16 | 0.00 / 30.89 |
| Skywork/Skywork-OR1-32B | 32B | 66.83 / 2720.33 | 723.76 / 3913.61 | 0.00 / 302.95 | 58.13 / 177.67 | 44.01 / 421.67 |
| **Average** |  | 166.97 / 3005.53 | 247.99 / 5213.46 | 1181.26 / 2694.61 | 148.99 / 300.10 | 245.54 / 592.03 |
_Shows the average number of tokens saved by the stopping rule..._

**Average Percentage of Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 2.19% / 39.84% | 1.09% / 58.47% | 4.26% / 14.24% | 4.40% / 3.26% | 2.13% / 3.43% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 3.18% / 47.38% | 4.03% / 57.26% | 16.91% / 44.26% | 3.21% / 7.77% | 2.47% / 6.65% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 3.66% / 62.12% | 3.02% / 68.16% | 4.54% / 16.82% | 9.67% / 18.47% | 16.62% / 21.77% |
| Skywork/Skywork-OR1-7B | 7B | 6.03% / 44.88% | 2.30% / 56.94% | 3.88% / 12.94% | 0.00% / 2.58% | 2.82% / 2.26% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% / 78.38% | 3.78% / 69.86% | 32.96% / 71.32% | 4.95% / 8.88% | 4.84% / 28.01% |
| nvidia/AceReason-Nemotron-14B | 14B | 1.80% / 47.21% | 3.14% / 54.77% | 11.77% / 21.17% | 2.38% / 4.95% | 4.24% / 6.08% |
| Qwen/QwQ-32B | 32B | 2.46% / 49.74% | 2.25% / 53.55% | 3.90% / 24.09% | 1.90% / 3.09% | 0.92% / 6.63% |
| Qwen/Qwen3-32B | 32B | 0.94% / 25.41% | 1.39% / 40.38% | 7.71% / 32.82% | 2.55% / 4.26% | 0.00% / 1.18% |
| Skywork/Skywork-OR1-32B | 32B | 1.35% / 52.23% | 9.83% / 46.00% | 0.00% / 6.31% | 2.26% / 5.39% | 1.42% / 4.37% |
| **Average** |  | 2.40% / 49.69% | 3.43% / 56.16% | 9.55% / 27.11% | 3.48% / 6.52% | 3.94% / 8.93% |
_Shows the average percentage of the reasoning trace that was saved..._

**Early Stopping Rate (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 3.00% / 73.00% | 1.49% / 83.58% | 5.00% / 20.00% | 10.14% / 8.70% | 4.00% / 9.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 5.00% / 94.00% | 8.96% / 83.58% | 20.00% / 70.00% | 5.80% / 14.49% | 5.00% / 16.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 7.00% / 98.00% | 4.48% / 83.58% | 5.00% / 25.00% | 18.84% / 27.54% | 23.00% / 35.00% |
| Skywork/Skywork-OR1-7B | 7B | 9.00% / 73.00% | 4.48% / 74.63% | 5.00% / 15.00% | 0.00% / 8.70% | 8.00% / 9.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% / 100.00% | 4.48% / 88.06% | 40.00% / 85.00% | 7.25% / 11.59% | 8.00% / 41.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 4.50% / 86.50% | 5.97% / 80.60% | 17.50% / 32.50% | 7.25% / 13.04% | 12.00% / 14.00% |
| Qwen/QwQ-32B | 32B | 6.06% / 89.39% | 5.97% / 79.10% | 5.00% / 45.00% | 5.80% / 10.14% | 3.00% / 12.00% |
| Qwen/Qwen3-32B | 32B | 3.00% / 66.00% | 4.48% / 62.69% | 10.00% / 50.00% | 7.25% / 11.59% | 0.00% / 4.00% |
| Skywork/Skywork-OR1-32B | 32B | 3.00% / 86.00% | 16.42% / 59.70% | 0.00% / 10.00% | 5.80% / 21.74% | 3.00% / 10.00% |
| **Average** |  | 4.51% / 85.10% | 6.30% / 77.28% | 11.94% / 39.17% | 7.57% / 14.17% | 7.33% / 16.67% |
_Indicates the percentage of queries where the stopping rule was triggered..._

**Accuracy Dropped (With Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 3.00% | 1.49% | 0.00% | 2.90% | 1.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 3.00% | 8.96% | 5.00% | 4.35% | 2.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 7.00% | 4.48% | 0.00% | 5.80% | 11.00% |
| Skywork/Skywork-OR1-7B | 7B | 7.00% | 4.48% | 0.00% | 0.00% | 4.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% | 2.99% | 10.00% | 5.80% | 4.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 3.50% | 5.97% | 10.00% | 5.80% | 4.00% |
| Qwen/QwQ-32B | 32B | 5.30% | 5.97% | 0.00% | 2.90% | 3.00% |
| Qwen/Qwen3-32B | 32B | 2.00% | 4.48% | 5.00% | 4.35% | 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 3.00% | 16.42% | 0.00% | 2.90% | 2.00% |
| **Average** |  | 3.76% | 6.14% | 3.33% | 3.86% | 3.44% |
_Shows the decrease in accuracy when the stopping rule was applied with context._

**Abstention Increased (Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 24.00% | 62.69% | 15.00% | 8.70% | 9.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 61.00% | 61.19% | 55.00% | 11.59% | 15.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 18.00% | 35.82% | 20.00% | 26.09% | 28.00% |
| Skywork/Skywork-OR1-7B | 7B | 35.00% | 55.22% | 15.00% | 7.25% | 7.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 29.00% | 32.84% | 55.00% | 10.14% | 39.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 32.50% | 53.73% | 25.00% | 13.04% | 12.00% |
| Qwen/QwQ-32B | 32B | 28.79% | 44.78% | 15.00% | 10.14% | 10.00% |
| Qwen/Qwen3-32B | 32B | 5.00% | 14.93% | 25.00% | 10.14% | 2.00% |
| Skywork/Skywork-OR1-32B | 32B | 29.00% | 40.30% | 10.00% | 21.74% | 9.00% |
| **Average** |  | 29.14% | 44.61% | 26.11% | 13.20% | 14.56% |
_Shows the increase in abstention when the stopping rule was applied without context._

---
## Distilled Models

**Average Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 364.14 / 409.19 | 1108.70 / 2149.18 | 89.90 / 372.15 | 0.00 / 0.00 | 15.73 / 0.00 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 165.96 / 898.76 | 246.79 / 2708.85 | 96.30 / 440.40 | 9.39 / 9.04 | 6.84 / 0.66 |
| Qwen/Qwen3-1.7B | 1.7B | 75.86 / 638.17 | 90.03 / 2413.34 | 459.40 / 460.80 | 29.42 / 66.61 | 101.19 / 86.58 |
| microsoft/Phi-4-mini-reasoning | 4B | 10.47 / 997.91 | 105.40 / 2348.27 | 1061.05 / 3075.00 | 78.52 / 130.22 | 234.83 / 106.00 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 39.40 / 1191.46 | 87.12 / 3895.09 | 1106.15 / 1833.10 | 73.75 / 318.39 | 104.68 / 484.97 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 112.03 / 345.77 | 18.06 / 2491.73 | 0.00 / 104.25 | 38.26 / 4.20 | 68.80 / 13.90 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 174.87 / 566.55 | 307.55 / 2425.49 | 29.15 / 269.85 | 15.93 / 1.29 | 0.00 / 0.00 |
| Qwen/Qwen3-8B | 8B | 249.08 / 1017.77 | 113.51 / 2597.10 | 653.10 / 429.25 | 0.00 / 12.55 | 27.63 / 82.02 |
| microsoft/Phi-4-reasoning | 14B | 24.26 / 1870.31 | 147.34 / 3640.64 | 1286.25 / 3131.90 | 205.52 / 213.74 | 326.12 / 544.61 |
| **Average** |  | 135.12 / 881.77 | 247.17 / 2741.08 | 531.26 / 1124.08 | 50.09 / 84.00 | 98.42 / 146.53 |
_Shows the average number of tokens saved by the stopping rule..._

**Average Percentage of Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 7.95% / 14.56% | 10.99% / 38.95% | 3.48% / 8.47% | 0.00% / 0.00% | 0.69% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 3.37% / 22.11% | 4.84% / 44.31% | 2.81% / 11.31% | 0.51% / 0.60% | 0.30% / 0.05% |
| Qwen/Qwen3-1.7B | 1.7B | 2.42% / 23.22% | 2.17% / 44.03% | 7.54% / 11.27% | 1.27% / 3.02% | 3.24% / 4.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.43% / 32.19% | 2.05% / 40.55% | 13.60% / 52.12% | 2.27% / 4.32% | 5.34% / 3.86% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 1.48% / 33.90% | 1.71% / 49.28% | 16.41% / 32.42% | 1.97% / 8.86% | 2.20% / 9.58% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 1.58% / 8.53% | 0.67% / 46.52% | 0.00% / 2.91% | 1.66% / 0.18% | 2.23% / 0.62% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 2.74% / 13.31% | 5.60% / 40.51% | 0.40% / 7.44% | 0.59% / 0.09% | 0.00% / 0.00% |
| Qwen/Qwen3-8B | 8B | 4.99% / 29.98% | 3.14% / 41.32% | 8.12% / 13.14% | 0.00% / 0.74% | 1.19% / 2.60% |
| microsoft/Phi-4-reasoning | 14B | 0.83% / 41.04% | 2.42% / 40.08% | 10.15% / 45.95% | 5.51% / 7.19% | 7.52% / 13.96% |
| **Average** |  | 2.86% / 24.32% | 3.73% / 42.84% | 6.95% / 20.56% | 1.53% / 2.78% | 2.52% / 3.85% |
_Shows the average percentage of the reasoning trace that was saved..._

**Early Stopping Rate (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 15.00% / 43.00% | 16.42% / 74.63% | 10.00% / 15.00% | 0.00% / 0.00% | 2.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 8.00% / 42.00% | 8.96% / 68.66% | 5.00% / 25.00% | 2.90% / 2.90% | 1.00% / 1.00% |
| Qwen/Qwen3-1.7B | 1.7B | 6.00% / 68.00% | 4.48% / 74.63% | 10.00% / 20.00% | 4.35% / 10.14% | 8.00% / 14.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 1.00% / 76.00% | 4.48% / 74.63% | 20.00% / 80.00% | 5.80% / 13.04% | 11.00% / 11.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 3.00% / 74.00% | 2.99% / 73.13% | 25.00% / 55.00% | 4.35% / 23.19% | 5.00% / 20.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 3.00% / 16.00% | 1.49% / 77.61% | 0.00% / 5.00% | 8.70% / 1.45% | 7.00% / 3.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 5.00% / 27.00% | 11.94% / 65.67% | 5.00% / 15.00% | 1.45% / 1.45% | 0.00% / 0.00% |
| Qwen/Qwen3-8B | 8B | 10.00% / 71.00% | 7.46% / 64.18% | 10.00% / 25.00% | 0.00% / 4.35% | 4.00% / 9.00% |
| microsoft/Phi-4-reasoning | 14B | 2.00% / 83.00% | 4.48% / 71.64% | 20.00% / 75.00% | 13.04% / 15.94% | 15.00% / 31.00% |
| **Average** |  | 5.89% / 55.56% | 6.97% / 71.64% | 11.67% / 35.00% | 4.51% / 8.05% | 5.89% / 9.89% |
_Indicates the percentage of queries where the stopping rule was triggered..._

**Accuracy Dropped (With Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 5.00% | 8.96% | 5.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 4.00% | 5.97% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 5.00% | 2.99% | 5.00% | 0.00% | 3.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 1.00% | 4.48% | 0.00% | 1.45% | 9.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 2.00% | 2.99% | 5.00% | 2.90% | 1.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 2.00% | 1.49% | 0.00% | 2.90% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 4.00% | 10.45% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-8B | 8B | 10.00% | 7.46% | 0.00% | 0.00% | 2.00% |
| microsoft/Phi-4-reasoning | 14B | 1.00% | 1.49% | 0.00% | 10.14% | 11.00% |
| **Average** |  | 3.78% | 5.14% | 1.67% | 1.93% | 3.11% |
_Shows the decrease in accuracy when the stopping rule was applied with context._

**Abstention Increased (Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 17.91% | 5.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 17.00% | 50.75% | 20.00% | 2.90% | 1.00% |
| Qwen/Qwen3-1.7B | 1.7B | 1.00% | 11.94% | 0.00% | 10.14% | 10.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 13.00% | 29.85% | 40.00% | 13.04% | 6.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 11.00% | 26.87% | 30.00% | 21.74% | 17.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 4.00% | 49.25% | 5.00% | 1.45% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 7.00% | 49.25% | 15.00% | 1.45% | 0.00% |
| Qwen/Qwen3-8B | 8B | 1.00% | 13.43% | 10.00% | 4.35% | 5.00% |
| microsoft/Phi-4-reasoning | 14B | 34.00% | 29.85% | 45.00% | 14.49% | 27.00% |
| **Average** |  | 9.78% | 31.01% | 18.89% | 7.73% | 7.56% |
_Shows the increase in abstention when the stopping rule was applied without context._