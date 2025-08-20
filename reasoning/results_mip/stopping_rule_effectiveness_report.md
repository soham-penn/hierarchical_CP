# Stopping Rule Effectiveness Report

---
## RL-Tuned Models

**Average Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Skywork/Skywork-OR1-7B | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| nvidia/AceReason-Nemotron-14B | 14B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen/QwQ-32B | 32B | 0.00 / 2879.22 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen/Qwen3-32B | 32B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Skywork/Skywork-OR1-32B | 32B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Average** |  | 0.00 / 319.91 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
_Shows the average number of tokens saved by the stopping rule..._

**Average Percentage of Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/QwQ-32B | 32B | 0.00% / 48.54% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-32B | 32B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| **Average** |  | 0.00% / 5.39% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
_Shows the average percentage of the reasoning trace that was saved..._

**Early Stopping Rate (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/QwQ-32B | 32B | 0.00% / 78.26% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-32B | 32B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| **Average** |  | 0.00% / 8.70% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
_Indicates the percentage of queries where the stopping rule was triggered..._

**Accuracy Dropped (With Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/QwQ-32B | 32B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-32B | 32B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Average** |  | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
_Shows the decrease in accuracy when the stopping rule was applied with context._

**Abstention Increased (Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/QwQ-32B | 32B | 14.65% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-32B | 32B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Average** |  | 1.63% | 0.00% | 0.00% | 0.00% | 0.00% |
_Shows the increase in abstention when the stopping rule was applied without context._

---
## Distilled Models

**Average Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen/Qwen3-1.7B | 1.7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Qwen/Qwen3-8B | 8B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| microsoft/Phi-4-reasoning | 14B | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| **Average** |  | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
_Shows the average number of tokens saved by the stopping rule..._

**Average Percentage of Tokens Saved (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-8B | 8B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-reasoning | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| **Average** |  | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
_Shows the average percentage of the reasoning trace that was saved..._

**Early Stopping Rate (With Context / Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| Qwen/Qwen3-8B | 8B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| microsoft/Phi-4-reasoning | 14B | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
| **Average** |  | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% | 0.00% / 0.00% |
_Indicates the percentage of queries where the stopping rule was triggered..._

**Accuracy Dropped (With Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-8B | 8B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-reasoning | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Average** |  | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
_Shows the decrease in accuracy when the stopping rule was applied with context._

**Abstention Increased (Without Context)**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Qwen/Qwen3-8B | 8B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-reasoning | 14B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **Average** |  | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
_Shows the increase in abstention when the stopping rule was applied without context._