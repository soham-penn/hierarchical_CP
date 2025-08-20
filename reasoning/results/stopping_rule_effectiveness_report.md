# Stopping Rule Effectiveness Report

---
## RL-Tuned Models

### Results for LengthStoppingRule
*(Data represents the first available quantile for each model, as quantile information is ignored.)*

#### With Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 39.19 | 93.42 | 0.00 | 19.33 | 27.44 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 5.46 | 0.00 | 353.45 | 90.54 | 144.51 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 102.65 | 35.66 | 0.00 | 27.10 | 26.22 |
| Skywork/Skywork-OR1-7B | 7B | 42.92 | 125.40 | 11.65 | 180.51 | 92.67 |
| microsoft/Phi-4-reasoning-plus | 14B | 238.97 | 311.24 | 6.35 | 16.90 | 90.94 |
| nvidia/AceReason-Nemotron-14B | 14B | 93.57 | 0.00 | 0.00 | 23.55 | 35.81 |
| Qwen/QwQ-32B | 32B | 247.23 | 10.93 | 0.00 | 3.01 | 152.63 |
| Qwen/Qwen3-32B | 32B | 86.70 | 173.48 | 0.00 | 25.78 | 36.02 |
| Skywork/Skywork-OR1-32B | 32B | 61.37 | 25.54 | 0.00 | 8.17 | 42.66 |
| **Average** |  | 102.01 | 86.18 | 41.27 | 43.88 | 72.10 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.33% | 0.91% | 0.00% | 0.30% | 0.32% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.05% | 0.00% | 1.08% | 1.07% | 1.27% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.62% | 0.17% | 0.00% | 0.24% | 0.16% |
| Skywork/Skywork-OR1-7B | 7B | 0.36% | 0.68% | 0.05% | 1.57% | 0.74% |
| microsoft/Phi-4-reasoning-plus | 14B | 1.95% | 1.05% | 0.02% | 0.15% | 0.47% |
| nvidia/AceReason-Nemotron-14B | 14B | 0.80% | 0.00% | 0.00% | 0.40% | 0.36% |
| Qwen/QwQ-32B | 32B | 2.76% | 0.08% | 0.00% | 0.08% | 1.90% |
| Qwen/Qwen3-32B | 32B | 1.19% | 0.96% | 0.00% | 0.76% | 0.75% |
| Skywork/Skywork-OR1-32B | 32B | 0.66% | 0.21% | 0.00% | 0.16% | 0.50% |
| **Average** |  | 0.97% | 0.45% | 0.13% | 0.53% | 0.72% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 3.00% | 5.97% | 0.00% | 2.90% | 5.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 1.00% | 0.00% | 5.00% | 4.35% | 9.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 2.00% | 1.49% | 0.00% | 1.45% | 2.00% |
| Skywork/Skywork-OR1-7B | 7B | 3.00% | 2.99% | 5.00% | 8.70% | 3.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 7.00% | 2.99% | 10.00% | 1.45% | 3.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 5.00% | 0.00% | 0.00% | 1.45% | 2.00% |
| Qwen/QwQ-32B | 32B | 6.00% | 1.49% | 0.00% | 1.45% | 9.00% |
| Qwen/Qwen3-32B | 32B | 3.00% | 2.99% | 0.00% | 4.35% | 3.00% |
| Skywork/Skywork-OR1-32B | 32B | 5.00% | 1.49% | 0.00% | 1.45% | 3.00% |
| **Average** |  | 3.89% | 2.16% | 2.22% | 3.06% | 4.33% |

**Accuracy Dropped**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 1.00% | 5.97% | 0.00% | 1.45% | 0.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 2.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 3.00% | 2.99% | 0.00% | 0.00% | 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 3.00% | 1.49% | 0.00% | 1.45% | 3.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 4.00% | 0.00% | 0.00% | 1.45% | 1.00% |
| Qwen/QwQ-32B | 32B | 4.00% | 1.49% | 0.00% | 0.00% | 3.00% |
| Qwen/Qwen3-32B | 32B | 1.00% | 1.49% | 0.00% | 2.90% | 0.00% |
| Skywork/Skywork-OR1-32B | 32B | 3.00% | 1.49% | 0.00% | 0.00% | 0.00% |
| **Average** |  | 2.11% | 1.66% | 0.00% | 0.81% | 1.00% |

#### Without Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00 | 287.19 | 0.00 | 60.23 | 18.91 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 24.67 | 528.24 | 0.00 | 71.16 | 202.70 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 6.07 | 252.69 | 0.00 | 3.36 | 36.61 |
| Skywork/Skywork-OR1-7B | 7B | 1051.49 | 656.12 | 0.00 | 0.00 | 13.48 |
| microsoft/Phi-4-reasoning-plus | 14B | 2610.70 | 2011.69 | 5.50 | 55.64 | 611.66 |
| nvidia/AceReason-Nemotron-14B | 14B | 163.27 | 301.13 | 0.00 | 43.57 | 26.21 |
| Qwen/QwQ-32B | 32B | 610.68 | 404.40 | 0.00 | 60.71 | 181.42 |
| Qwen/Qwen3-32B | 32B | 123.49 | 542.73 | 0.00 | 53.29 | 77.47 |
| Skywork/Skywork-OR1-32B | 32B | 97.38 | 263.94 | 0.00 | 0.00 | 3.94 |
| **Average** |  | 520.86 | 583.13 | 0.61 | 38.66 | 130.27 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% | 2.49% | 0.00% | 0.91% | 0.19% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 0.20% | 2.50% | 0.00% | 0.85% | 0.95% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.06% | 1.14% | 0.00% | 0.04% | 0.20% |
| Skywork/Skywork-OR1-7B | 7B | 3.59% | 2.81% | 0.00% | 0.00% | 0.15% |
| microsoft/Phi-4-reasoning-plus | 14B | 17.21% | 6.76% | 0.02% | 0.47% | 2.60% |
| nvidia/AceReason-Nemotron-14B | 14B | 1.60% | 1.74% | 0.00% | 0.81% | 0.32% |
| Qwen/QwQ-32B | 32B | 8.39% | 2.11% | 0.00% | 1.33% | 2.52% |
| Qwen/Qwen3-32B | 32B | 2.14% | 3.68% | 0.00% | 1.51% | 1.68% |
| Skywork/Skywork-OR1-32B | 32B | 1.07% | 1.79% | 0.00% | 0.00% | 0.06% |
| **Average** |  | 3.81% | 2.78% | 0.00% | 0.66% | 0.96% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% | 14.93% | 0.00% | 7.25% | 1.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 2.00% | 8.96% | 0.00% | 5.80% | 2.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 2.00% | 10.45% | 0.00% | 1.45% | 1.00% |
| Skywork/Skywork-OR1-7B | 7B | 7.00% | 13.43% | 0.00% | 0.00% | 2.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 51.00% | 25.37% | 10.00% | 7.25% | 9.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 9.00% | 10.45% | 0.00% | 8.70% | 3.00% |
| Qwen/QwQ-32B | 32B | 32.00% | 13.43% | 0.00% | 7.25% | 13.00% |
| Qwen/Qwen3-32B | 32B | 11.00% | 16.42% | 0.00% | 8.70% | 8.00% |
| Skywork/Skywork-OR1-32B | 32B | 10.00% | 11.94% | 0.00% | 0.00% | 2.00% |
| **Average** |  | 13.78% | 13.93% | 1.11% | 5.15% | 4.56% |

**Abstention Increased**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 0.00% | 11.94% | 0.00% | 7.25% | 1.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 1.00% | 8.96% | 0.00% | 4.35% | 1.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 0.00% | 8.96% | 0.00% | 1.45% | 0.00% |
| Skywork/Skywork-OR1-7B | 7B | 5.00% | 7.46% | 0.00% | 0.00% | 1.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 22.00% | 17.91% | 10.00% | 7.25% | 8.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 3.00% | 5.97% | 0.00% | 8.70% | 3.00% |
| Qwen/QwQ-32B | 32B | 4.00% | 10.45% | 0.00% | 7.25% | 11.00% |
| Qwen/Qwen3-32B | 32B | 2.00% | 1.49% | 0.00% | 7.25% | 5.00% |
| Skywork/Skywork-OR1-32B | 32B | 1.00% | 7.46% | 0.00% | 0.00% | 2.00% |
| **Average** |  | 4.22% | 8.96% | 1.11% | 4.83% | 3.56% |

### Results for UncertaintyStoppingRule
*(Data represents the first available quantile for each model, as quantile information is ignored.)*

#### With Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 113.03 | 119.52 | 53.70 | 167.46 | 152.56 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 96.03 | 63.28 | 730.55 | 2.72 | 193.61 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 137.58 | 444.61 | 1431.50 | 530.36 | 413.80 |
| Skywork/Skywork-OR1-7B | 7B | 169.43 | 132.51 | 434.30 | 118.77 | 17.86 |
| microsoft/Phi-4-reasoning-plus | 14B | 335.28 | 991.07 | 3297.20 | 543.91 | 338.83 |
| nvidia/AceReason-Nemotron-14B | 14B | 446.42 | 285.43 | 139.35 | 37.26 | 89.66 |
| Qwen/QwQ-32B | 32B | 140.76 | 217.94 | 351.60 | 5.25 | 192.34 |
| Qwen/Qwen3-32B | 32B | 26.41 | 114.60 | 0.00 | 66.23 | 45.32 |
| Skywork/Skywork-OR1-32B | 32B | 90.37 | 322.84 | 806.65 | 104.23 | 84.05 |
| **Average** |  | 172.81 | 299.09 | 804.98 | 175.13 | 169.78 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 1.58% | 2.99% | 0.86% | 3.58% | 2.73% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 1.99% | 1.59% | 8.17% | 0.05% | 3.48% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 2.38% | 5.93% | 9.02% | 10.06% | 5.54% |
| Skywork/Skywork-OR1-7B | 7B | 2.78% | 2.61% | 4.31% | 3.41% | 0.62% |
| microsoft/Phi-4-reasoning-plus | 14B | 3.25% | 8.99% | 15.26% | 9.98% | 3.84% |
| nvidia/AceReason-Nemotron-14B | 14B | 7.08% | 5.38% | 3.25% | 1.21% | 1.79% |
| Qwen/QwQ-32B | 32B | 2.78% | 3.10% | 4.12% | 0.38% | 3.89% |
| Qwen/Qwen3-32B | 32B | 0.71% | 1.97% | 0.00% | 2.11% | 1.04% |
| Skywork/Skywork-OR1-32B | 32B | 1.04% | 5.86% | 4.57% | 3.14% | 1.91% |
| **Average** |  | 2.62% | 4.27% | 5.51% | 3.77% | 2.76% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 3.00% | 5.97% | 5.00% | 8.70% | 9.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 4.00% | 2.99% | 10.00% | 1.45% | 10.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 4.00% | 8.96% | 10.00% | 15.94% | 8.00% |
| Skywork/Skywork-OR1-7B | 7B | 5.00% | 4.48% | 5.00% | 11.59% | 2.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 5.00% | 11.94% | 20.00% | 15.94% | 7.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 12.00% | 10.45% | 5.00% | 4.35% | 3.00% |
| Qwen/QwQ-32B | 32B | 6.00% | 7.46% | 5.00% | 2.90% | 8.00% |
| Qwen/Qwen3-32B | 32B | 1.00% | 4.48% | 0.00% | 4.35% | 4.00% |
| Skywork/Skywork-OR1-32B | 32B | 3.00% | 11.94% | 5.00% | 8.70% | 7.00% |
| **Average** |  | 4.78% | 7.63% | 7.22% | 8.21% | 6.44% |

**Accuracy Dropped**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 2.00% | 5.97% | 5.00% | 2.90% | 1.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 3.00% | 2.99% | 5.00% | 0.00% | 3.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 4.00% | 8.96% | 5.00% | 5.80% | 6.00% |
| Skywork/Skywork-OR1-7B | 7B | 4.00% | 4.48% | 5.00% | 1.45% | 0.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 3.00% | 2.99% | 5.00% | 10.14% | 7.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 11.00% | 10.45% | 5.00% | 1.45% | 1.00% |
| Qwen/QwQ-32B | 32B | 5.00% | 7.46% | 5.00% | 1.45% | 5.00% |
| Qwen/Qwen3-32B | 32B | 1.00% | 4.48% | 0.00% | 2.90% | 3.00% |
| Skywork/Skywork-OR1-32B | 32B | 2.00% | 11.94% | 5.00% | 4.35% | 6.00% |
| **Average** |  | 3.89% | 6.63% | 4.44% | 3.38% | 3.56% |

#### Without Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 1418.09 | 4111.70 | 205.65 | 280.45 | 177.11 |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 2376.30 | 4271.97 | 3826.65 | 136.55 | 252.23 |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 3109.18 | 8596.24 | 5569.70 | 1023.25 | 1107.11 |
| Skywork/Skywork-OR1-7B | 7B | 3668.76 | 6760.70 | 1502.20 | 239.41 | 121.40 |
| microsoft/Phi-4-reasoning-plus | 14B | 7511.59 | 11318.21 | 11140.05 | 1676.22 | 2294.63 |
| nvidia/AceReason-Nemotron-14B | 14B | 2311.58 | 4739.39 | 1754.00 | 203.52 | 297.04 |
| Qwen/QwQ-32B | 32B | 2218.16 | 6318.03 | 4363.80 | 266.28 | 559.11 |
| Qwen/Qwen3-32B | 32B | 875.52 | 4016.87 | 966.85 | 141.57 | 169.91 |
| Skywork/Skywork-OR1-32B | 32B | 2670.69 | 4290.54 | 783.85 | 140.87 | 222.53 |
| **Average** |  | 2906.65 | 6047.07 | 3345.86 | 456.46 | 577.90 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 35.96% | 61.66% | 3.94% | 6.58% | 3.97% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 52.38% | 57.90% | 41.28% | 3.82% | 4.62% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 60.76% | 73.74% | 41.95% | 16.44% | 16.52% |
| Skywork/Skywork-OR1-7B | 7B | 45.17% | 63.88% | 15.67% | 6.60% | 3.26% |
| microsoft/Phi-4-reasoning-plus | 14B | 78.14% | 74.77% | 72.17% | 24.32% | 27.30% |
| nvidia/AceReason-Nemotron-14B | 14B | 46.02% | 61.24% | 19.61% | 5.63% | 6.22% |
| Qwen/QwQ-32B | 32B | 46.97% | 64.91% | 43.47% | 7.43% | 12.78% |
| Qwen/Qwen3-32B | 32B | 25.69% | 54.97% | 14.65% | 5.45% | 5.82% |
| Skywork/Skywork-OR1-32B | 32B | 52.44% | 52.00% | 14.61% | 4.96% | 6.27% |
| **Average** |  | 49.28% | 62.79% | 29.71% | 9.02% | 9.64% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 77.00% | 86.57% | 10.00% | 20.29% | 8.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 98.00% | 83.58% | 55.00% | 7.25% | 9.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 98.00% | 89.55% | 50.00% | 28.99% | 27.00% |
| Skywork/Skywork-OR1-7B | 7B | 72.00% | 83.58% | 20.00% | 15.94% | 8.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 100.00% | 88.06% | 85.00% | 39.13% | 39.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 85.00% | 85.07% | 25.00% | 17.39% | 12.00% |
| Qwen/QwQ-32B | 32B | 89.00% | 85.07% | 55.00% | 13.04% | 27.00% |
| Qwen/Qwen3-32B | 32B | 61.00% | 83.58% | 25.00% | 20.29% | 14.00% |
| Skywork/Skywork-OR1-32B | 32B | 91.00% | 68.66% | 20.00% | 17.39% | 17.00% |
| **Average** |  | 85.67% | 83.75% | 38.33% | 19.97% | 17.89% |

**Abstention Increased**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| nvidia/AceReason-Nemotron-7B | 7B | 39.00% | 71.64% | 5.00% | 17.39% | 7.00% |
| nvidia/AceReason-Nemotron-1.1-7B | 7B | 60.00% | 55.22% | 45.00% | 5.80% | 8.00% |
| XiaomiMiMo/MiMo-7B-RL-0530 | 7B | 22.00% | 52.24% | 30.00% | 24.64% | 20.00% |
| Skywork/Skywork-OR1-7B | 7B | 39.00% | 64.18% | 20.00% | 14.49% | 7.00% |
| microsoft/Phi-4-reasoning-plus | 14B | 32.00% | 58.21% | 75.00% | 37.68% | 37.00% |
| nvidia/AceReason-Nemotron-14B | 14B | 30.00% | 61.19% | 20.00% | 14.49% | 11.00% |
| Qwen/QwQ-32B | 32B | 17.00% | 59.70% | 40.00% | 11.59% | 21.00% |
| Qwen/Qwen3-32B | 32B | 3.00% | 20.90% | 15.00% | 11.59% | 10.00% |
| Skywork/Skywork-OR1-32B | 32B | 36.00% | 47.76% | 15.00% | 14.49% | 14.00% |
| **Average** |  | 30.89% | 54.56% | 29.44% | 16.91% | 15.00% |

---
## Distilled Models

### Results for LengthStoppingRule
*(Data represents the first available quantile for each model, as quantile information is ignored.)*

#### With Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 86.65 | 72.39 | 0.00 | 4.28 | 29.29 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 9.04 | 19.40 | 0.00 | 25.97 | 89.86 |
| Qwen/Qwen3-1.7B | 1.7B | 117.89 | 37.72 | 586.65 | 10.99 | 47.82 |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00 | 244.03 | 411.25 | 12.74 | 137.73 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 231.55 | 0.00 | 0.00 | 161.77 | 248.49 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 38.76 | 46.34 | 0.00 | 22.84 | 22.93 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 53.65 | 65.81 | 19.65 | 0.00 | 23.02 |
| Qwen/Qwen3-8B | 8B | 81.38 | 36.96 | 0.00 | 0.00 | 112.37 |
| microsoft/Phi-4-reasoning | 14B | 86.30 | 252.05 | 0.00 | 101.71 | 161.20 |
| **Average** |  | 78.36 | 86.08 | 113.06 | 37.81 | 96.97 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.74% | 0.45% | 0.00% | 0.16% | 0.93% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.11% | 0.15% | 0.00% | 1.10% | 1.46% |
| Qwen/Qwen3-1.7B | 1.7B | 1.47% | 0.22% | 2.75% | 0.32% | 0.68% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 1.53% | 1.93% | 0.22% | 1.55% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 1.18% | 0.00% | 0.00% | 2.83% | 2.92% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.43% | 0.42% | 0.00% | 0.56% | 0.33% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.59% | 0.52% | 0.15% | 0.00% | 0.40% |
| Qwen/Qwen3-8B | 8B | 0.85% | 0.28% | 0.00% | 0.00% | 1.60% |
| microsoft/Phi-4-reasoning | 14B | 2.04% | 2.41% | 0.00% | 1.97% | 1.53% |
| **Average** |  | 0.82% | 0.66% | 0.54% | 0.80% | 1.27% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 3.00% | 2.99% | 0.00% | 2.90% | 5.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 2.00% | 1.49% | 0.00% | 7.25% | 4.00% |
| Qwen/Qwen3-1.7B | 1.7B | 7.00% | 1.49% | 10.00% | 2.90% | 4.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 4.48% | 10.00% | 2.90% | 7.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 5.00% | 0.00% | 0.00% | 5.80% | 9.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 2.00% | 2.99% | 0.00% | 4.35% | 3.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 3.00% | 4.48% | 5.00% | 0.00% | 4.00% |
| Qwen/Qwen3-8B | 8B | 4.00% | 2.99% | 0.00% | 0.00% | 8.00% |
| microsoft/Phi-4-reasoning | 14B | 7.00% | 6.06% | 0.00% | 10.14% | 5.00% |
| **Average** |  | 3.67% | 3.00% | 2.78% | 4.03% | 5.44% |

**Accuracy Dropped**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 1.00% | 1.49% | 0.00% | 1.45% | 1.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 1.45% | 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 2.00% | 1.49% | 5.00% | 0.00% | 0.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 1.49% | 10.00% | 0.00% | 4.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 3.00% | 0.00% | 0.00% | 1.45% | 3.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 1.00% | 2.99% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 1.00% | 1.49% | 0.00% | 0.00% | 1.00% |
| Qwen/Qwen3-8B | 8B | 2.00% | 2.99% | 0.00% | 0.00% | 3.00% |
| microsoft/Phi-4-reasoning | 14B | 4.00% | 4.55% | 0.00% | 10.14% | 2.00% |
| **Average** |  | 1.56% | 1.83% | 1.67% | 1.61% | 1.56% |

#### Without Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00 | 9.25 | 0.00 | 0.00 | 50.42 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00 | 0.00 | 0.00 | 7.22 | 7.27 |
| Qwen/Qwen3-1.7B | 1.7B | 14.15 | 0.00 | 0.00 | 39.39 | 30.90 |
| microsoft/Phi-4-mini-reasoning | 4B | 33.82 | 132.72 | 0.00 | 0.00 | 42.48 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 163.02 | 195.47 | 0.00 | 376.26 | 220.97 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00 | 259.40 | 0.00 | 16.09 | 38.77 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00 | 68.10 | 0.00 | 9.62 | 36.10 |
| Qwen/Qwen3-8B | 8B | 7.73 | 277.58 | 0.00 | 3.07 | 81.62 |
| microsoft/Phi-4-reasoning | 14B | 1447.71 | 3384.52 | 0.00 | 104.03 | 202.41 |
| **Average** |  | 185.16 | 480.78 | 0.00 | 61.74 | 78.99 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 0.06% | 0.00% | 0.00% | 1.16% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 0.36% | 0.19% |
| Qwen/Qwen3-1.7B | 1.7B | 0.20% | 0.00% | 0.00% | 1.00% | 0.38% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.46% | 0.92% | 0.00% | 0.00% | 0.53% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 2.66% | 1.10% | 0.00% | 7.41% | 3.26% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% | 1.99% | 0.00% | 0.41% | 0.51% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% | 0.52% | 0.00% | 0.27% | 0.56% |
| Qwen/Qwen3-8B | 8B | 0.12% | 1.70% | 0.00% | 0.07% | 1.29% |
| microsoft/Phi-4-reasoning | 14B | 25.13% | 23.47% | 0.00% | 2.18% | 2.67% |
| **Average** |  | 3.17% | 3.31% | 0.00% | 1.30% | 1.17% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 1.49% | 0.00% | 0.00% | 3.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 4.35% | 1.00% |
| Qwen/Qwen3-1.7B | 1.7B | 1.00% | 0.00% | 0.00% | 4.35% | 2.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 3.00% | 4.48% | 0.00% | 0.00% | 3.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 13.00% | 6.25% | 0.00% | 20.29% | 14.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% | 8.96% | 0.00% | 4.35% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% | 4.48% | 0.00% | 2.90% | 3.00% |
| Qwen/Qwen3-8B | 8B | 3.00% | 10.45% | 0.00% | 1.45% | 7.00% |
| microsoft/Phi-4-reasoning | 14B | 61.00% | 46.97% | 0.00% | 13.04% | 10.00% |
| **Average** |  | 9.00% | 9.23% | 0.00% | 5.64% | 5.00% |

**Abstention Increased**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 0.00% | 1.49% | 0.00% | 0.00% | 3.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 0.00% | 0.00% | 0.00% | 4.35% | 1.00% |
| Qwen/Qwen3-1.7B | 1.7B | 0.00% | 0.00% | 0.00% | 2.90% | 1.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.00% | 0.00% | 0.00% | 0.00% | 2.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 3.00% | 3.12% | 0.00% | 15.94% | 14.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 0.00% | 7.46% | 0.00% | 4.35% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 0.00% | 2.99% | 0.00% | 2.90% | 3.00% |
| Qwen/Qwen3-8B | 8B | 0.00% | 4.48% | 0.00% | 0.00% | 4.00% |
| microsoft/Phi-4-reasoning | 14B | 20.00% | 36.36% | 0.00% | 13.04% | 8.00% |
| **Average** |  | 2.56% | 6.21% | 0.00% | 4.83% | 4.22% |

### Results for UncertaintyStoppingRule
*(Data represents the first available quantile for each model, as quantile information is ignored.)*

#### With Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 87.72 | 207.36 | 445.95 | 0.00 | 4.66 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 121.77 | 91.16 | 0.00 | 22.46 | 4.28 |
| Qwen/Qwen3-1.7B | 1.7B | 56.07 | 42.93 | 1897.90 | 12.61 | 26.14 |
| microsoft/Phi-4-mini-reasoning | 4B | 7.44 | 22.06 | 367.30 | 15.61 | 81.02 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 22.79 | 49.88 | 1517.25 | 53.23 | 52.32 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 122.03 | 161.81 | 0.00 | 58.33 | 89.02 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 198.84 | 516.39 | 338.05 | 18.46 | 23.30 |
| Qwen/Qwen3-8B | 8B | 264.11 | 24.97 | 0.00 | 12.09 | 79.49 |
| microsoft/Phi-4-reasoning | 14B | 80.90 | 278.32 | 0.00 | 89.68 | 0.00 |
| **Average** |  | 106.85 | 154.98 | 507.38 | 31.39 | 40.03 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 1.37% | 4.10% | 9.80% | 0.00% | 0.24% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 2.06% | 1.59% | 0.00% | 1.14% | 0.29% |
| Qwen/Qwen3-1.7B | 1.7B | 1.00% | 0.88% | 19.91% | 0.47% | 0.83% |
| microsoft/Phi-4-mini-reasoning | 4B | 0.33% | 0.53% | 4.15% | 0.49% | 1.79% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 0.74% | 1.16% | 15.85% | 1.45% | 0.79% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 1.89% | 4.14% | 0.00% | 1.73% | 2.59% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 3.88% | 7.28% | 4.09% | 0.67% | 0.99% |
| Qwen/Qwen3-8B | 8B | 5.39% | 0.45% | 0.00% | 0.50% | 2.29% |
| microsoft/Phi-4-reasoning | 14B | 1.99% | 2.44% | 0.00% | 1.97% | 0.00% |
| **Average** |  | 2.07% | 2.51% | 5.98% | 0.94% | 1.09% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 3.00% | 5.97% | 20.00% | 0.00% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 4.00% | 2.99% | 0.00% | 4.35% | 2.00% |
| Qwen/Qwen3-1.7B | 1.7B | 3.00% | 1.49% | 30.00% | 1.45% | 2.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 1.00% | 1.49% | 5.00% | 1.45% | 7.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 2.00% | 6.25% | 25.00% | 4.35% | 1.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 4.00% | 10.45% | 0.00% | 5.80% | 7.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 6.00% | 16.42% | 5.00% | 1.45% | 5.00% |
| Qwen/Qwen3-8B | 8B | 12.00% | 2.99% | 0.00% | 2.90% | 6.00% |
| microsoft/Phi-4-reasoning | 14B | 5.00% | 4.55% | 0.00% | 2.90% | 0.00% |
| **Average** |  | 4.44% | 5.84% | 9.44% | 2.74% | 3.56% |

**Accuracy Dropped**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 1.00% | 4.48% | 0.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 3.00% | 1.49% | 0.00% | 1.45% | 0.00% |
| Qwen/Qwen3-1.7B | 1.7B | 2.00% | 1.49% | 5.00% | 0.00% | 1.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 1.00% | 1.49% | 5.00% | 1.45% | 2.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 1.00% | 6.25% | 10.00% | 0.00% | 0.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 3.00% | 10.45% | 0.00% | 1.45% | 2.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 4.00% | 13.43% | 5.00% | 1.45% | 3.00% |
| Qwen/Qwen3-8B | 8B | 12.00% | 2.99% | 0.00% | 2.90% | 2.00% |
| microsoft/Phi-4-reasoning | 14B | 4.00% | 1.52% | 0.00% | 2.90% | 0.00% |
| **Average** |  | 3.44% | 4.84% | 2.78% | 1.29% | 1.11% |

#### Without Context

**Average Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 432.95 | 2373.61 | 958.25 | 0.00 | 13.75 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 787.66 | 2856.81 | 889.65 | 17.81 | 27.70 |
| Qwen/Qwen3-1.7B | 1.7B | 610.97 | 3233.01 | 2106.15 | 19.17 | 101.21 |
| microsoft/Phi-4-mini-reasoning | 4B | 1044.04 | 2356.52 | 1520.20 | 59.33 | 200.83 |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 1192.46 | 3629.78 | 4178.80 | 549.45 | 359.17 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 146.90 | 3055.42 | 569.55 | 12.35 | 14.81 |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 581.40 | 3651.48 | 463.45 | 28.75 | 43.77 |
| Qwen/Qwen3-8B | 8B | 931.93 | 3653.96 | 1644.25 | 165.70 | 363.18 |
| microsoft/Phi-4-reasoning | 14B | 1920.83 | 5488.47 | 3862.95 | 341.20 | 557.67 |
| **Average** |  | 849.90 | 3366.56 | 1799.25 | 132.64 | 186.90 |

**Average Percentage of Tokens Saved**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 15.57% | 41.97% | 22.62% | 0.00% | 0.86% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 21.53% | 50.82% | 20.28% | 0.91% | 0.97% |
| Qwen/Qwen3-1.7B | 1.7B | 21.64% | 51.57% | 40.74% | 0.85% | 4.07% |
| microsoft/Phi-4-mini-reasoning | 4B | 32.06% | 45.52% | 25.95% | 2.02% | 5.88% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 33.93% | 51.07% | 48.90% | 13.13% | 6.99% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 3.93% | 51.42% | 11.58% | 0.63% | 0.68% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 14.97% | 57.80% | 7.73% | 1.14% | 1.58% |
| Qwen/Qwen3-8B | 8B | 28.51% | 53.64% | 29.86% | 6.33% | 11.36% |
| microsoft/Phi-4-reasoning | 14B | 40.72% | 56.45% | 47.61% | 10.39% | 13.09% |
| **Average** |  | 23.65% | 51.14% | 28.36% | 3.93% | 5.05% |

**Early Stopping Rate**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 48.00% | 73.13% | 55.00% | 0.00% | 4.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 42.00% | 79.10% | 40.00% | 4.35% | 5.00% |
| Qwen/Qwen3-1.7B | 1.7B | 72.00% | 83.58% | 75.00% | 4.35% | 14.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 74.00% | 76.12% | 40.00% | 7.25% | 14.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 74.00% | 75.00% | 65.00% | 28.99% | 15.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 8.00% | 83.58% | 20.00% | 1.45% | 4.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 31.00% | 88.06% | 15.00% | 4.35% | 6.00% |
| Qwen/Qwen3-8B | 8B | 72.00% | 85.07% | 50.00% | 24.64% | 30.00% |
| microsoft/Phi-4-reasoning | 14B | 87.00% | 84.85% | 80.00% | 23.19% | 23.00% |
| **Average** |  | 56.44% | 80.95% | 48.89% | 10.95% | 12.78% |

**Abstention Increased**
|  |  | Math | Math | Science | Medical | Medical |
|---|---|---|---|---|---|---|
| Model | Size | gsm8k | mmlu | gpqa | icraft | imedqa |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | 0.6B | 4.00% | 11.94% | 25.00% | 0.00% | 4.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 9.00% | 64.18% | 35.00% | 4.35% | 5.00% |
| Qwen/Qwen3-1.7B | 1.7B | 2.00% | 19.40% | 25.00% | 2.90% | 9.00% |
| microsoft/Phi-4-mini-reasoning | 4B | 13.00% | 43.28% | 30.00% | 5.80% | 9.00% |
| microsoft/Phi-4-mini-flash-reasoning | 4B | 11.00% | 37.50% | 45.00% | 23.19% | 12.00% |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B | 1.00% | 56.72% | 15.00% | 1.45% | 4.00% |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | 8B | 12.00% | 65.67% | 10.00% | 4.35% | 6.00% |
| Qwen/Qwen3-8B | 8B | 1.00% | 29.85% | 20.00% | 20.29% | 21.00% |
| microsoft/Phi-4-reasoning | 14B | 31.00% | 59.09% | 60.00% | 23.19% | 21.00% |
| **Average** |  | 9.33% | 43.07% | 29.44% | 9.50% | 10.11% |

---
## Final Comparison: LengthStoppingRule vs. UncertaintyStoppingRule
*(Comparing the aggregated averages across all datasets for each model group.)*

### RL-Tuned Models Aggregated Averages
| Metric | LengthStoppingRule | UncertaintyStoppingRule |
|---|---|---|
| Avg. Tokens Saved (W/ Context) | 69.09 | 324.36 |
| Avg. % Saved (W/ Context) | 0.56% | 3.79% |
| Early Stopping Rate (W/ Context) | 3.13% | 6.86% |
| Accuracy Dropped (W/ Context) | 1.11% | 4.38% |
| Metric | LengthStoppingRule | UncertaintyStoppingRule |
|---|---|---|
| Avg. Tokens Saved (W/o Context) | 254.71 | 2666.79 |
| Avg. % Saved (W/o Context) | 1.64% | 32.09% |
| Early Stopping Rate (W/o Context) | 7.71% | 49.12% |
| Abstention Increased (W/o Context) | 4.54% | 29.36% |

### Distilled Models Aggregated Averages
| Metric | LengthStoppingRule | UncertaintyStoppingRule |
|---|---|---|
| Avg. Tokens Saved (W/ Context) | 82.45 | 168.13 |
| Avg. % Saved (W/ Context) | 0.82% | 2.52% |
| Early Stopping Rate (W/ Context) | 3.78% | 5.20% |
| Accuracy Dropped (W/ Context) | 1.64% | 2.69% |
| Metric | LengthStoppingRule | UncertaintyStoppingRule |
|---|---|---|
| Avg. Tokens Saved (W/o Context) | 161.34 | 1267.05 |
| Avg. % Saved (W/o Context) | 1.79% | 22.43% |
| Early Stopping Rate (W/o Context) | 5.77% | 42.00% |
| Abstention Increased (W/o Context) | 3.56% | 20.29% |