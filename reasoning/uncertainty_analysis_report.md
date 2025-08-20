# Uncertainty Analysis Report

**Significance Level (alpha):** 0.1

**Legend:**
- ✅ = Significant difference (p-value < alpha)
- 0️⃣ = t-statistic ≤ 0

**Success Rate Definitions:**
- **✅ Success Rate:** Proportion of checkmarks only (significant results)
- **✅+0️⃣ Success Rate:** Proportion of checkmarks plus zeros (significant results + non-positive t-statistics)

# Results Grouped by Category

## Category: `acknowledging_potential_error`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            | ✅                 |                  | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| gsm8k     | 0️⃣            | ✅                 | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| icraft    |                |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| imedqa    |                | 0️⃣               |                  | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| mmlu      | ✅              | ✅                 | 0️⃣              | ✅                                          |                                           | 0️⃣                         |

## Category: `contrasting_possibilities`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | ✅              | 0️⃣               | 0️⃣              |                                            |                                           | 0️⃣                         |
| gsm8k     | 0️⃣            | ✅                 |                  |                                            | ✅                                         | ✅                           |
| icraft    | ✅              | ✅                 | ✅                | 0️⃣                                        | ✅                                         | 0️⃣                         |
| imedqa    | 0️⃣            | 0️⃣               |                  | 0️⃣                                        | 0️⃣                                       |                             |
| mmlu      | 0️⃣            | ✅                 | 0️⃣              | 0️⃣                                        | ✅                                         | ✅                           |

## Category: `doubt_speculation`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        |                                           | ✅                           |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          |                                           | ✅                           |
| icraft    | ✅              | 0️⃣               | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| imedqa    | ✅              | 0️⃣               | ✅                | 0️⃣                                        |                                           | ✅                           |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

## Category: `hesitation_processing_pause`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | 0️⃣                         |
| icraft    | ✅              |                   |                  | 0️⃣                                        | 0️⃣                                       |                             |
| imedqa    | 0️⃣            | 0️⃣               |                  |                                            | ✅                                         | 0️⃣                         |
| mmlu      |                | 0️⃣               | ✅                | ✅                                          | ✅                                         | ✅                           |

## Category: `hypothetical_conditional_doubt`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      |                | 0️⃣               |                  |                                            | 0️⃣                                       | 0️⃣                         |
| gsm8k     | 0️⃣            | ✅                 | 0️⃣              | 0️⃣                                        |                                           | 0️⃣                         |
| icraft    | ✅              | ✅                 | ✅                | 0️⃣                                        | ✅                                         | 0️⃣                         |
| imedqa    | ✅              | 0️⃣               | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| mmlu      |                | ✅                 | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |

## Category: `inability_to_conclude_respond`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            |                   | ✅                | 0️⃣                                        | ✅                                         |                             |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| icraft    | 0️⃣            | 0️⃣               |                  | 0️⃣                                        | ✅                                         | 0️⃣                         |
| imedqa    | 0️⃣            |                   | ✅                | ✅                                          |                                           | 0️⃣                         |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | 0️⃣                         |

## Category: `inability_to_provide`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      |                |                   | ✅                | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| icraft    | ✅              | ✅                 | 0️⃣              | 0️⃣                                        | 0️⃣                                       | ✅                           |
| imedqa    | ✅              | 0️⃣               | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

## Category: `missing_information`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            |                   | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| icraft    | ✅              | ✅                 | ✅                | 0️⃣                                        | 0️⃣                                       | ✅                           |
| imedqa    | ✅              | 0️⃣               | ✅                | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

## Category: `questioning_the_premise`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | ✅              | ✅                 | ✅                | 0️⃣                                        | 0️⃣                                       | ✅                           |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          |                                           | ✅                           |
| icraft    | ✅              | 0️⃣               | ✅                | 0️⃣                                        | 0️⃣                                       |                             |
| imedqa    | ✅              | 0️⃣               | 0️⃣              | 0️⃣                                        |                                           | ✅                           |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

## Category: `self_correction_re_evaluation`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | ✅              | 0️⃣               | 0️⃣              | 0️⃣                                        | ✅                                         | 0️⃣                         |
| gsm8k     | ✅              | ✅                 | ✅                |                                            |                                           | ✅                           |
| icraft    | ✅              |                   | 0️⃣              |                                            |                                           | 0️⃣                         |
| imedqa    | ✅              | ✅                 | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| mmlu      |                | ✅                 |                  | 0️⃣                                        | ✅                                         | ✅                           |

## Category: `stated_assumptions`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| gpqa      | 0️⃣            |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| gsm8k     | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| icraft    | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        | 0️⃣                                       |                             |
| imedqa    | ✅              |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | ✅                           |
| mmlu      | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

---

# Results Grouped by Dataset

## Dataset: `gpqa`
---

| Category                       | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:-------------------------------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| acknowledging_potential_error  | 0️⃣            | ✅                 |                  | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| contrasting_possibilities      | ✅              | 0️⃣               | 0️⃣              |                                            |                                           | 0️⃣                         |
| doubt_speculation              | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        |                                           | ✅                           |
| hesitation_processing_pause    | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| hypothetical_conditional_doubt |                | 0️⃣               |                  |                                            | 0️⃣                                       | 0️⃣                         |
| inability_to_conclude_respond  | 0️⃣            |                   | ✅                | 0️⃣                                        | ✅                                         |                             |
| inability_to_provide           |                |                   | ✅                | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| missing_information            | 0️⃣            |                   | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| questioning_the_premise        | ✅              | ✅                 | ✅                | 0️⃣                                        | 0️⃣                                       | ✅                           |
| self_correction_re_evaluation  | ✅              | 0️⃣               | 0️⃣              | 0️⃣                                        | ✅                                         | 0️⃣                         |
| stated_assumptions             | 0️⃣            |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |

### Summary Counts with Success Rates:

| Category                       |   ✅ Count |   0️⃣ Count |   Total Symbols | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|----------------:|:-----------------|:---------------------|
| questioning_the_premise        |         4 |           2 |               6 | 66.7%            | 100.0%               |
| missing_information            |         3 |           2 |               5 | 50.0%            | 83.3%                |
| self_correction_re_evaluation  |         2 |           4 |               6 | 33.3%            | 100.0%               |
| inability_to_conclude_respond  |         2 |           2 |               4 | 33.3%            | 66.7%                |
| acknowledging_potential_error  |         1 |           4 |               5 | 16.7%            | 83.3%                |
| doubt_speculation              |         1 |           4 |               5 | 16.7%            | 83.3%                |
| contrasting_possibilities      |         1 |           3 |               4 | 16.7%            | 66.7%                |
| inability_to_provide           |         1 |           3 |               4 | 16.7%            | 66.7%                |
| hesitation_processing_pause    |         0 |           6 |               6 | 0.0%             | 100.0%               |
| stated_assumptions             |         0 |           5 |               5 | 0.0%             | 83.3%                |
| hypothetical_conditional_doubt |         0 |           3 |               3 | 0.0%             | 50.0%                |

## Dataset: `gsm8k`
---

| Category                       | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:-------------------------------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| acknowledging_potential_error  | 0️⃣            | ✅                 | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| contrasting_possibilities      | 0️⃣            | ✅                 |                  |                                            | ✅                                         | ✅                           |
| doubt_speculation              | ✅              | ✅                 | ✅                | ✅                                          |                                           | ✅                           |
| hesitation_processing_pause    | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | 0️⃣                         |
| hypothetical_conditional_doubt | 0️⃣            | ✅                 | 0️⃣              | 0️⃣                                        |                                           | 0️⃣                         |
| inability_to_conclude_respond  | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| inability_to_provide           | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| missing_information            | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| questioning_the_premise        | ✅              | ✅                 | ✅                | ✅                                          |                                           | ✅                           |
| self_correction_re_evaluation  | ✅              | ✅                 | ✅                |                                            |                                           | ✅                           |
| stated_assumptions             | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

### Summary Counts with Success Rates:

| Category                       |   ✅ Count |   0️⃣ Count |   Total Symbols | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|----------------:|:-----------------|:---------------------|
| inability_to_conclude_respond  |         6 |           0 |               6 | 100.0%           | 100.0%               |
| inability_to_provide           |         6 |           0 |               6 | 100.0%           | 100.0%               |
| missing_information            |         6 |           0 |               6 | 100.0%           | 100.0%               |
| stated_assumptions             |         6 |           0 |               6 | 100.0%           | 100.0%               |
| hesitation_processing_pause    |         5 |           1 |               6 | 83.3%            | 100.0%               |
| doubt_speculation              |         5 |           0 |               5 | 83.3%            | 83.3%                |
| questioning_the_premise        |         5 |           0 |               5 | 83.3%            | 83.3%                |
| self_correction_re_evaluation  |         4 |           0 |               4 | 66.7%            | 66.7%                |
| contrasting_possibilities      |         3 |           1 |               4 | 50.0%            | 66.7%                |
| acknowledging_potential_error  |         2 |           3 |               5 | 33.3%            | 83.3%                |
| hypothetical_conditional_doubt |         1 |           4 |               5 | 16.7%            | 83.3%                |

## Dataset: `icraft`
---

| Category                       | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:-------------------------------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| acknowledging_potential_error  |                |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| contrasting_possibilities      | ✅              | ✅                 | ✅                | 0️⃣                                        | ✅                                         | 0️⃣                         |
| doubt_speculation              | ✅              | 0️⃣               | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| hesitation_processing_pause    | ✅              |                   |                  | 0️⃣                                        | 0️⃣                                       |                             |
| hypothetical_conditional_doubt | ✅              | ✅                 | ✅                | 0️⃣                                        | ✅                                         | 0️⃣                         |
| inability_to_conclude_respond  | 0️⃣            | 0️⃣               |                  | 0️⃣                                        | ✅                                         | 0️⃣                         |
| inability_to_provide           | ✅              | ✅                 | 0️⃣              | 0️⃣                                        | 0️⃣                                       | ✅                           |
| missing_information            | ✅              | ✅                 | ✅                | 0️⃣                                        | 0️⃣                                       | ✅                           |
| questioning_the_premise        | ✅              | 0️⃣               | ✅                | 0️⃣                                        | 0️⃣                                       |                             |
| self_correction_re_evaluation  | ✅              |                   | 0️⃣              |                                            |                                           | 0️⃣                         |
| stated_assumptions             | 0️⃣            | 0️⃣               | 0️⃣              | 0️⃣                                        | 0️⃣                                       |                             |

### Summary Counts with Success Rates:

| Category                       |   ✅ Count |   0️⃣ Count |   Total Symbols | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|----------------:|:-----------------|:---------------------|
| contrasting_possibilities      |         4 |           2 |               6 | 66.7%            | 100.0%               |
| doubt_speculation              |         4 |           2 |               6 | 66.7%            | 100.0%               |
| hypothetical_conditional_doubt |         4 |           2 |               6 | 66.7%            | 100.0%               |
| missing_information            |         4 |           2 |               6 | 66.7%            | 100.0%               |
| inability_to_provide           |         3 |           3 |               6 | 50.0%            | 100.0%               |
| questioning_the_premise        |         2 |           3 |               5 | 33.3%            | 83.3%                |
| inability_to_conclude_respond  |         1 |           4 |               5 | 16.7%            | 83.3%                |
| hesitation_processing_pause    |         1 |           2 |               3 | 16.7%            | 50.0%                |
| self_correction_re_evaluation  |         1 |           2 |               3 | 16.7%            | 50.0%                |
| stated_assumptions             |         0 |           5 |               5 | 0.0%             | 83.3%                |
| acknowledging_potential_error  |         0 |           4 |               4 | 0.0%             | 66.7%                |

## Dataset: `imedqa`
---

| Category                       | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:-------------------------------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| acknowledging_potential_error  |                | 0️⃣               |                  | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| contrasting_possibilities      | 0️⃣            | 0️⃣               |                  | 0️⃣                                        | 0️⃣                                       |                             |
| doubt_speculation              | ✅              | 0️⃣               | ✅                | 0️⃣                                        |                                           | ✅                           |
| hesitation_processing_pause    | 0️⃣            | 0️⃣               |                  |                                            | ✅                                         | 0️⃣                         |
| hypothetical_conditional_doubt | ✅              | 0️⃣               | ✅                | 0️⃣                                        | ✅                                         | ✅                           |
| inability_to_conclude_respond  | 0️⃣            |                   | ✅                | ✅                                          |                                           | 0️⃣                         |
| inability_to_provide           | ✅              | 0️⃣               | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| missing_information            | ✅              | 0️⃣               | ✅                | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| questioning_the_premise        | ✅              | 0️⃣               | 0️⃣              | 0️⃣                                        |                                           | ✅                           |
| self_correction_re_evaluation  | ✅              | ✅                 | ✅                | 0️⃣                                        |                                           | 0️⃣                         |
| stated_assumptions             | ✅              |                   | 0️⃣              | 0️⃣                                        | 0️⃣                                       | ✅                           |

### Summary Counts with Success Rates:

| Category                       |   ✅ Count |   0️⃣ Count |   Total Symbols | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|----------------:|:-----------------|:---------------------|
| hypothetical_conditional_doubt |         4 |           2 |               6 | 66.7%            | 100.0%               |
| doubt_speculation              |         3 |           2 |               5 | 50.0%            | 83.3%                |
| self_correction_re_evaluation  |         3 |           2 |               5 | 50.0%            | 83.3%                |
| missing_information            |         2 |           4 |               6 | 33.3%            | 100.0%               |
| inability_to_provide           |         2 |           3 |               5 | 33.3%            | 83.3%                |
| questioning_the_premise        |         2 |           3 |               5 | 33.3%            | 83.3%                |
| stated_assumptions             |         2 |           3 |               5 | 33.3%            | 83.3%                |
| inability_to_conclude_respond  |         2 |           2 |               4 | 33.3%            | 66.7%                |
| hesitation_processing_pause    |         1 |           3 |               4 | 16.7%            | 66.7%                |
| acknowledging_potential_error  |         0 |           4 |               4 | 0.0%             | 66.7%                |
| contrasting_possibilities      |         0 |           4 |               4 | 0.0%             | 66.7%                |

## Dataset: `mmlu`
---

| Category                       | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Llama-8B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:-------------------------------|:---------------|:------------------|:-----------------|:-------------------------------------------|:------------------------------------------|:----------------------------|
| acknowledging_potential_error  | ✅              | ✅                 | 0️⃣              | ✅                                          |                                           | 0️⃣                         |
| contrasting_possibilities      | 0️⃣            | ✅                 | 0️⃣              | 0️⃣                                        | ✅                                         | ✅                           |
| doubt_speculation              | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| hesitation_processing_pause    |                | 0️⃣               | ✅                | ✅                                          | ✅                                         | ✅                           |
| hypothetical_conditional_doubt |                | ✅                 | 0️⃣              | 0️⃣                                        | 0️⃣                                       | 0️⃣                         |
| inability_to_conclude_respond  | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | 0️⃣                         |
| inability_to_provide           | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| missing_information            | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| questioning_the_premise        | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |
| self_correction_re_evaluation  |                | ✅                 |                  | 0️⃣                                        | ✅                                         | ✅                           |
| stated_assumptions             | ✅              | ✅                 | ✅                | ✅                                          | ✅                                         | ✅                           |

### Summary Counts with Success Rates:

| Category                       |   ✅ Count |   0️⃣ Count |   Total Symbols | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|----------------:|:-----------------|:---------------------|
| doubt_speculation              |         6 |           0 |               6 | 100.0%           | 100.0%               |
| inability_to_provide           |         6 |           0 |               6 | 100.0%           | 100.0%               |
| missing_information            |         6 |           0 |               6 | 100.0%           | 100.0%               |
| questioning_the_premise        |         6 |           0 |               6 | 100.0%           | 100.0%               |
| stated_assumptions             |         6 |           0 |               6 | 100.0%           | 100.0%               |
| inability_to_conclude_respond  |         5 |           1 |               6 | 83.3%            | 100.0%               |
| hesitation_processing_pause    |         4 |           1 |               5 | 66.7%            | 83.3%                |
| contrasting_possibilities      |         3 |           3 |               6 | 50.0%            | 100.0%               |
| acknowledging_potential_error  |         3 |           2 |               5 | 50.0%            | 83.3%                |
| self_correction_re_evaluation  |         3 |           1 |               4 | 50.0%            | 66.7%                |
| hypothetical_conditional_doubt |         1 |           4 |               5 | 16.7%            | 83.3%                |

---

# Aggregated Analysis Across All Datasets

## Overall Category Performance

This table shows the total performance of each category across all datasets and models:

| Category                       |   Total ✅ |   Total 0️⃣ |   Total Occurrences | ✅ Success Rate   | ✅+0️⃣ Success Rate   |
|:-------------------------------|----------:|------------:|--------------------:|:-----------------|:---------------------|
| missing_information            |        21 |           8 |                  30 | 70.0%            | 96.7%                |
| doubt_speculation              |        19 |           8 |                  30 | 63.3%            | 90.0%                |
| questioning_the_premise        |        19 |           8 |                  30 | 63.3%            | 90.0%                |
| inability_to_provide           |        18 |           9 |                  30 | 60.0%            | 90.0%                |
| inability_to_conclude_respond  |        16 |           9 |                  30 | 53.3%            | 83.3%                |
| stated_assumptions             |        14 |          13 |                  30 | 46.7%            | 90.0%                |
| self_correction_re_evaluation  |        13 |           9 |                  30 | 43.3%            | 73.3%                |
| hesitation_processing_pause    |        11 |          13 |                  30 | 36.7%            | 80.0%                |
| contrasting_possibilities      |        11 |          13 |                  30 | 36.7%            | 80.0%                |
| hypothetical_conditional_doubt |        10 |          15 |                  30 | 33.3%            | 83.3%                |
| acknowledging_potential_error  |         6 |          17 |                  30 | 20.0%            | 76.7%                |

