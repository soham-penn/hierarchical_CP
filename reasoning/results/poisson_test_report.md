# Poisson Process Assumption Test Report

## Legend
- **✅**: Passes both K-S and Ljung-Box tests (Consistent with Poisson process).
- **❌**: Fails at least one of the tests (Not consistent with Poisson process).
- **⚪️**: Not enough events to perform the test (<20 intervals).
- **Format**: `With Context` / `Without Context`

## Results for Category: `assumption_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ✅/✅            | ✅/⚪️              | ✅/❌              | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| gsm8k     | ⚪️/✅           | ✅/❌               | ⚪️/❌             | ⚪️/⚪️                                     | ⚪️/✅                        |
| icraft    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| mmlu      | ⚪️/❌           | ✅/❌               | ✅/❌              | ⚪️/✅                                      | ⚪️/❌                        |

## Results for Category: `cant_respond`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/✅                                      | ⚪️/⚪️                       |
| gsm8k     | ⚪️/❌           | ⚪️/❌              | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/❌                        |
| icraft    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| mmlu      | ⚪️/❌           | ⚪️/✅              | ⚪️/✅             | ⚪️/✅                                      | ⚪️/⚪️                       |

## Results for Category: `contrast_and_doubt_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ❌/❌            | ❌/✅               | ✅/✅              | ❌/❌                                       | ✅/✅                         |
| gsm8k     | ❌/❌            | ❌/❌               | ✅/❌              | ✅/⚪️                                      | ⚪️/❌                        |
| icraft    | ✅/✅            | ✅/✅               | ❌/✅              | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ❌/❌            | ❌/✅               | ✅/✅              | ✅/✅                                       | ⚪️/⚪️                       |
| mmlu      | ❌/❌            | ❌/❌               | ✅/❌              | ❌/❌                                       | ⚪️/❌                        |

## Results for Category: `doubt_the_question`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ✅/❌            | ✅/❌               | ⚪️/❌             | ✅/❌                                       | ⚪️/✅                        |
| gsm8k     | ❌/❌            | ❌/❌               | ✅/❌              | ⚪️/⚪️                                     | ⚪️/❌                        |
| icraft    | ⚪️/✅           | ✅/✅               | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ✅/❌            | ❌/✅               | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/✅                        |
| mmlu      | ✅/❌            | ❌/❌               | ✅/❌              | ⚪️/✅                                      | ⚪️/❌                        |

## Results for Category: `hesitation_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ❌/❌            | ❌/❌               | ✅/❌              | ❌/❌                                       | ✅/✅                         |
| gsm8k     | ❌/❌            | ❌/❌               | ✅/❌              | ❌/⚪️                                      | ✅/❌                         |
| icraft    | ✅/❌            | ❌/❌               | ❌/✅              | ❌/❌                                       | ⚪️/✅                        |
| imedqa    | ❌/❌            | ❌/❌               | ✅/❌              | ❌/✅                                       | ✅/✅                         |
| mmlu      | ❌/❌            | ❌/❌               | ❌/❌              | ❌/❌                                       | ❌/❌                         |

## Results for Category: `missing_info_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/✅             | ⚪️/❌                                      | ⚪️/⚪️                       |
| gsm8k     | ⚪️/❌           | ⚪️/❌              | ⚪️/❌             | ⚪️/⚪️                                     | ⚪️/❌                        |
| icraft    | ⚪️/✅           | ⚪️/⚪️             | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/✅                        |
| imedqa    | ✅/❌            | ✅/❌               | ⚪️/✅             | ✅/✅                                       | ✅/✅                         |
| mmlu      | ⚪️/❌           | ⚪️/❌              | ⚪️/❌             | ⚪️/✅                                      | ⚪️/❌                        |

## Results for Category: `mistake_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| gsm8k     | ⚪️/⚪️          | ⚪️/❌              | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| icraft    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| mmlu      | ⚪️/⚪️          | ✅/❌               | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/⚪️                       |

## Results for Category: `redo_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ⚪️/⚪️          | ⚪️/⚪️             | ✅/✅              | ⚪️/❌                                      | ⚪️/⚪️                       |
| gsm8k     | ✅/✅            | ✅/⚪️              | ✅/✅              | ⚪️/⚪️                                     | ⚪️/✅                        |
| icraft    | ⚪️/❌           | ⚪️/⚪️             | ⚪️/✅             | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ❌/❌            | ⚪️/⚪️             | ⚪️/✅             | ✅/❌                                       | ⚪️/✅                        |
| mmlu      | ✅/❌            | ✅/✅               | ✅/✅              | ✅/✅                                       | ⚪️/✅                        |

## Results for Category: `unsure_keywords`
---

| Dataset   | Qwen/QwQ-32B   | Qwen/Qwen3-0.6B   | Qwen/Qwen3-32B   | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B   | microsoft/Phi-4-reasoning   |
|:----------|:---------------|:------------------|:-----------------|:------------------------------------------|:----------------------------|
| gpqa      | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| gsm8k     | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| icraft    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| imedqa    | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |
| mmlu      | ⚪️/⚪️          | ⚪️/⚪️             | ⚪️/⚪️            | ⚪️/⚪️                                     | ⚪️/⚪️                       |

