# Ideas
## Abstention 
- Baselines
  - Use another LLM to judge when to stop/abstein (costly)
  - Directly prompt the model to be "careful"
- Our method
  - define a set of indicators for uncertainty (token based, other ways??)
    - what about existing uncertainty measures (entropy, likelihood)?
    - The impression is that most previous methods are **short-form**, may not be easily adaptable to reasoning scenarios. (Maybe another baseline?)
  - during inference, check whether the empirical "density" of uncertainty exceeds a threshold; if so, stop thinking and use an intervention (e.g. instruct the model to explain its uncertainty to the user)
  - the threshold is defined via conformal outlier detection (collect a dataset of "normal" behavior and find a 1-$\alpha$ threshold)
- Metrics:
  - Abstention rate
  - number of saved tokens/time
  - accuracy on the base dataset -- as a sanity check
  - measures of "coherence" for interpretability?

# Tasks

- [ ] @soham & @tao & @Maxine:
  - Investigating the uncertainty expression process in the reasoning traces? parametric methods might include modelling as a Poisson arrival process
    - testing some possible assumptions
    - a probability model that fits reasonably well
    - how can this fit into a conformal framework without strong parametric assumptions?
    - is it true that we can still use, say, a Poisson process assumption to calculate statistics but still remain "distribution-free"?
- [ ] @yan & @xinyu & soham EDA to mine the uncertainty keywords/metrics
  - [x] finer/more linguistically motivated categories of uncertainty expressions
  - [ ] How to test "conpounding" in normal situations? Is it true that the occurance is rare enough not to hurt "independence" assumption? Some uncertainty keywords are more likely to occur together when the context is missing; but what about when the context is present?
    - [ ] Compute the correlation of the "count" uncertainty keywords within the reasoning traces
    - [ ] if correlation is significant, then we can model the "gaps" between the occurances of uncertainty keywords 
      - [ ] maybe then a **Markov Chain** or **HMM** arises?
    - [ ] if correlation is significant, one other fix is to "thin/group", i.e. remove/merge one of the correlated keywords/categories
- [ ] @xinyu: Implement baselines
  - [ ] Use another LLM to judge when to stop/abstein (costly)
  - [ ] Directly prompt the model to be "careful"


- [ ] @xinyu:
  - [ ] use the correct test for hypotheses
  - [x] create a pipeline for Train/Test split for each reasoning trace dataset
  - [x] basic support for conformal methods
      - a **simulation environment** that mimics how decoding happens in real time
        - scan through the reasoning traces
          - how many tokens there have been so far
          - calculate the density of uncertainty so far
          - stop generation when a decision is made
      - create an abstract class and define the methods that need to be implemented for comformal decisions
      - given prompt $x$ and completion $y$, produce transformation $h(y)$ which matches longer messages to shorter messages.
        - in abstention settings, can be truncating reasoning traces, or setting to empty strings
- [ ] @georgy:
  - [ ] track "blabbering," what are other signs of uncertainty from the reasoning traces
- [ ] @Yan & @Maxine & @Tao:
  - [ ] lit review on
    - [ ] uncertainty quantification & abstention for reasoning
    - [ ] reasoning in agentic setup
  - [x] create an overleaf

# Decoding Simulator

Please read [stopping_rule.md](/stopping_rule.md) to build your custom early stopping rule and calculate savings.

# Datasets
Create a standardized dataset generator for the following datasets:
Code adapted from this [repo](https://github.com/facebookresearch/AbstentionBench), under [`data/`](/data/)
- Medical Reasoning Benchmarks
  - [x] iCRAFT-MD
  - [x] iMedQA
- Scientific Reasoning Benchmarks
  - [x] GPQA
- Mathematical Reasoning Benchmarks
  - [x] MMLU
  - [x] GSM8K
- **Agentic Benchmarks**
  - e.g. [Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners](https://robot-help.github.io) 
  - [ ] These are special because they might directly use the model response to make downstream decisions, but "assumptions made in the response" might not be properly captured in this context.
  - [ ] May be a good idea to **extend** to reasoning to inflate uncertainty/resolve ambiguity?

# Inference:
  - We support the following inference methods:
    - [x] `Qwen3` families
    - [x] `QwQ-32B`
    - [x] `Phi-4-reasoning`
    - [x] `DeepSeek-R1` distilled models

  - Completed inference models:
    - RL-Tuned Models
      - [x] `microsoft/Phi-4-reasoning`
      - [x] `Qwen/QwQ-32B`
      - [x] `Qwen/Qwen3-32B`
    - Distilled Models
      - [x] `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
      - [x] `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
      - [x] `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
      - [x] `Qwen/Qwen3-0.6B`
      - [x] `Qwen/Qwen3-1.7B`
      - [x] `Qwen/Qwen3-8B`

# EDA
code in [`eda.py`](eda.py)
- [ ] **More complete analysis of the reasoning traces**

# Evaluation:
Code under [`detectors.py`](detect/detectors.py)
- [x] LLM-ad-Judge evaluation scripts for abstention
- [x] Keyword-based evaluation scripts for abstention
- [ ] **Keyword-based evaluation scripts for uncertainty expression**
  - [ ] Differentiate categories of uncertainty expressions
  - [ ] Develop a standardized uncertainty expression detector


# Infrastructure:
- [ ] Train/Test split for each dataset
  - [ ] Within domain split
  - [ ] Cross domain split
- [ ] Infrastructure for applying comformal methods

# Hypotheses:
- [ ] Does the model spend more time thinking when provided with partial information?
  - $H_0$: The thinking time (measured in words/tokens) is the same regardless of whether the model is provided with partial information or not, i.e. 
  
  $H_0: \mu_{partial} = \mu_{full}$

  where $\mu_{partial}$ is the mean thinking time when the model is provided with partial information, and $\mu_{full}$ is the mean thinking time when the model is not provided with partial information.
  - $H_1$: The thinking time (measured in words/tokens) is significantly longer when the model is provided with partial information compared to when it is not, i.e.
  
  $H_1: \mu_{partial} > \mu_{full}$
  
- [ ] Does the model express more uncertainty when provided with partial information?
  - $H_0$: The frequency of uncertainty phrases in the model's thinking content is the same regardless of whether the model is provided with partial information or not, i.e. 
  
  $H_0: \pi_{partial} = \pi_{full}$
  
  where $\pi_{partial}$ is the density of uncertainty phrases in the model's thinking content when the model is provided with partial information, and $\pi_{full}$ is the density of uncertainty phrases in the model's thinking content when the model is not provided with partial information.

  - $H_1$: The frequency of uncertainty phrases in the model's thinking content is significantly higher when the model is provided with partial information compared to when it is not, i.e.

  $H_1: \pi_{partial} > \pi_{full}$
  
- Are there different categories of uncertainty expressions in the model's thinking content? For example, there may be expressions of doubt (maybe, perhaps, possibly), expressions of hesitation (wait, hold on, let me think), expression of missing information ( without knowing, without context, without information), or maybe other categories?
  - [ ] Brainstorm and explore via EDA to pin down the categories of uncertainty expressions in the model's thinking content.
  - [ ] Test whether each category of uncertainty expressions is significantly more frequent when the model is provided with partial information compared to when it is not.
  - [ ] Test whether there is a significant difference in the frequency of different categories of uncertainty expressions when the model is provided with partial information compared to when it is not.

- Domain Transferability: Do models present the same kind of uncertainty expressing behavior when the domain of the tasks change?

- Can we stop the model early to save token usage without affecting the model's performance when it comes to abstention?
  - [ ] Conformal prediction to stop the model early when it is uncertain.

# Testing Methods:
- [ ] What are the right formulations for our hypothesis tests?
  - [ ] How to deal with discrete counts?
  - [ ] Are there multiple-testing problems?
  - [ ] Calibration techniques for unlabeled data? Full conformal methods?
- [ ] How do we adapt comformal methods? What assumptions will be made?

# Data Analysis:

If you haven't installed the required packages, you can do so by running the following command:
```bash
conda create -n reasoning python=3.12 -y
conda activate reasoning
pip install -r requirements.txt
```

Run the following command to activate the virtual environment:
```bash
conda activate reasoning
```

# Inference

If, additionally, you'd like to use [vllm](https://github.com/vllm-project/vllm) to make LLM inferences, you can install it by running the following command:
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
```

Below is the code snippet to run the reasoning environment on the Polaris cluster:

Run the following command to activate the virtual environment:
```bash
qsub -I -l select=1 -l filesystems=home:eagle -l walltime=1:00:00 -q debug-scaling -A araia
module use /soft/modulefiles
module load conda ; conda activate reasoning
cd reasoning
```

Run the following command to activate the virtual environment solely for data analysis:
```bash
module use /soft/modulefiles
module load conda ; conda activate reasoning
```

To remove files with a certain suffix, you can run the following command:
```bash
find . -type f -name "*_stopping_rule_results.jsonl" -delete
```