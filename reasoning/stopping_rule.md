# Implementing Custom Stopping Rules

This document explains how to create and test new stopping rules for our language model decoding simulator. The goal is to implement logic that can stop the generation of a reasoning trace early, saving computational resources.

You will primarily be working in **`stopping_rules.py`** to define the logic and **`simulator.py`** to test it.

-----

## 🧐 How It Works: The Big Picture

1.  The **`simulator.py`** script reads a reasoning trace (e.g., from the GSM8K dataset).
2.  It simulates the token-by-token generation of that trace.
3.  After each token, it asks a **`StoppingRule`** object, "Should we stop now?"
4.  Your job is to create a new, smarter `StoppingRule` class that can make this decision.

-----

## 🚀 How to Implement a New Stopping Rule

Your entire implementation will be done in the **`stopping_rules.py`** file.

### Step 1: Create Your New Rule Class

Create a new class that inherits from the abstract `StoppingRule` class. You can use `LengthStoppingRule` as a simple example or `UncertaintyStoppingRule` as a template.

```python
# In stopping_rules.py

class MySophisticatedRule(StoppingRule):
    # ... your implementation will go here
```

### Step 2: Implement the `__init__` Method

In the initializer, call the parent `__init__` and define any parameters your rule needs. For example, you might need a window size or a specific threshold.

```python
class MySophisticatedRule(StoppingRule):
    def __init__(self, lazy_interval: int = 1, my_threshold: float = 0.8, window_size: int = 10):
        super().__init__(lazy_interval)
        self.my_threshold = my_threshold
        self.window_size = window_size
        # This value will be calculated during training
        self.learned_parameter = None
```

### Step 3: Implement the `train` Method

This is where your rule "learns" from data. The method receives a list of tokenized reasoning traces from the training set. Your goal is to calculate and set the parameters your rule will use to make decisions (like the `self.learned_parameter` from the example above).

```python
# In your MySophisticatedRule class

def train(self, tokenized_reasoning_traces):
    # tokenized_reasoning_traces is a list of lists, e.g., [[3, 14, 25], [4, 52, ...]]
    print("Training my sophisticated rule...")
    
    # Example: Calculate some complex metric over the traces
    all_metrics = []
    for trace in tokenized_reasoning_traces:
        # Replace this with your actual logic
        metric = len(trace) / 100.0 
        all_metrics.append(metric)
    
    # Set a learned parameter, like a quantile of the metrics you calculated
    self.learned_parameter = np.quantile(all_metrics, self.my_threshold)
    print(f"Learned parameter set to: {self.learned_parameter}")
```

### Step 4: Implement the `_should_stop` Method

This is the core logic of your rule. It's called during the simulation to decide if generation should stop. You have access to `self.tokens_so_far`, which is a list of all tokens generated for the *current* reasoning trace.

Return `True` to stop decoding or `False` to continue.

```python
# In your MySophisticatedRule class

def _should_stop(self) -> bool:
    # self.tokens_so_far contains the list of tokens generated in the current simulation
    current_length = len(self.tokens_so_far)
    
    # Your logic here. For example:
    # Analyze the last `self.window_size` tokens
    if current_length < self.window_size:
        return False

    recent_tokens = self.tokens_so_far[-self.window_size:]
    
    # Calculate some live metric based on recent_tokens
    # (This is just a placeholder example)
    live_metric = sum(recent_tokens) / current_length

    # Compare the live metric to your learned parameter
    if live_metric > self.learned_parameter:
        print("Decision: Stop!")
        return True
    else:
        return False
```

**That's it\!** You have now defined a new, trainable stopping rule.

-----

## 🧪 How to Test Your Stopping Rule

Now, let's use your new rule in the simulation.

### Step 1: Open `simulator.py`

Navigate to the `if __name__ == "__main__":` block at the bottom of the file.

### Step 2: Import and Use Your Rule

1.  Import your new class from `stopping_rules`.
2.  Replace `LengthStoppingRule` with your new class.

<!-- end list -->

```python
# In simulator.py

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    model_name = args.model_name
    data_name = args.data_name
    
    # --- YOUR CHANGES GO HERE ---
    
    # 1. Import your new rule
    from stopping_rules import MySophisticatedRule

    # 2. Instantiate your rule instead of the old one
    # stopping_rule = LengthStoppingRule(lazy_interval=1, quantile=0.95)
    stopping_rule = MySophisticatedRule(lazy_interval=1, my_threshold=0.85)

    # --- NO MORE CHANGES NEEDED ---

    # train the stopping rule
    train_stopping_rule(stopping_rule, model_name, data_name)

    # run the decoding test environment
    decoding_env = DecodingTestEnvironment(model_name=model_name, data_name=data_name, stopping_rule=stopping_rule)
    avg_tokens_saved, avg_percentage_saved = decoding_env.run_simulation()
    print(f"Average tokens saved: {avg_tokens_saved:.2f}")
    print(f"Average percentage saved: {avg_percentage_saved:.2f}%")
```

### Step 3: Run the Simulation

Run the script from your terminal. You can specify different models or datasets.

```bash
python simulator.py --model_name microsoft/Phi-4-reasoning --data_name 'gsm8k'
```

The script will first call your `train()` method using the training data. Then, it will run the simulation on the test data, using your `_should_stop()` method to make decisions. Finally, it will report the average tokens and percentage saved.

Happy coding\!