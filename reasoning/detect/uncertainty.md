# Uncertainty Expression Keywords

## Categories of Uncertainty

The `uncertainty_expressions` dictionary groups keywords and phrases into the following categories, providing a structured approach to identifying different types of uncertainty:

### Epistemic Modality (Doubt/Possibility)

  * **`doubt_speculation`**: Expressions directly indicating a lack of certainty or suggesting a possibility (e.g., "maybe," "not entirely sure").
  * **`hypothetical_conditional_doubt`**: Phrases that introduce a condition under which current understanding might be flawed (e.g., "unless there," "maybe I missed").

### Metacognitive/Self-Correction

  * **`self_correction_re_evaluation`**: Phrases indicating a need to re-examine a problem or thought process (e.g., "try again," "another angle").
  * **`hesitation_processing_pause`**: Interjections or phrases signaling a momentary pause for consideration (e.g., "wait," "let me think").
  * **`acknowledging_potential_error`**: Direct admissions or suggestions of a flaw in the problem statement or interpretation (e.g., "there's a typo," "mistake in the problem").
  * **`questioning_the_premise`**: Expressions that cast doubt on the clarity or accuracy of the question itself (e.g., "maybe the question," "the problem could be").

### Information Sufficiency/Limitation

  * **`missing_information`**: Statements indicating missing, insufficient, or unclear data (e.g., "don't have any information," "without specific details").
  * **`inability_to_provide`**: Phrases indicating a functional limitation in generating a complete or certain response due to insufficient information (e.g., "does not provide," "not given").

### Speech Act/Performative Limitations

  * **`inability_to_conclude_respond`**: Direct statements about the speaker's inability to give a definitive answer or perform a requested action (e.g., "can't respond," "cannot verify").

### Conditional/Contrastive Uncertainty

  * **`contrasting_possibilities`**: Phrases that introduce an alternative or contrasting idea while maintaining doubt (e.g., "but maybe," "alternatively it could be").
  * **`stated_assumptions`**: Expressions that explicitly state a condition or premise taken as true, often implying that the conclusion is dependent on this assumption (e.g., "assuming," "standard information").

-----

## Usage

You can import the `uncertainty_expressions` dictionary into your Python projects to identify and analyze different types of uncertainty in text data.

Here's a quick example:

```python
from uncertainty_keywords import uncertainty_expressions

# Accessing a category of uncertainty expressions
doubt_keywords = uncertainty_expressions["doubt_speculation"]
print("Doubt/Speculation Keywords:", doubt_keywords)

# You can iterate through all categories
for category, keywords in uncertainty_expressions.items():
    print(f"\nCategory: {category.replace('_', ' ').title()}")
    for keyword in keywords:
        print(f"  - {keyword}")
```



## 💡 Categories of Uncertainty

The `uncertainty_expressions` dictionary groups keywords and phrases into the following categories, providing a structured approach to identifying different types of uncertainty:

| Category                     | Sub-Category                         | Description                                                                 | Examples                                                                |
| :--------------------------- | :----------------------------------- | :-------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| **Epistemic Modality** | `doubt_speculation`                  | Expressions directly indicating a lack of certainty or suggesting a possibility. | "maybe", "not entirely sure", "could be"                                |
|                              | `hypothetical_conditional_doubt`     | Phrases introducing a condition under which current understanding might be flawed. | "unless there", "maybe I missed"                                        |
| **Metacognitive/Self-Correction** | `self_correction_re_evaluation`      | Phrases indicating a need to re-examine a problem or thought process.       | "try again", "another angle", "reread"                                  |
|                              | `hesitation_processing_pause`        | Interjections or phrases signaling a momentary pause for consideration.     | "wait", "let me think", "um"                                            |
|                              | `acknowledging_potential_error`      | Direct admissions or suggestions of a flaw in the problem statement or interpretation. | "there's a typo", "mistake in the problem", "mistakenly"                |
|                              | `questioning_the_premise`            | Expressions that cast doubt on the clarity or accuracy of the question itself. | "maybe the question", "the problem could be"                            |
| **Information Sufficiency/Limitation** | `missing_information`                | Statements indicating missing, insufficient, or unclear data.               | "don't have any information", "without specific details", "no data"     |
|                              | `inability_to_provide`               | Phrases indicating a functional limitation in generating a complete or certain response due to insufficient information. | "does not provide", "not given", "cannot supply"                        |
| **Speech Act/Performative Limitations** | `inability_to_conclude_respond`      | Direct statements about the speaker's inability to give a definitive answer or perform a requested action. | "can't respond", "cannot verify", "unable to determine"                 |
| **Conditional/Contrastive Uncertainty** | `contrasting_possibilities`          | Phrases that introduce an alternative or contrasting idea while maintaining doubt. | "but maybe", "alternatively it could be", "but, possibly"               |
|                              | `stated_assumptions`                 | Expressions that explicitly state a condition or premise taken as true, often implying that the conclusion is dependent on this assumption. | "assuming", "standard information", "presumed"                          |

