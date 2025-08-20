import os
os.environ['HF_HOME'] = './hf_home'
os.environ['TRANSFORMERS_CACHE'] = './hf_home/hub'
import torch

# from huggingface_hub import login
# login("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")

from vllm import LLM, SamplingParams

class Inference:
    def __init__(self, model_name):
        self.model_name = model_name

        tp_size = 4 if torch.cuda.device_count() >= 4 else torch.cuda.device_count()
        kwargs = {
                "model": model_name,
                "enforce_eager": True,
                "trust_remote_code": True,
                "tensor_parallel_size": tp_size,
        }

        if model_name == "microsoft/Phi-4-mini-flash-reasoning":
                kwargs["enable_prefix_caching"] = False
                kwargs["enable_chunked_prefill"] = False
        try:
            self.llm = LLM(**kwargs)
        except:
            self.llm = LLM(**kwargs)
            
        self.sampling_params = SamplingParams(max_tokens=32768)

    def get_response(self, prompts, enable_thinking=True):
        messages = []
        for prompt in prompts:
            messages.append(
                [{"role": "user", "content": prompt}]
            )
        try:
            outputs = self.llm.chat(
                messages,
                self.sampling_params,
                chat_template_kwargs={"enable_thinking": enable_thinking},
                use_tqdm=True,
            )
        except:
            outputs = self.llm.chat(
                messages,
                self.sampling_params,
                use_tqdm=True,
            )
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            if enable_thinking:
                if "</think>" in generated_text:
                    content = generated_text.split("</think>")[1].strip()
                    thinking_content = generated_text.split("</think>")[0].strip()
                else:
                    # If the </think> token is not found, it assumes that the entire output is thinking_content.
                    thinking_content = generated_text.strip()
                    content = ""
                results.append((thinking_content, content))
            else:
                results.append(generated_text.strip())
        return results