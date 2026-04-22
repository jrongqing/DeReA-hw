import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import json

MODEL_SIZE = "8B"
PARALLEL_SIZE = 1



input_file_path = 'slang/eval/data/LoMI-CN-linked-new.json'
output_file_path = 'slang/eval/cot/Cot_eval_new.json'
model_path = '/data/shared/Qwen3-8B/'

tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, top_k=50, max_tokens=2048*4)
llm = LLM(model=model_path, tensor_parallel_size=PARALLEL_SIZE)


system_prompt_template ='''
You are a translation assistant whose task is to analyze translation background information and ultimately provide the optimal Chinese translation.
You will be given the following translation background information:

1. The original English sentence.
2. The literal translation result from the model.
3. The analysis method and detection result for unconventional expressions within the sentence.
4. Dictionary entries retrieved from RAG (Retrieval-Augmented Generation) (Unconventional expressions: their figurative meanings).

'''

user_prompt_template = '''
Input Information:
1.Original English Sentence: {en}
2.Analysis Method and Detection Result for Unconventional Expressions in the Sentence: {think}
3.Retrieved Dictionary Entries: {rag_info}
Translation 0: Model's Literal Translation Result: {direct}

Output Format:Based on the retrieved dictionary entries, generate three separate translations: Translation 1, Translation 2, and Translation 3.
Systematically analyze the translation quality of unconventional expressions and the overall sentence quality for Translation 0, Translation 1, Translation 2, and Translation 3, and finally determine which translation will be selected as the final output:
Output Example:
*** RAG Translation Results ***
Translation 1: Based on dictionary entry '...', translated as: "..."
Translation 2: Based on dictionary entry '...', translated as: "..."
Translation 3: Based on dictionary entry '...', translated as: "..."
*** Translation Quality Analysis ***
Translation 0: Result: ... Analysis of unconventional expression quality: ... Analysis of overall sentence quality: ...
Translation 1: Result: ... Analysis of unconventional expression quality: ... Analysis of overall sentence quality: ...
Translation 2: Result: ... Analysis of unconventional expression quality: ... Analysis of overall sentence quality: ...
Translation 3: Result: ... Analysis of unconventional expression quality: ... Analysis of overall sentence quality: ...

*** Final Translation ***
(Output only the final translation result in this section; no other information is permitted)
'''

user_prompt_template_direct = '''
Translate the following English sentence into Chinese.
English: {sentence}
Chinese:
'''

user_prompt_template_new = '''
As a translation expert, please translate the sentence I provide to you into Chinese:
The sentence may contain unconventional expressions (such as slang, idioms, proverbs, and other expressions whose true meaning differs from the literal meaning). Please first detect whether the sentence contains any unconventional expressions. If they exist, provide an explanation of the unconventional expression before translating the sentence; if they do not exist, translate the sentence directly.

input sentence:{sentence}

Output Format:
**Whether it contains unconventional expressions (and their explanations)**
**Final Translation Result**
'''

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

buffer = []

for entry in data:
    if entry["detect_idiom"] == []:
        user_prompt = user_prompt_template_direct.format(sentence = entry["sentence"])
        message = [
            {"role": "user", "content": user_prompt}
        ]
    elif entry["rag_info"] != []:
        user_prompt = user_prompt_template.format(
            en = entry["sentence"],
            direct = entry["direct_8b"],
            think = entry["dpo_eval_8B"],
            rag_info = entry["rag_info"],
            )
        message = [
            {"role": "system", "content": system_prompt_template},
            {"role": "user", "content": user_prompt}
        ]
    else:
        user_prompt = user_prompt_template_new.format(sentence = entry["sentence"])
        message = [
            {"role": "user", "content": user_prompt}
        ]

    text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  
    )
    buffer.append(text)

print("test prompt: \n", buffer[0])

def main():  
    outputs = llm.generate(buffer, sampling_params=sampling_params)
    assert len(data) == len(outputs), "The lists are not of the same length."
    for entry, output in zip(data, outputs):
        entry["cot_result"] = output.outputs[0].text.strip()

    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent = 2)

    print(f"翻译结果已保存至: {output_file_path}")


if __name__ == "__main__":
    main()
    
