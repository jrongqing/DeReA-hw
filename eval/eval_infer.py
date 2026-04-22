import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import json


input_file_path = 'slang/eval/ende/eval/LoMI-DE.json'
output_file_path = 'slang/eval/ende/eval/llama_result'
model_path = '/data/shared/Llama-3-8B-Instruct'
LANG = 'German'
PARALLEL_SIZE = 1
tokenizer = AutoTokenizer.from_pretrained(model_path)
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
sampling_params = SamplingParams(
    temperature=0.0, 
    top_p=0.95, 
    top_k=50, 
    max_tokens=1024,  
    stop_token_ids=stop_token_ids 
)
llm = LLM(
    model=model_path, 
    tensor_parallel_size=PARALLEL_SIZE,
    gpu_memory_utilization=0.95, 
    max_model_len=4096, 
    trust_remote_code=True
)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


user_prompt_template = '''
Translate the following English sentence into {lang}. {sentence}
'''
buffer = []

for entry in data:
    user_prompt = user_prompt_template.format(lang = LANG,sentence = entry["sentence"])
    message = [
        {"role": "system", "content": "You are a professional translator. Translate the user's English input into standard Simplified {lang}. Output ONLY the translated {lang} text. Do not repeat the English.".format(lang=LANG)},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    buffer.append(text)

outputs = llm.generate(buffer, sampling_params=sampling_params)
assert len(data) == len(outputs), "The lists are not of the same length."
for entry, output in zip(data, outputs):
    entry["direct"] = output.outputs[0].text.strip()
if not os.path.exists(output_file_path):
    os.makedirs(output_file_path)
with open(os.path.join(output_file_path, 'direct.json'), 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent = 2)



user_prompt_template = '''
As a translation expert, please translate the sentence I provide to you into {lang}:
The sentence may contain unconventional expressions (such as slang, idioms, proverbs, and other expressions whose true meaning differs from the literal meaning). Please first detect whether the sentence contains any unconventional expressions. If they exist, provide an explanation of the unconventional expression before translating the sentence; if they do not exist, translate the sentence directly.

input sentence:{sentence}

Output Format:
**Whether it contains unconventional expressions (and their explanations)**
**Final Translation Result**
'''

buffer = []
for entry in data:
    user_prompt = user_prompt_template.format(lang = LANG, sentence = entry["sentence"])
    message = [
        {"role": "system", "content": "You are a professional translator. Translate the user's English input into standard Simplified {lang}. Output ONLY the translated {lang} text. Do not repeat the English.".format(lang=LANG)},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )
    buffer.append(text)

outputs = llm.generate(buffer, sampling_params=sampling_params)
assert len(data) == len(outputs), "The lists are not of the same length."
for entry, output in zip(data, outputs):
    entry["slangdit"] = output.outputs[0].text.strip()
    if "**Final Translation Result:**" in entry["slangdit"]:
        text = entry["slangdit"].split("**Final Translation Result:**")
        entry["slangdit"] = text[1]
        entry["slangdit"] = entry["slangdit"].replace("\n", "")
        entry["slangdit"] = entry["slangdit"].strip()
    elif "**Final Translation Result**" in entry["slangdit"]:
        text = entry["slangdit"].split("**Final Translation Result**")
        entry["slangdit"] = text[1]
        entry["slangdit"] = entry["slangdit"].replace("\n", "")
        entry["slangdit"] = entry["slangdit"].strip()
    else:
        entry["slangdit"] = entry["direct"]

with open(os.path.join(output_file_path, 'slangdit.json'), 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent = 2)


# sRAG
user_prompt_template = '''
{srag_info}
Given the above knowledge, translate the following English sentence into {lang}.
English:{sentence}
'''
buffer = []
for entry in data:
    user_prompt = user_prompt_template.format(srag_info = entry["srag_info"],lang = LANG ,sentence = entry["sentence"])
    message = [
        {"role": "system", "content": "You are a professional translator. Translate the user's English input into standard Simplified {lang}. Output ONLY the translated {lang} text. Do not repeat the English.".format(lang=LANG)},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    buffer.append(text)

outputs = llm.generate(buffer, sampling_params=sampling_params)
assert len(data) == len(outputs), "The lists are not of the same length."
for entry, output in zip(data, outputs):
    entry["srag"] = output.outputs[0].text.strip()

with open(os.path.join(output_file_path, 'srag.json'), 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent = 2)

# our
system_prompt_template ='''
You are a translation assistant whose task is to analyze translation background information and ultimately provide the optimal {lang} translation.
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
Translate the following English sentence into {lang}.
English: {sentence}
'''

buffer = []
for entry in data:
    if entry["detect_idiom"] == []:
        user_prompt = user_prompt_template_direct.format(lang = LANG, sentence = entry["sentence"])
        message = [
            {"role": "system", "content": "You are a professional translator. Translate the user's English input into standard Simplified {lang}. Output ONLY the translated {lang} text. Do not repeat the English.".format(lang=LANG)},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        buffer.append(text)
    else:
        user_prompt = user_prompt_template.format(
            en = entry["sentence"],
            direct = entry["direct"],
            think = entry["dpo_eval_8B"],
            rag_info = entry["rag_info"],
            )
        message = [
            {"role": "system", "content": system_prompt_template.format(lang=LANG)},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        buffer.append(text)

outputs = llm.generate(buffer, sampling_params=sampling_params)
assert len(data) == len(outputs), "The lists are not of the same length."
for entry, output in zip(data, outputs):
    entry["cot"] = output.outputs[0].text.strip()
    if "*** Final Translation ***" in entry["cot"]:
        text = entry["cot"].split("*** Final Translation ***")
        entry["cot"] = text[1]
        entry["cot"] = entry["cot"].replace("\n", "")
        entry["cot"] = entry["cot"].strip()
    else:
        entry["cot"] = entry["direct"]

with open(os.path.join(output_file_path, 'cot.json'), 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent = 2)
