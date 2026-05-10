import os
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import torch
import json

MODEL_SIZE = "8B"
PARALLEL_SIZE = 1

MO = "DPO"
N = 645


input_file_path = '/data0/jrq/slang/detect/eval/LoML-CN-deal.json'
output_file_path = 'eval\detect\result\LoML-result.json'
model_path = 'model\detect\qwen_dpo'

tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0, top_p=0.95, top_k=50, max_tokens=2048)
llm = LLM(model=model_path, tensor_parallel_size=PARALLEL_SIZE)


system_template = '''
# 核心方法

你将扮演一名严谨的语言分析师。你的核心方法论是：通过“字面含义”与“语境推断含义”的逻辑对比，来客观判断一个表达是否为“非常规表达”。

# 关键定义：非常规表达 (Key Definition: Unconventional Expression)

“非常规表达” (Unconventional Expression) 指的是一个词汇或短语，其整体含义不能从其组成单词的字面意思直接推断得出。这种表达的意义通常是比喻性的、惯用的、或具有特定文化背景的，并且高度依赖于上下文。它主要包括习语 (idioms)、俚语 (slang) 和隐喻 (metaphors) 等。

# 具体要求
1.语句中的非常规表达可能包含人称，时态，缩写等变化，请从中识别出原始形态
2.常规用法中的非常规表达也纳入考虑范围
3.字面含义和比喻含义都解释的通时，应结合语境选择最佳的解释来进行判断
'''

user_prompt_template = '''
包含非常规表达的示例句：
"When the share market crashed his fingers were burnt from all the investments that he had made ."

输出示例：

***Thought process:
“his fingers were burnt”：字面意思是“他的手指被烧伤”，但原句中语境为“股市崩盘时，他因投资不当而导致了'his fingers were burnt'...”，"burn fingers"用来表达主语“他”因为自己不恰当的金融投资而受到的损失，字面含义在语境中很显然不符合，因此将其定义为非常规表达，供后续词典查阅；
其余表达如the share market crashed（股市崩盘），the investments（投资）等均符合字面含义。

***Output: ## YES ## 1.burn fingers


不包含非常规表达的示例句：
“She is reading a book.”（她正在读一本书。）

输出示例：
***Thought process:  
该句结构清晰，reading a book等都是常见短语，没有隐藏含义。句中各个表达均为单词本身的字面意义，没有任何非常规表达。

输出（Output）: ## NO


请根据系统提供的方法，并参考以上示例，处理以下英文句子中的非常规表达。
英文文本：{sentence}

思考过程与输出（Thought process & Output）

请严格遵循以下输出格式：
***Thought process:  
***Output: ## YES ##<非常规表达>

或

***Thought process:  
***Output: ## NO
'''

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

buffer = []

for entry in data:
    # if entry["usage"] == "figurative":
    #     user_prompt_template = user_prompt_template_figurative.format(sentence = entry["sentence"], split = entry["split"])
    # else:
    #     user_prompt_template = user_prompt_template_literal.format(sentence = entry["sentence"], idiom = entry["idiom"])
    user_prompt = user_prompt_template.format(sentence = entry["sentence"])
    message = [
        {"role": "system", "content": system_template},
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

    idiom_count = 0
    without_idiom_count = 0
    

    for entry, output in zip(data, outputs):
        if "Output: ## YES" in output.outputs[0].text.strip():
            entry["detect_label"] = "Figurative"
        else:
            entry["detect_label"] = "Without_Idiom"

        if entry["label"] == "Figurative" and "Output: ## YES" in output.outputs[0].text.strip():
            idiom_count += 1
        if entry["label"] == "Without_Idiom" and "Output: ## NO" in output.outputs[0].text.strip():
            without_idiom_count += 1
        entry["dpo_eval_"+MODEL_SIZE] = output.outputs[0].text.strip()
    
    idiom_count_percent = idiom_count / N * 100
    without_idiom_count_percent = without_idiom_count / N * 100
    print(f"非常规表达存在且被正确识别的数量: {idiom_count}，占比: {idiom_count_percent:.2f}%")
    print(f"不包含非常规表达且被正确识别的数量: {without_idiom_count}，占比: {without_idiom_count_percent:.2f}%")

    TP = idiom_count
    TN = without_idiom_count
    FN = N - TP
    FP = N - TN
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1-score: {f1:.4f}")



    result_data = []
    for entry in data:
        if entry["label"] == "Figurative" or entry["label"] == "Without_Idiom":
            result_data.append(entry)


    with open(output_file_path, 'w', encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent = 4)

    print(f"翻译结果已保存至: {output_file_path}")


if __name__ == "__main__":
    main()
    

#  CUDA_VISIBLE_DEVICES=0 python eval\detect\detect.py