from openai import OpenAI
import json
from tqdm import tqdm

TARGET = "laj_translation"

input_file_path = "slang/eval/ende/LAJ/laj_all.json"
output_file_path = "slang/eval/ende/LAJ/laj_all_score.json"
result_file_path = "slang/eval/eval/LAJ/result.txt"


client = OpenAI(
    api_key="",
    base_url=""
)


def translate_text(messages):
    try:
        response = client.chat.completions.create(
            model="", 
            messages = messages,
            temperature=0,
        )
        # 提取翻译结果
        # print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error during translation: {e}")
        return None

translated_data = []
with open(input_file_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

data = []
for entry in all_data:
     if entry["label"] == "Figurative":
          data.append(entry)


#Q
user_prompt_template = '''
/* Task prompt */Please evaluate the translation quality of specific expressions from English sentences into German sentences. 

/* Evaluation Criteria */ /** Translation Quality Score **/
Based on the common meaning of the given expression, judge whether the translation conveys the actual intended meaning, tone, and nuance.

**Score from 1 to 10, applying a strict and wide distribution:**
* **Low Range (1-4):** For translations that are wrong, omitted, or overly literal (word-for-word) causing confusion.
* **Mid Range (5-7):** For translations that convey the basic meaning but lack idiomatic flavor or refinement.
* **High Range (8-10):** For translations that are not only accurate but also culturally adapted and elegant (信达雅).

**Important:** Do not hesitate to give extreme scores (1 or 10) to truly differentiate translation quality.

/* Test Data */Evaluate the following translation: 
English sentence: {source} 
Idiom in the sentence: {idiom} 
Idiom Sense: {sense} 
German translation: {translation} 

Evaluation (Output only the integer score between 1 and 10; do not output any other information):"
'''
for entry in tqdm(data):
        user_prompt = user_prompt_template.format(source = entry["sentence"], idiom = entry["idiom"], sense = entry["sense"], translation = entry[TARGET])
        messages=[
                {"role": "user", "content": user_prompt}     # User message with the sentence to translate
            ]

        score = translate_text(messages)
        
        if score:
            entry["q_score_"+TARGET] = score
#C
user_prompt_template = '''
/* Task prompt */Please evaluate the translation quality of specific expressions from English sentences into German sentences. 

/* Evaluation Criteria */ /** Contextual Consistency Score **/
Judge whether the translated expression is consistent with the meaning in the context and whether it integrates smoothly and logically into the sentence.

**Score from 1 to 10, applying a strict and wide distribution:**
* **Low Range (1-4):** The expression does not fit the context logically, breaks the sentence flow, or feels forcefully inserted (awkward or jarring).
* **Mid Range (5-7):** The sentence is generally fluent and conveys the meaning, but the integration of the expression feels slightly stiff or unnatural.
* **High Range (8-10):** The context is completely natural and coherent; the expression is embedded seamlessly and reads like a native sentence without any forced integration.

**Important:** Do not hesitate to give extreme scores (1 or 10) to truly differentiate how well the expression fits the context.

/* Test Data */Evaluate the following translation: 
English sentence: {source} 
Idiom in the English sentence: {idiom} 
Idiom Sense: {sense} 
German translation: {translation} 

Evaluation (Output only the integer score between 1 and 10; do not output any other information):"
'''
for entry in tqdm(data):
        user_prompt = user_prompt_template.format(source = entry["sentence"], idiom = entry["idiom"], sense = entry["sense"], translation = entry[TARGET])
        # print(user_prompt)
        messages=[
                {"role": "user", "content": user_prompt}     # User message with the sentence to translate
            ]

        score = translate_text(messages)
        
        if score:
            entry["c_score_"+TARGET] = score



with open(output_file_path, 'w', encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent = 2)
