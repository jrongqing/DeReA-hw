import json
from comet import download_model, load_from_checkpoint
import subprocess
import datetime

model_path = download_model("Unbabel/wmt22-comet-da")
input_file_path = "slang/translate/gpt5mini/top1_fr.json"
output_file = "slang/eval/eval/result/result.txt"
model = load_from_checkpoint(model_path)

SOURCE_KEY= "sentence"
#********************
# dpo_translation
TARGET_KEY = "fr_cot"
# TARGET_KEY = "direct_8b"
# REFERENCE_KEY = "reference_fr"
REFERENCE_KEY = "reference_fr"


with open(input_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

input_data_Figurative = []
input_data_Without_Idiom = []
input_data_Avg = []

for entry in data:
    if entry["label"] == "Figurative":
        s = {
                "src":entry[SOURCE_KEY],
                "mt":entry[TARGET_KEY],
                "ref":entry[REFERENCE_KEY]
            }
        input_data_Figurative.append(s)
    else:
        s = {
                "src":entry[SOURCE_KEY],
                "mt":entry[TARGET_KEY],
                "ref":entry[REFERENCE_KEY]
            }
        input_data_Without_Idiom.append(s)

input_data_Avg = input_data_Figurative+input_data_Without_Idiom


# print()
model_output = model.predict(input_data_Avg, batch_size=8, gpus=1)
Avg_score = model_output.system_score


model_output = model.predict(input_data_Figurative, batch_size=8, gpus=1)
Figurative_score = model_output.system_score


model_output = model.predict(input_data_Without_Idiom, batch_size=8, gpus=1)
Without_Idiom_score = model_output.system_score
print(Avg_score)
print(Figurative_score)
print(Without_Idiom_score)

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


with open(output_file, 'a') as f:
    f.write(f"{current_time}\n")  
    f.write(f"TARGET: {TARGET_KEY}\n")  
    f.write(f"LABEL: Avg\n")
    f.write(f"comet score: {Avg_score}\n")  
    f.write(f"LABEL: Figurative\n")
    f.write(f"comet score: {Figurative_score}\n")  
    f.write(f"LABEL: Without_Idiom\n")
    f.write(f"comet score: {Without_Idiom_score}\n")  
    f.write("\n") 

