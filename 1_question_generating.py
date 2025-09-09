import io
import os
import re
import csv
import time
import json
import logging
import base64
import datetime
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from PIL import Image, ImageOps

client_openai = OpenAI(
    api_key="key",
    base_url="url",
)
client_qwen =  OpenAI(
    api_key="key",
    base_url="url",
)

def get_question_from_llm(client, question, model):
    while True:
        try:
            messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            completion = client.chat.completions.create(
                model = model,
                temperature = 0,
                messages = messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            print(f"API Error: {error_message}")

            if "4" in error_message:
                time.sleep(10)
                return None 

            else:
                raise

df_chunk = pd.read_excel('knowledge_unit.xlsx', sheet_name="Sheet1")
df_chunk = df_chunk[['unit']].dropna(subset=['unit'])


MODEL = "gpt-4o"
client = client_qwen if 'qwen' in MODEL else client_openai

results = [] 

for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
    q_chunk = row['unit']              

    prompt = f"""Please refer to the following example and generate three questions for each knowledge unit that will be presented.
    
Example：
- Question type: single_choice
- Knowledge unit 1: 标准设计施工总承包招标文件。第四章 合同条款及格式。第一节  通用合同条款。5. 设计。5.3 设计审查。5.3.3 设计文件需政府有关部门审查或批准的，发包人应在审查同意承包人的设计文件后7天内，向政府有关部门报送设计文件，承包人应予以协助。对于政府有关部门的审查意见，不需要修改发包人要求的，承包人需按该审查意见修改承包人的设计文件；需要修改发包人要求的，发包人应重新提出发包人要求，承包人应根据新提出的发包人要求修改承包人文件。上述情形还应适用第15条、第1.13款的有关约定。政府有关部门审查批准的，承包人应当严格按照批准后的承包人的设计文件设计和实施工程。
- Question 1: 根据《标准设计施工总承包招标文件》，设计文件需政府有关部门审查或批准的工程、发包人应在审查同意承包人的设计文件（   ）天内，向政府有关部门应送设计文件。
A. 7
B. 14
C. 21
D. 28

- Question type: single_choice
- Knowledge unit 2: 标准设计施工总承包招标文件。第四章 合同条款及格式。第一节  通用合同条款。1. 一般约定。1.4 合同文件的优先顺序。组成合同的各项文件应互相解释，互为说明。除专用合同条款另有约定外，解释合同文件的优先顺序如下：
（1）合同协议书；
（2）中标通知书；
（3）投标函及投标函附录；
（4）专用合同条款；
（5）通用合同条款；
（6）发包人要求；
（7）承包人建议书； 
（8）价格清单；
（9）其他合同文件。
- Question 2: 根据《标准设计施工总承包招标文件》，合同文件包括：①承包人建议书;②中标通知书;③合同协议书。仅就上述组成文件而言，正确的优先解释顺序为()。
A.①一②一③
B.③一①一②
C.①一③一②
D.③一②一①

- Question type: multiple_choice
- Knowledge unit 3: 标准施工招标文件。第四章 合同条款及格式。第一节 通用合同条款。11. 开工和竣工。11.3 发包人的工期延误。
在履行合同过程中，由于发包人的下列原因造成工期延误的，承包人有权要求发包人延长工期和（或）增加费用，并支付合理利润。需要修订合同进度计划的，按照第10.2款的约定办理。
（1）增加合同工作内容；
（2）改变合同中任何一项工作的质量要求或其他特性；
（3）发包人迟延提供材料、工程设备或变更交货地点的；
（4）因发包人原因导致的暂停施工；
（5）提供图纸延误；
（6）未按合同约定及时支付预付款、进度款；
（7）发包人造成工期延误的其他原因。
- Question 3: 根据《标准施工招标文件》，应归属于发包人原因且承包人有权获得工期顺延的情况有（）
A. 增加合同工作内容
B. 改变合同中任何一项工作的质量要求
C. 合同中某项工作的施工质量不满足要求
D. 提供设计图纸延误
E. 不利气候条件造成的影响

- Question type: single_choice
- Knowledge unit 4: 标准设计施工总承包招标文件。第四章 合同条款及格式。第一节  通用合同条款。17. 合同价格与支付。17.3 工程进度付款。17.3.4 进度付款证书和支付时间。（1）监理人在收到承包人进度付款申请单以及相应的支持性证明文件后的14天内完成审核，提出发包人到期应支付给承包人的金额以及相应的支持性材料，经发包人审批同意后，由监理人向承包人出具经发包人签认的进度付款证书。监理人未能在前述时间完成审核的，视为监理人同意承包人进度付款申请。监理人有权核减承包人未能按照合同要求履行任何工作或义务的相应金额。（2）发包人最迟应在监理人收到进度付款申请单后的28 天内，将进度应付款支付给承包人。发包人未能在前述时间内完成审批或不予答复的，视为发包人同意进度付款申请。发包人不按期支付的，按专用合同条款的约定支付逾期付款违约金。（3）监理人出具进度付款证书，不应视为监理人已同意、批准或接受了承包人完成的该部分工作。（4）进度付款涉及政府投资资金的，按照国库集中支付等国家相关规定和专用合同条款的约定执行。
- Question 4: 某工程实施过程中，监理人于2023年3月3日收到承包人提交的2月份进度付款中请单及支持性证明文性.并于3月10日完般审核。根据《标准设计施工总承包招标文件》，发包人向承包人支付该笔进度款的最迟时间应是2023年()
A.4月9日
B.4月7日
C.3月31日
D.3月17日


Generation rules:
(1) Each generated question should be relevant to the corresponding unit.
(2) The answer must be retrievable from the content of that unit.
(3) The set of questions must include at least one single-choice and one multiple-choice question without duplication.

Knowledge unit: {knowledge_unit}”.
"""
    

    output = get_question_from_llm(client, prompt, MODEL)

    if output is None:
        continue

    blocks = [b.strip() for b in output.strip().split("\n\n") if b.strip()]
    questions = []

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue

        try:
            q_type = re.search(r"type：(.+)", lines[0]).group(1).strip()
            q_stem = re.search(r"question\d+：(.+)", lines[1]).group(1).strip()
            options = [line.strip() for line in lines[2:] if re.match(r"^[A-E]\.", line.strip())]

            questions.append({
                "type": q_type,
                "stem": q_stem,
                "option": options
            })
        except Exception as e:
            print(f"fail：{e}\ncontent：{block}\n")
            continue

    row_data = [q_chunk]
    for q in questions:
        row_data.append(q["type"])
        row_data.append(q["stem"])
        row_data.append("\n".join(q["option"]))
    
    max_question_count = max(max_question_count, len(questions))
    all_rows.append(row_data)

columns = ["unit"]
for i in range(max_question_count):
    columns.extend([f"question{i+1}type", f"question{i+1}stem", f"question{i+1}option"])

for i in range(len(all_rows)):
    while len(all_rows[i]) < len(columns):
        all_rows[i].append("")


df_out = pd.DataFrame(all_rows, columns=columns)
output_path = "output_folder/question_generating_for_finetuning_embedding_model.xlsx"
df_out.to_excel(output_path, index=False)



