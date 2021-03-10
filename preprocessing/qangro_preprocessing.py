import csv
import pandas as pd
import re
import os
import json
import gzip


in_contfile = r"C:\Users\qianp\Desktop\qangaroo_v1.1\medhop\dev.json"#path to wikihop/medhop data file

out_file = r"C:\Users\qianp\Desktop\qangaroo_v1.1\medhop\dev-medhop.json"
name = "medhop_dev"
static_id = 1
impossible_Q_count = 0
possible_Q_count = 0
typeErr_count = 0

write_list=[]
with open(in_contfile,"r") as file:
    js_data=json.load(file)
    write_file = open(os.path.join(out_file), 'w', encoding='utf-8')
    for item in js_data:
        write_data=dict()
        paragraph = item["supports"]
        write_data["context"] = str(paragraph[0])

        write_questions = []
        qas = item["query"]

        write_answer=dict()
        answer_text = item["answer"]


        write_qa = dict()
        write_qa['id'] = '{} {}'.format(name, static_id)
        write_qa["question"] = qas
        write_qa['plausible_answers'] = []

        write_qa["answers"]=[]
        answer_search=re.search(str(answer_text), str(paragraph),re.IGNORECASE)

        if not answer_search==None:
            answer_span = answer_search.span()
            write_qa["is_impossible"] = False
            answer_span = answer_search.span()

            write_answer["text"] = answer_text
            write_answer["start_span"] = answer_span[0]
            write_answer["end_span"] = answer_span[1]
            possible_Q_count += 1
        else:
            write_qa["is_impossible"] = True
            impossible_Q_count += 1

        write_qa["answers"].append(write_answer)
        write_questions.append(write_qa)
        static_id += 1
        write_data["questions"]=write_questions
        write_list.append(write_data)

   # write_file.write(json.dumps(write_list, indent=1))
    json.dump(write_list, write_file,  indent=1)
    write_file.close()

print(impossible_Q_count / (impossible_Q_count + possible_Q_count))
