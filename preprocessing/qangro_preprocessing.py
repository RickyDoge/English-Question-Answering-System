import csv
import pandas as pd
import re
import os
import json
import gzip


in_contfile = r"C:\Users\qianp\Desktop\qangaroo_v1.1\wikihop\train.json"

out_file = r"C:\Users\qianp\Desktop\qangaroo_v1.1\wikihop\procesed_train.json"
name = "wikihop_train"
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
        write_data["context"] = paragraph
        write_questions = []
        qas = item["query"]

        write_answer=dict()
        answer_text = item["answer"]


        write_qa = dict()
        write_qa['id'] = '{} {}'.format(name, static_id)
        write_qa["question"] = qas
        write_qa['plausible_answers'] = []

        answer_search=re.search(str(answer_text), str(paragraph),re.IGNORECASE)

        if not answer_search==None:
            answer_span = answer_search.span()
            write_qa["is_impossible"] = False
            answer_span = answer_search.span()

            write_answer["text"] = answer_text
            write_answer["start_span"] = answer_span[0]
            write_answer["end_span"] = answer_span[1]
            write_qa["answer"] = write_answer
            possible_Q_count += 1
        else:
            write_qa["is_impossible"] = True
            impossible_Q_count += 1


        write_questions.append(write_qa)
        static_id += 1
        write_data["questions"]=write_questions
        write_list.append(write_data)

   # write_file.write(json.dumps(write_list, indent=1))
    json.dump(write_list, write_file,  indent=1)
    write_file.close()

print(impossible_Q_count / (impossible_Q_count + possible_Q_count))

        #answers = list(ijson.items(file, 'answer.item'))



"""
    # print(js_data["data"])
    write_file = open(os.path.join(os.path.curdir, out_file), 'w', encoding='utf-8')
    write_list = []
    for item in js_data:
        write_data = dict()
        paragraph = item["text"]
        write_data["context"] = paragraph
        write_questions = []
        qas = item["questions"]
        for qa in qas:
            write_qa = dict()
            write_qa["question"] = qa["q"]
            write_qa['id'] = '{} {}'.format(name, static_id)
            write_qa['answers'] = []
            write_qa['plausible_answers'] = []

            if "s" in qa["consensus"].keys() and "e" in qa["consensus"].keys():
                write_qa["is_impossible"] = False
                write_answer = dict()
                answer_start = qa["consensus"]["s"]
                answer_end = qa["consensus"]["e"]
                answer_text = write_data["context"][answer_start:answer_end]

                write_answer["text"] = answer_text
                write_answer["start_span"] = answer_start
                write_answer["end_span"] = answer_end
                write_qa["answers"].append(write_answer)
                possible_Q_count += 1
            else:
                write_qa["is_impossible"] = True
                impossible_Q_count += 1

            write_questions.append(write_qa)
            static_id += 1
        write_data["questions"] = write_questions
        write_list.append(write_data)
    write_file.write(json.dumps(write_list, indent=1))
    write_file.close()
    """
#print(impossible_Q_count / (impossible_Q_count + possible_Q_count))
