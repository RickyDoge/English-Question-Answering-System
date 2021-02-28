#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 21:25:37 2021

@author: xiaoyu
"""

import re
import os
import json

# def preprocessing(in_file):
in_file = "dev-drop.json"
out_file = "dev-drop.json"
name = "drop_dev"
static_id = 1
impossible_Q_count = 0
possible_Q_count = 0
typeErr_count = 0

with open(os.path.join(os.path.curdir, '../dataset', in_file)) as file:
    js_data = json.load(file)
    write_file = open(os.path.join(os.path.curdir, '../processed_dataset', out_file), 'w', encoding='utf-8')
    write_list = []
    for item in js_data.values():
        write_data = dict()
        paragraph = item["passage"]
        write_data["context"] = paragraph
        write_questions = []
        qas = item["qa_pairs"]
        for qa in qas:
            write_qa = dict()
            write_qa["question"] = qa["question"]
            if len(qa["answer"]["spans"]) == 0:
                write_qa["is_impossible"] = True
                impossible_Q_count += 1
            else:
                write_qa["is_impossible"] = False
                write_answer = dict()
                answer_text = qa["answer"]["spans"][0]
                try:
                    answer_span = re.search(answer_text, paragraph).span()
                    write_answer["text"] = answer_text
                    write_answer["start_span"] = answer_span[0]
                    write_answer["end_span"] = answer_span[1]
                    possible_Q_count += 1
                except AttributeError:
                    write_qa["is_impossible"] = True
                    impossible_Q_count += 1
                except:
                    write_qa["is_impossible"] = True
                    typeErr_count += 1
                    
            write_qa["id"] = "{} {}".format(name, static_id)
            write_qa["answers"] = []
            write_qa["plausible_answers"] = []
            
            if write_qa["is_impossible"] == False:
                write_qa["answers"].append(write_answer)
                # validated_answers = qa["validated_answers"]
                # for a in validated_answers:
                #     if len(a["spans"])>0:
                #         write_answer = dict()
                #         answer_text = a["spans"][0]
                #         try:
                #             answer_span = re.search(answer_text, paragraph).span()
                #         except AttributeError:
                #             continue
                #         write_answer["text"] = answer_text
                #         write_answer["start_span"] = answer_span[0]
                #         write_answer["end_span"] = answer_span[1]
                #         write_qa["answers"].append(write_answer)
                        
            write_questions.append(write_qa)    
            static_id += 1
        write_data["questions"] = write_questions
        write_list.append(write_data)
    write_file.write(json.dumps(write_list, indent=1))
    write_file.close()
    
print("Answerable question count: {}".format(possible_Q_count))
print("Unanswerable question count: {}".format(impossible_Q_count))

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


# Q = "Before the UNPROFOR fully deployed, the HV clashed with an armed force of the RSK in the village of Nos Kalik, located in a pink zone near Sibenik, and captured the village at 4:45 p.m. on 2 March 1992. The JNA formed a battlegroup to counterattack the next day."
# print(re.search("2 March 1992", Q).span())