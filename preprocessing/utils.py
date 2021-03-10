import re
import os
import json
import random


def count_answerable_and_unanswerable(js_list):
    count_una = 0  # unanswerable
    count_ans = 0  # answerable
    for paragraph in js_list:
        for question in paragraph['questions']:
            if len(question) == 0:
                continue
            if question['is_impossible']:
                count_una += 1
            else:
                count_ans += 1
    return count_una, count_ans


def generate_unanswerable_context(context, text):
    """
    :param context: context
    :param text: answer text
    :return: context with no answer
    """
    return context.replace(text, '')


def generate_corrupted_context(original_context, unrelated_context):
    """
    :param original_context: original context
    :param unrelated_context: unrelated context
    :return: corrupted context
    """
    return original_context + unrelated_context


def generate_corrupted_data_list(in_file, num=500):
    """
    :param in_file: input json file. The format should be the the same as data-example.md
    :param num: generate how many corrupted context
    """
    random.seed(2020)
    with open(os.path.join(os.path.curdir, '../processed_dataset', in_file), encoding='utf-8') as file:
        js_list = json.load(file)
        count_una, count_ans = count_answerable_and_unanswerable(js_list)
        print('{} Original Unanswerable: {}, Answerable {}'.format(in_file, count_una, count_ans))
        for i in range(num):
            context = js_list[i]['context']
            questions = js_list[i]['questions']
            for question in questions:
                question['id'] += '-corrupted'
            unrelated_idx = random.randint(0, len(js_list) - 1)
            corrupted_context = generate_corrupted_context(context, js_list[unrelated_idx]['context'])
            new_dict = dict()
            new_dict['context'] = corrupted_context
            new_dict['questions'] = questions
            js_list.append(new_dict)
    return js_list


def generate_balance_dataset(js_list, out_file):
    random.seed(2020)
    with open(os.path.join(os.path.curdir, '../processed_dataset', out_file), 'r+', encoding='utf-8') as file:
        count_una, count_ans = count_answerable_and_unanswerable(js_list)
        print('{} Corrupted Unanswerable: {}, Answerable {}'.format(out_file, count_una, count_ans))
        for paragraph in js_list:
            if count_una >= count_ans:
                break
            unanswerable_paragraph = dict()
            unanswerable_paragraph['context'] = paragraph['context']
            unanswerable_paragraph['questions'] = []
            for question in paragraph['questions']:
                new_question = dict()
                if not question['is_impossible']:
                    new_question['question'] = question['question']
                    new_question['is_impossible'] = True
                    new_question['id'] = question['id'] + '-delAnswer'
                    new_question['answers'] = []
                    new_question['plausible_answers'] = []
                    for answer in question['answers']:
                        unanswerable_paragraph['context'] = generate_unanswerable_context(
                            unanswerable_paragraph['context'], answer['text'])
                    count_una += 1
                unanswerable_paragraph['questions'].append(new_question)
                if count_una >= count_ans:
                    break
            js_list.append(unanswerable_paragraph)
        json.dump(js_list, file, indent=1)
        print('{} Balanced Unanswerable: {}, Answerable {}'.format(out_file, count_una, count_ans))


def data_argumentation(in_file, out_file, num=500):
    """
    :param in_file: input data directory, in /processed_dataset
    :param out_file: output data directory, in /processed_dataset. Can be same with in_file
    :param num: generate how many corrupted context
    """
    generate_balance_dataset(generate_corrupted_data_list(in_file, num=num), out_file)
