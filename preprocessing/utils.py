import re
import os
import json
import random


def generate_answerable_question(context, start_span, end_span):
    """
    :param context: context
    :param start_span: start position (given by character index)
    :param end_span: end position (given by character index)
    :return: context with no answer
    """
    if start_span > 1:
        return context[:start_span - 1] + context[end_span:]
    else:
        return context[end_span:]


def generate_corrupted_context(original_context, unrelated_context):
    """
    :param original_context: original context
    :param unrelated_context: unrelated context
    :return: corrupted context
    """
    return original_context + unrelated_context


def generate_corrupted_dataset(in_file, num=500):
    random.seed(2020)
    with open(os.path.join(os.path.curdir, '../processed_dataset', in_file), 'r+') as file:
        js_list = json.load(file)
        for i in range(num):
            context = js_list[i]['context']
            questions = js_list[i]['questions']
            for question in questions:
                question['id'] += '-corrupted'
            unrelated_idx = random.randint(0, len(js_list) - 1)
            corrupted_context = generate_corrupted_context(context, js_list[unrelated_idx]['context'])
            js_list.append({'context': corrupted_context, 'questions': questions})
        random.shuffle(js_list)
        file.write(json.dumps(js_list, indent=1))


def generate_balance_dataset(in_file):
    return 0
