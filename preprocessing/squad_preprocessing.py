import os
import json
from preprocessing import utils


def read_dict(dictionary):  # used for test
    print(type(dictionary))
    for key in dictionary:
        print(key)


def preprocessing(in_file, out_file, name):
    static_id = 1
    count_ans = 0  # answerable
    count_una = 0  # unanswerable
    with open(os.path.join(os.path.curdir, '../dataset', in_file)) as file:
        js_data = json.load(file)
        write_file = open(os.path.join(os.path.curdir, '../processed_dataset', out_file), 'w', encoding='utf-8')
        write_list = []
        js_data = js_data['data']
        for item in js_data:
            paragraphs = item['paragraphs']
            for paragraph in paragraphs:
                write_data = dict()
                write_data['context'] = paragraph['context']
                write_questions = []
                list_of_question = paragraph['qas']
                for qas in list_of_question:
                    write_qas = dict()
                    write_qas['question'] = qas['question']
                    write_qas['is_impossible'] = qas['is_impossible']
                    write_qas['id'] = '{} {}'.format(name, static_id)
                    write_qas['answers'] = []
                    write_qas['plausible_answers'] = []

                    def find_end_span(list_of_dict):
                        out_list = []
                        for a_dict in list_of_dict:
                            out_dict = dict()
                            out_dict['text'] = a_dict['text']
                            out_dict['start_span'] = a_dict['answer_start']
                            out_dict['end_span'] = out_dict['start_span'] + len(out_dict['text'])
                            out_list.append(out_dict)
                        return out_list

                    if write_qas['is_impossible']:
                        count_una += 1
                        write_qas['plausible_answers'] = find_end_span(qas['plausible_answers'])
                    else:
                        count_ans += 1
                        write_qas['answers'] = find_end_span(qas['answers'])
                    static_id += 1
                    write_questions.append(write_qas)
                write_data['questions'] = write_questions
                write_list.append(write_data)
        write_file.write(json.dumps(write_list, indent=1))
        write_file.close()
        print('{}: Unanswerable {}, Answerable {}'.format(name, count_una, count_ans))
        print('Context count: {}'.format(len(write_list)))


if __name__ == '__main__':
    preprocessing('train-squad2.0.json', 'train-squad2.0.json', 'squad2.0train')
    preprocessing('dev-squad2.0.json', 'dev-squad2.0.json', 'squad2.0dev')
    utils.data_argumentation('train-squad2.0.json', 'train-squad2.0.json', num=10000)
    # 只对训练集数据增强
