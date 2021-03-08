import torch
import torch.utils.data as tud
import os
import json


class QuestionAnsweringDatasetConfiguration():
    def __init__(self, squad_train=False, squad_dev=False, drop_train=False, drop_dev=False, newsqa_train=False,
                 newsqa_dev=False, medhop_dev=False, medhop_train=False, quoref_dev=False, quoref_train=False,
                 wikihop_dev=False, wikihop_train=False):
        exist_true = squad_train or squad_dev or drop_train or drop_dev or newsqa_train or newsqa_dev
        if not exist_true:
            raise Exception('Please give at least one dataset.')
        self.squad_train = squad_train
        self.squad_dev = squad_dev
        self.drop_train = drop_train
        self.drop_dev = drop_dev
        self.newsqa_train = newsqa_train
        self.newsqa_dev = newsqa_dev
        self.medhop_dev = medhop_dev
        self.medhop_train = medhop_train
        self.quoref_dev = quoref_dev
        self.quoref_train = quoref_train
        self.wikihop_dev = wikihop_dev
        self.wikihop_train = wikihop_train

    def read_files_list(self):
        files_dir = []
        main_dir = os.path.join(os.path.curdir, 'processed_dataset')
        if self.squad_train:
            files_dir.append(os.path.join(main_dir, 'train-squad2.0.json'))
        if self.squad_dev:
            files_dir.append(os.path.join(main_dir, 'dev-squad2.0.json'))
        if self.drop_train:
            files_dir.append(os.path.join(main_dir, 'train-drop.json'))
        if self.drop_dev:
            files_dir.append(os.path.join(main_dir, 'dev-drop.json'))
        if self.newsqa_train:
            files_dir.append(os.path.join(main_dir, 'newsqa_train.json'))
        if self.newsqa_dev:
            files_dir.append(os.path.join(main_dir, 'newsqa_dev.json'))
        if self.medhop_train:
            files_dir.append(os.path.join(main_dir, 'train-medhop.json'))
        if self.medhop_dev:
            files_dir.append(os.path.join(main_dir, 'dev-medhop.json'))
        if self.quoref_train:
            files_dir.append(os.path.join(main_dir, 'train-quoref.json'))
        if self.quoref_dev:
            files_dir.append(os.path.join(main_dir, 'dev-quoref.json'))
        if self.wikihop_train:
            files_dir.append(os.path.join(main_dir, 'train-wikihop.json'))
        if self.wikihop_dev:
            files_dir.append(os.path.join(main_dir, 'dev-wikihop.json'))
        return files_dir


class QuestionAnsweringDataset(tud.Dataset):
    def __init__(self, configuration: QuestionAnsweringDatasetConfiguration, tokenizer):
        super(QuestionAnsweringDataset, self).__init__()
        global_dataset = []
        for file_dir in configuration.read_files_list():  # read file
            file = open(file_dir, encoding='utf-8')
            js_list = json.load(file)
            global_dataset += js_list
            file.close()

        self.tokenizer = tokenizer
        self.contexts_list = []
        self.questions_list = []
        for paragraph in global_dataset:  # store data
            self.contexts_list.append(paragraph['context'])
            context_idx = len(self.contexts_list) - 1
            for question in paragraph['questions']:
                if len(question) != 0:
                    self.questions_list.append((question, context_idx))

    def __len__(self):
        return len(self.questions_list)

    def __getitem__(self, idx):
        question, context_idx = self.questions_list[idx]
        return question['question'], self.contexts_list[context_idx], question['is_impossible'], question['answers'], \
               question['id']


def my_collate_fn(batch, tokenizer):
    batch_split_list = list(zip(*batch))
    questions, contexts, is_impossibles, answers_batch, id = list(batch_split_list[0]), list(batch_split_list[1]), list(
        batch_split_list[2]), list(batch_split_list[3]), list(batch_split_list[4])

    # tokenization and word2idx
    batch_encoding = tokenizer(contexts, questions, return_tensors='pt', padding=True, truncation=True)
    # answer span vector
    start_position, end_position = char_span_to_token_span(batch_encoding, answers_batch, is_impossibles,
                                                           max_length=tokenizer.model_max_length)

    # is_impossible vector
    is_impossibles = torch.tensor([[0, 1] if is_impossible else [1, 0] for is_impossible in is_impossibles]).float()

    return batch_encoding, is_impossibles, start_position, end_position, id


def char_span_to_token_span(batch_encoding, answers_batch, is_impossibles, max_length=512):
    # batch_size
    start_position = torch.zeros(batch_encoding['input_ids'].size(0))
    end_position = torch.zeros(batch_encoding['input_ids'].size(0))

    for batch_idx, answers in enumerate(answers_batch):
        if is_impossibles[batch_idx]:
            continue
        for j, answer in enumerate(answers):
            # convert char idx to token idx
            start_span = batch_encoding.char_to_token(batch_idx, answer['start_span'])
            end_span = batch_encoding.char_to_token(batch_idx, answer['end_span'])
            if end_span is None:
                end_span = batch_encoding.char_to_token(batch_idx, answer['end_span'] + 1)  # add one space
                if end_span is None:
                    end_span = batch_encoding.char_to_token(batch_idx, answer['end_span'] + 2)  # add two space
            # write to tensor
            if start_span is not None:
                start_position[batch_idx] = start_span
            if end_span is not None:
                end_position[batch_idx] = end_span

    return start_position.long(), end_position.long()
