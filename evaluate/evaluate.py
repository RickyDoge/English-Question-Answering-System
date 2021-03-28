import torch
import torch.utils.data as tud
import json
import os
import argparse
from transformers import ElectraTokenizerFast
from model import utils
from model.dataset import QuestionAnsweringDataset, QuestionAnsweringDatasetConfiguration, my_collate_fn
from model.baseline import BaselineModel
from model.intensive_reading_ca import IntensiveReadingWithCrossAttention
from model.intensive_reading_ma import IntensiveReadingWithMatchAttention
from model.intensive_reading_cnn import IntensiveReadingWithConvolutionNet
from functools import partial


def test_multi_task_learner(valid_iterator, model, device, tokenizer):
    question_answer_dict = dict()
    model.eval()
    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, _, _, _, question_id = data
            cls_out, start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                                      attention_mask=batch_encoding['attention_mask'].to(device),
                                                      token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                      )
            cls_out = torch.argmax(cls_out, dim=-1)  # batch_size
            start_pos = torch.argmax(start_logits, dim=-1)  # batch_size
            end_pos = torch.argmax(end_logits, dim=-1)  # batch_size
            for i, cls in enumerate(cls_out):
                if cls.item() == 0:  # answerable
                    start = start_pos[i].item()
                    end = end_pos[i].item()
                    answer = tokenizer.decode(batch_encoding['input_ids'][i][start + 1: end + 1])
                    question_answer_dict[question_id[i]] = answer
                else:
                    question_answer_dict[question_id[i]] = ''
    with open(os.path.join(os.path.curdir, 'eval.json'), 'w') as file:
        json.dump(question_answer_dict, file)


def test_multi_task_learner_2(valid_iterator, model, device, tokenizer):
    question_answer_dict = dict()
    model.eval()
    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, _, _, _, question_id = data
            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            cls_out, start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                                      attention_mask=batch_encoding['attention_mask'].to(device),
                                                      token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                      pad_idx=tokenizer.pad_token_id,
                                                      max_qus_length=max_qus_len,
                                                      max_con_length=max_con_len,
                                                      )
            cls_out = torch.argmax(cls_out, dim=-1)  # batch_size
            start_pos = torch.argmax(start_logits, dim=-1)  # batch_size
            end_pos = torch.argmax(end_logits, dim=-1)  # batch_size
            for i, cls in enumerate(cls_out):
                if cls.item() == 0:  # answerable
                    start = start_pos[i].item()
                    end = end_pos[i].item()
                    answer = tokenizer.decode(batch_encoding['input_ids'][i][start + 1: end + 1])
                    question_answer_dict[question_id[i]] = answer
                else:
                    question_answer_dict[question_id[i]] = ''
    with open(os.path.join(os.path.curdir, 'eval.json'), 'w') as file:
        json.dump(question_answer_dict, file)


def test_separate_learner(valid_iterator, sketch_model, intensive_model, device, tokenizer):
    question_answer_dict = dict()
    sketch_model.eval()
    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, _, _, _, question_id = data
            cls_out = sketch_model(batch_encoding['input_ids'].to(device),
                                   attention_mask=batch_encoding['attention_mask'].to(device),
                                   token_type_ids=batch_encoding['token_type_ids'].to(device),
                                   )
            cls_out = torch.argmax(cls_out, dim=-1)  # batch_size
            for i, cls in enumerate(cls_out):
                if cls.item() == 0:  # answerable
                    max_con_len, max_qus_len = utils.find_max_qus_con_length(
                        attention_mask=batch_encoding['attention_mask'],
                        token_type_ids=batch_encoding['token_type_ids'],
                        max_length=batch_encoding['input_ids'].size(1),
                    )
                    start_logits, end_logits = intensive_model(batch_encoding['input_ids'].to(device),
                                                               batch_encoding['attention_mask'].to(device),
                                                               batch_encoding['token_type_ids'].to(device),
                                                               pad_idx=tokenizer.pad_idx,
                                                               max_qus_length=max_qus_len,
                                                               max_con_length=max_con_len,
                                                               )
                    start = start_logits[i].item()
                    end = end_logits[i].item()
                    answer = tokenizer.decode(batch_encoding['input_ids'][i][start + 1: end + 1])
                    question_answer_dict[question_id[i]] = answer
                else:
                    question_answer_dict[question_id[i]] = ''
    with open(os.path.join(os.path.curdir, 'eval.json'), 'w') as file:
        json.dump(question_answer_dict, file)


def test_retro_reader_learner(valid_iterator, model, device, tokenizer, threshold=5.):
    # Threshold tuning:
    # cross-attention (-2, 0.796), (-4, 0.797), (-6, 0.794)
    # match-attention
    question_answer_dict = dict()
    model.eval()

    correct_count = 0.
    total_count = 0.
    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, is_impossibles, _, _, question_id = data
            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            cls_out, start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                                      attention_mask=batch_encoding['attention_mask'].to(device),
                                                      token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                      pad_idx=tokenizer.pad_token_id,
                                                      max_qus_length=max_qus_len,
                                                      max_con_length=max_con_len,
                                                      )
            score_has = torch.max(start_logits, dim=-1)[0] + torch.max(end_logits, dim=-1)[0]
            score_null = start_logits[:, 0] + end_logits[:, 0]
            # if larger, means more likely to be unanswerable
            score_diff = score_null - score_has
            score_ext = torch.logit(cls_out[:, 1]) - torch.logit(cls_out[:, 0])

            start_logits = torch.argmax(start_logits, dim=-1)  # batch_size
            end_logits = torch.argmax(end_logits, dim=-1)  # batch_size
            score = score_diff + score_ext  # batch_size
            # print(score)
            # print(is_impossibles.argmax(dim=-1))

            for i, start in enumerate(start_logits):
                if score[i] < threshold:  # answerable
                    end = end_logits[i].item()
                    answer = tokenizer.decode(batch_encoding['input_ids'][i][start + 1: end + 1])
                    question_answer_dict[question_id[i]] = answer
                    total_count += 1
                    if is_impossibles[i][0].item() == 1:
                        correct_count += 1
                else:
                    question_answer_dict[question_id[i]] = ''
                    total_count += 1
                    if is_impossibles[i][1].item() == 1:
                        correct_count += 1
    with open(os.path.join(os.path.curdir, 'eval.json'), 'w') as file:
        json.dump(question_answer_dict, file)

    return correct_count / total_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='which config')
    args = parser.parse_args()
    config = args.config

    CONFIG = ['cross-attention', 'match-attention', 'cnn-span', 'baseline', 'cnn-span-large', 'cross-attention-large']
    assert config in CONFIG, 'Given config wrong'

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
    config_valid = QuestionAnsweringDatasetConfiguration(squad_dev=True)
    dataset_valid = QuestionAnsweringDataset(config_valid, tokenizer=tokenizer)
    dataloader_valid = tud.DataLoader(dataset=dataset_valid, batch_size=4, shuffle=False, drop_last=False,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))
    if config == 'cross-attention':
        retro_reader_model = IntensiveReadingWithCrossAttention()
        ts = -1.  # normal: -4 / lr: -1 / dwa: -1
    elif config == 'match-attention':
        retro_reader_model = IntensiveReadingWithMatchAttention()
        ts = -1.  # dwa: -1
    elif config == 'cnn-span':
        retro_reader_model = IntensiveReadingWithConvolutionNet(out_channel=100, filter_size=3)
        ts = -1.
        # 8 channels: -4 / 16 channels: -1 / 48 channels: -1 (DWA -5)
        # 100 channels: -1 (DWA -1)
    elif config == 'baseline':
        retro_reader_model = BaselineModel()
        ts = -1.
    elif config == 'cnn-span-large':
        retro_reader_model = IntensiveReadingWithConvolutionNet(out_channel=100, filter_size=3, hidden_dim=768,
                                                                clm_model='google/electra-base-discriminator')
        ts = -1.
    elif config == 'cross-attention-large':
        retro_reader_model = IntensiveReadingWithCrossAttention(hidden_dim=768,
                                                                clm_model='google/electra-base-discriminator')
        ts = -1.
    else:
        raise Exception('Wrong config error')

    retro_reader_model.load_state_dict(torch.load(os.path.join('..', 'single_gpu_model.pth')))
    retro_reader_model.to(device)
    if config == 'baseline':
        cls_acc = test_multi_task_learner(iter(dataloader_valid), retro_reader_model, device, tokenizer)
    else:
        # cls_acc = test_retro_reader_learner(iter(dataloader_valid), retro_reader_model, device, tokenizer, threshold=ts)
        cls_acc = test_multi_task_learner_2(iter(dataloader_valid), retro_reader_model, device, tokenizer)
    print("CLS accuracy: {}".format(cls_acc))
