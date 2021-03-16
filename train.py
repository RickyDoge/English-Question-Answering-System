import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.optim as optim
import os
import logging
import argparse
from train_baseline import test
from evaluate.evaluate import test_multi_task_learner
from transformers import ElectraTokenizerFast
from model import utils
from model.intensive_reading_ca import IntensiveReadingWithCrossAttention
from model.intensive_reading_ma import IntensiveReadingWithMatchAttention
from model.dataset import my_collate_fn, QuestionAnsweringDatasetConfiguration, QuestionAnsweringDataset
from functools import partial


def test_intensive_reader(valid_iterator, model, device, pad_idx):
    model.eval()
    start_end_loss = nn.CrossEntropyLoss()

    loss_sum = 0  # loss
    loss_count = 0

    f1_sum = 0  # F1-score
    f1_count = 0

    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, _, start_position, end_position, _ = data
            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)
            start_position = torch.where(start_position >= 1, start_position - 1, start_position)
            end_position = torch.where(end_position >= 1, end_position - 1, end_position)
            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                             attention_mask=batch_encoding['attention_mask'].to(device),
                                             token_type_ids=batch_encoding['token_type_ids'].to(device),
                                             pad_idx=pad_idx,
                                             max_qus_length=max_qus_len,
                                             max_con_length=max_con_len,
                                             )
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)

            span_loss = (start_loss + end_loss) / 2

            loss_sum += span_loss.item()
            loss_count += 1

            predict_start = torch.argmax(start_logits, dim=-1)
            predict_end = torch.argmax(end_logits, dim=-1)

            predict_start = predict_start.cpu().numpy()
            predict_end = predict_end.cpu().numpy()
            start_position = start_position.cpu().numpy()
            end_position = end_position.cpu().numpy()
            for ps, pe, rs, re in zip(predict_start, predict_end, start_position, end_position):
                recall = utils.calculate_recall(ps, pe, rs, re)
                precision = utils.calculate_recall(rs, re, ps, pe)
                f1_sum += (recall + precision) / 2
                f1_count += 1

    model.train()
    return loss_sum / loss_count, f1_sum / f1_count


def test_sketch_reader(valid_iterator, model, device):
    model.eval()
    cls_loss = nn.BCELoss()
    loss_sum = 0  # loss
    loss_count = 0

    cls_correct_count = 0  # is impossible
    cls_total_count = 0

    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, is_impossibles, _, _, _ = data
            is_impossibles = utils.move_to_device(is_impossibles, device)
            cls_out = model(batch_encoding['input_ids'].to(device),
                            attention_mask=batch_encoding['attention_mask'].to(device),
                            token_type_ids=batch_encoding['token_type_ids'].to(device),
                            )
            impossible_loss = cls_loss(cls_out, is_impossibles)

            loss_sum += impossible_loss.item()
            loss_count += 1

            cls_out = torch.argmax(cls_out, dim=-1)
            cls_out = (cls_out == is_impossibles.argmax(dim=-1)).float()
            cls_correct_count += torch.sum(cls_out).item()
            cls_total_count += cls_out.size(0)

    model.train()
    return loss_sum / loss_count, cls_correct_count / cls_total_count


def main(epoch=4, which_config='cross-attention', which_dataset='small', multitask_weight=1.0, seed=2020):
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)

    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # load configuration
    hidden_dim = 256
    which_model = 'google/electra-small-discriminator'

    # load dataset
    tokenizer = ElectraTokenizerFast.from_pretrained(which_model)
    if which_dataset == 'small':
        config_train = QuestionAnsweringDatasetConfiguration(squad_train=True)
        config_valid = QuestionAnsweringDatasetConfiguration(squad_dev=True)
    else:
        config_train = QuestionAnsweringDatasetConfiguration(squad_train=True, squad_dev=False, drop_train=True,
                                                             drop_dev=True, newsqa_train=True, newsqa_dev=True,
                                                             medhop_dev=True, medhop_train=True, quoref_dev=True,
                                                             quoref_train=True, wikihop_dev=True, wikihop_train=True)
        config_valid = QuestionAnsweringDatasetConfiguration(squad_dev=True)
    dataset_train = QuestionAnsweringDataset(config_train, tokenizer=tokenizer)
    dataset_valid = QuestionAnsweringDataset(config_valid, tokenizer=tokenizer)
    dataloader_train = tud.DataLoader(dataset=dataset_train, batch_size=48, shuffle=True, drop_last=True,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))
    dataloader_valid = tud.DataLoader(dataset=dataset_valid, batch_size=48, shuffle=False, drop_last=True,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))

    # load pre-trained model
    if which_config == 'cross-attention':
        intensive_model = IntensiveReadingWithCrossAttention(clm_model=which_model, hidden_dim=hidden_dim)
    else:
        intensive_model = IntensiveReadingWithMatchAttention(clm_model=which_model, hidden_dim=hidden_dim)
    intensive_model.train()

    # GPU Config:
    if torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        intensive_model.to(device)
        intensive_model = nn.DataParallel(module=intensive_model)
        print('Use Multi GPUs. Number of GPUs: ', torch.cuda.device_count())
    elif torch.cuda.device_count() == 1:
        device = torch.cuda.current_device()
        intensive_model.to(device)
        print('Use 1 GPU')
    else:
        device = torch.device('cpu')  # CPU
        print("use CPU")

    if torch.cuda.device_count() > 1:
        optimizer = optim.Adam(
            [{'params': intensive_model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.module.Hq_proj.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.module.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             ] if config == 'match-attention' else
            [{'params': intensive_model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.module.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.module.attention.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.module.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             ]
        )
    else:
        optimizer = optim.Adam(
            [{'params': intensive_model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.Hq_proj.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             ] if config == 'match-attention' else
            [{'params': intensive_model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.attention.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             {'params': intensive_model.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
             ]
        )

    cls_loss = nn.BCELoss()  # Binary Cross Entropy Loss
    start_end_loss = nn.CrossEntropyLoss()

    best_score = 0.25

    for e in range(epoch):
        for i, data in enumerate(iter(dataloader_train)):
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            intensive_model.train()
            is_impossibles = utils.move_to_device(is_impossibles, device)
            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)

            # minus one, because we removed [CLS] when utils.generate_question_and_passage_hidden
            start_position = torch.where(start_position > 1, start_position - 1, start_position)
            end_position = torch.where(end_position > 1, end_position - 1, end_position)
            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            cls_output, start_logits, end_logits = intensive_model(batch_encoding['input_ids'].to(device),
                                                                   attention_mask=batch_encoding['attention_mask']
                                                                   .to(device),
                                                                   token_type_ids=batch_encoding['token_type_ids']
                                                                   .to(device),
                                                                   pad_idx=tokenizer.pad_token_id,
                                                                   max_qus_length=max_qus_len,
                                                                   max_con_length=max_con_len,
                                                                   )
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            answerable_loss = cls_loss(cls_output, is_impossibles)
            printable = (((start_loss + end_loss) / 2).item(), answerable_loss.item())
            loss = (start_loss + end_loss) / 2 + answerable_loss * multitask_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                logger.info('Epoch {}, Iteration {}, Span Loss: {:.4f}, Ans Loss{:.4f}'.format(e, i, printable[0],
                                                                                               printable[1]))
                v_loss_intensive, acc, f1 = test(iter(dataloader_valid), intensive_model, device)
                logger.info('Epoch {}, Iteration {}, Intensive valid loss {:.4f}, CLS acc{:.4f}, F1-score {:.4f}'
                            .format(e, i, v_loss_intensive, acc, f1))
                score = acc * f1
                if score >= best_score:  # save the best model
                    best_score = score
                    torch.save(intensive_model.state_dict(), 'intensive_model_parameters.pth')

    intensive_model.load_state_dict(torch.load('intensive_model_parameters.pth'))

    # refine last few layers
    logger.info('-------------------------------------------------------------------------')
    if torch.cuda.device_count() > 1:
        optimizer = optim.Adam([{'params': [param for name, param in intensive_model.module.named_parameters() if
                                            'pre_trained_clm' not in name]}], lr=3e-4, weight_decay=0.01)
    else:
        optimizer = optim.Adam([{'params': [param for name, param in intensive_model.named_parameters() if
                                            'pre_trained_clm' not in name]}], lr=3e-4, weight_decay=0.01)

    for e in range(50):
        for i, data in enumerate(iter(dataloader_train)):
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            intensive_model.train()
            is_impossibles = utils.move_to_device(is_impossibles, device)
            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)

            # minus one, because we removed [CLS] when utils.generate_question_and_passage_hidden
            start_position = torch.where(start_position > 1, start_position - 1, start_position)
            end_position = torch.where(end_position > 1, end_position - 1, end_position)
            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            cls_output, start_logits, end_logits = intensive_model(batch_encoding['input_ids'].to(device),
                                                                   attention_mask=batch_encoding['attention_mask']
                                                                   .to(device),
                                                                   token_type_ids=batch_encoding['token_type_ids']
                                                                   .to(device),
                                                                   pad_idx=tokenizer.pad_token_id,
                                                                   max_qus_length=max_qus_len,
                                                                   max_con_length=max_con_len,
                                                                   )
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            answerable_loss = cls_loss(cls_output, is_impossibles)
            printable = (((start_loss + end_loss) / 2).item(), answerable_loss.item())
            loss = (start_loss + end_loss) / 2 + answerable_loss * multitask_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info('Epoch {}, Span Loss: {:.4f}, Ans Loss{:.4f}'
                    .format(e, printable[0], printable[1]))
        v_loss_intensive, acc, f1 = test(iter(dataloader_valid), intensive_model, device)
        logger.info('Epoch {}, Intensive valid loss {:.4f}, CLS acc{:.4f}, F1-score {:.4f}'
                    .format(e, v_loss_intensive, acc, f1))
        score = acc * f1
        if score >= best_score:  # save the best model
            best_score = score
            torch.save(intensive_model.state_dict(), 'intensive_model_parameters.pth')

    # test our model
    print('-------------------------------------------------------------------------')
    intensive_model.load_state_dict(torch.load('intensive_model_parameters.pth'))
    test_multi_task_learner(iter(dataloader_valid), intensive_model, device, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='which config')
    parser.add_argument('-d', '--dataset', type=str, help='train on which dataset')
    parser.add_argument('-w', '--multitask-weight', type=float, default=1.0, help='learn [CLS] and span jointly, given '
                                                                                  'the loss weight')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='random seed')
    args = parser.parse_args()

    config = args.config
    dataset = args.dataset
    weight = args.multitask_weight
    seed = args.seed

    CONFIG = ['cross-attention', 'match-attention']
    DATASET = ['small', 'normal']

    assert config in CONFIG, 'Given config wrong'
    assert dataset in DATASET, 'Given dataset wrong'
    assert weight > 0, 'Given weight should be larger than zero'
    main(epoch=4, which_config=config, which_dataset=dataset, multitask_weight=weight, seed=seed)
