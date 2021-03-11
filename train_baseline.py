import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import os
import logging
import argparse
from transformers import ElectraTokenizerFast
from model import utils
from model.baseline import BaselineModel
from model.dataset import my_collate_fn, QuestionAnsweringDatasetConfiguration, QuestionAnsweringDataset
from functools import partial


def test(valid_iterator, model, device):
    model.eval()
    cls_loss = nn.BCELoss()
    start_end_loss = nn.CrossEntropyLoss()

    loss_sum = 0  # loss
    loss_count = 0

    cls_correct_count = 0  # is impossible
    cls_total_count = 0

    f1_sum = 0  # F1-score
    f1_count = 0

    with torch.no_grad():
        for data in valid_iterator:
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            is_impossibles = utils.move_to_device(is_impossibles, device)
            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)
            cls_out, start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                                      attention_mask=batch_encoding['attention_mask'].to(device),
                                                      token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                      )
            impossible_loss = cls_loss(cls_out, is_impossibles)
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            loss = start_loss + end_loss + impossible_loss

            loss_sum += loss.item()
            loss_count += 1

            predict_start = torch.argmax(start_logits, dim=-1)
            predict_end = torch.argmax(end_logits, dim=-1)

            cls_out = torch.argmax(cls_out, dim=-1)
            cls_out = (cls_out == is_impossibles.argmax(dim=-1)).float()
            cls_correct_count += torch.sum(cls_out)
            cls_total_count += cls_out.size(0)

            predict_start = predict_start.cpu().numpy()
            predict_end = predict_end.cpu().numpy()
            start_position = start_position.cpu().numpy()
            end_position = end_position.cpu().numpy()
            for ps, pe, rs, re in zip(predict_start, predict_end, start_position, end_position):
                recall = utils.calculate_recall(ps, pe, rs, re)
                precision = utils.calculate_recall(rs, re, ps, pe)
                f1_sum += (recall + precision) / 2
                f1_count += 1

    return loss_sum / loss_count, cls_correct_count / cls_total_count, f1_sum / f1_count


def main(epoch=4, which_config='baseline-small', which_dataset='small', multitask_weight=0.5, seed=2020):
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
    if which_config == 'baseline-small':
        hidden_dim = 256
        which_model = 'google/electra-small-discriminator'
    elif which_config == 'baseline-base':
        hidden_dim = 768
        which_model = 'google/electra-base-discriminator'

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
    dataloader_train = tud.DataLoader(dataset=dataset_train, batch_size=48, shuffle=True,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))
    dataloader_valid = tud.DataLoader(dataset=dataset_valid, batch_size=48, shuffle=False,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))

    # load pre-trained model
    model = BaselineModel(clm_model=which_model, hidden_dim=hidden_dim)
    model.train()

    # GPU Config:
    if torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        model.to(device)
        model = nn.DataParallel(module=model)
        print('Use Multi GPUs. Number of GPUs: ', torch.cuda.device_count())
    elif torch.cuda.device_count() == 1:
        device = torch.cuda.current_device()
        model.to(device)
        print('Use 1 GPU')
    else:
        device = torch.device('cpu')  # CPU
        print("use CPU")

    if torch.cuda.device_count() > 1:
        optimizer = optim.Adam([{'params': model.module.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                                {'params': model.module.cls_fc_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                                {'params': model.module.span_detect_layer.parameters(), 'lr': 1e-3,
                                 'weight_decay': 0.01},
                                ])
    else:
        optimizer = optim.Adam([{'params': model.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                                {'params': model.cls_fc_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                                {'params': model.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                                ])

    cls_loss = nn.BCELoss()  # Binary Cross Entropy Loss
    start_end_loss = nn.CrossEntropyLoss()

    best_score = 0.25  # f1 * cls_acc

    if os.path.isfile('model_parameters.pth'):  # load previous best model
        model.load_state_dict(torch.load('model_parameters.pth'))

    valid_loss, cls_acc, f1 = test(iter(dataloader_valid), model, device)
    logger.info('Initial result: Valid loss {:.4f}, ClS Acc {:.4f}, F1-score {:.4f}'.format(valid_loss, cls_acc, f1))

    for e in range(epoch):
        for i, data in enumerate(iter(dataloader_train)):
            model.train()
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            is_impossibles = utils.move_to_device(is_impossibles, device)
            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)
            cls_out, start_logits, end_logits = model(batch_encoding['input_ids'].to(device),
                                                      attention_mask=batch_encoding['attention_mask'].to(device),
                                                      token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                      )
            impossible_loss = cls_loss(cls_out, is_impossibles)
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            loss = start_loss + end_loss + impossible_loss * multitask_weight
            if i % 1000 == 0:
                logger.info('Epoch {}, Iteration {}, Train Loss: {:.4f}'.format(e, i, loss.item()))

                valid_loss, cls_acc, f1 = test(iter(dataloader_valid), model, device)
                logger.info('Epoch {}, Iteration {}, Valid loss {:.4f}, ClS Acc {:.4f}, F1-score {:.4f}'
                            .format(e, i, valid_loss, cls_acc, f1))

                score = f1 * cls_acc
                if score >= best_score:  # save the best model
                    best_score = score
                    torch.save(model.state_dict(), 'model_parameters.pth')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='which config')
    parser.add_argument('-d', '--dataset', type=str, help='train on which dataset')
    parser.add_argument('-w', '--multitask-weight', type=float, default=0.5, help='learn [CLS] and span jointly, given '
                                                                                  'the loss weight')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='random seed')
    args = parser.parse_args()

    config = args.config
    dataset = args.dataset
    weight = args.multitask_weight
    seed = args.seed

    CONFIG = ['baseline-small', 'baseline-base']
    DATASET = ['small', 'normal']

    assert config in CONFIG, 'Given config wrong'
    assert dataset in DATASET, 'Given dataset wrong'
    assert weight > 0, 'Given weight should be larger than zero'
    main(epoch=4, which_config=config, which_dataset=dataset, multitask_weight=weight, seed=seed)
