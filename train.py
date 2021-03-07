import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import torch.optim as optim
import logging
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

    for data in valid_iterator:
        batch_encoding, is_impossibles, start_position, end_position, id = data
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

        for ps, pe, rs, re in zip(predict_start, predict_end, start_position, end_position):
            recall = utils.calculate_recall(ps, pe, rs, re)
            precision = utils.calculate_recall(rs, re, ps, pe)
            f1_sum += (recall + precision) / 2
            f1_count += 1

    return loss_sum / loss_count, cls_correct_count / cls_total_count, f1_sum / f1_count


def main(epoch=10):
    torch.random.manual_seed(2020)
    torch.manual_seed(2020)

    # log
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log.log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # load dataset
    tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')

    config_train = QuestionAnsweringDatasetConfiguration(squad_train=True)
    config_valid = QuestionAnsweringDatasetConfiguration(squad_dev=True)
    dataset_train = QuestionAnsweringDataset(config_train, tokenizer=tokenizer)
    dataset_valid = QuestionAnsweringDataset(config_valid, tokenizer=tokenizer)
    dataloader_train = tud.DataLoader(dataset=dataset_train, batch_size=64, shuffle=True,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))
    dataloader_valid = tud.DataLoader(dataset=dataset_valid, batch_size=64, shuffle=False,
                                      collate_fn=partial(my_collate_fn, tokenizer=tokenizer))
    train_iterator = iter(dataloader_train)
    valid_iterator = iter(dataloader_valid)

    # load pre-trained model
    model = BaselineModel()
    model.train()

    # GPU Config:
    if torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        model.to(device)
        model = nn.DataParallel(module=model)
        print('Use Multi GPUs: ', device, '.Number of GPUs: ', torch.cuda.device_count())
    elif torch.cuda.device_count() == 1:
        device = torch.cuda.current_device()
        model.to(device)
        print('Use one GPU: ', device)
    else:
        device = torch.device('cpu')  # CPU
        print("use CPU: ", device)

    if torch.cuda.device_count() > 1:
        optimizer = optim.Adam([{'params': model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
                                {'params': model.module.cls_fc_layer.parameters(), 'lr': 1e-3},
                                {'params': model.module.span_detect_layer.parameters(), 'lr': 1e-3},
                                ])
    else:
        optimizer = optim.Adam([{'params': model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
                                {'params': model.cls_fc_layer.parameters(), 'lr': 1e-3},
                                {'params': model.span_detect_layer.parameters(), 'lr': 1e-3},
                                ])

    cls_loss = nn.BCELoss()  # Binary Cross Entropy Loss
    start_end_loss = nn.CrossEntropyLoss()

    for e in range(epoch):
        valid_loss, cls_acc, f1 = test(valid_iterator, model, device)
        logger.info('Epoch {}, Valid loss {}, Classification Acc {}, F1-score {}'.format(e, valid_loss, cls_acc, f1))
        for i, data in enumerate(train_iterator):
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
            loss = start_loss + end_loss + impossible_loss
            if i % 1000 == 0:
                logger.info('Epoch {}, Iteration {}, Train Loss: {}'.format(e, i, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main(epoch=10)
