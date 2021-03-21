import torch
import torch.nn as nn
import torch.utils.data as tud
import torch.optim as optim
import logging
import argparse
from evaluate.evaluate import test_multi_task_learner_2
from transformers import ElectraTokenizerFast
from model import utils
from model.intensive_reading_ca import IntensiveReadingWithCrossAttention
from model.intensive_reading_ma import IntensiveReadingWithMatchAttention
from model.intensive_reading_cnn import IntensiveReadingWithConvolutionNet
from model.dataset import my_collate_fn, QuestionAnsweringDatasetConfiguration, QuestionAnsweringDataset
from functools import partial


def test(valid_iterator, model, device, tokenizer):
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
            # minus one, because we removed [CLS] when utils.generate_question_and_passage_hidden
            start_position = torch.where(start_position > 1, start_position - 1, start_position)
            end_position = torch.where(end_position > 1, end_position - 1, end_position)
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
            impossible_loss = cls_loss(cls_out, is_impossibles)
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            loss = (start_loss + end_loss) / 2 + impossible_loss

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

    # initialize model
    if which_config == 'cross-attention':
        retro_reader = IntensiveReadingWithCrossAttention(clm_model=which_model, hidden_dim=hidden_dim)
    elif which_config == 'match-attention':
        retro_reader = IntensiveReadingWithMatchAttention(clm_model=which_model, hidden_dim=hidden_dim)
    elif which_config == 'cnn-span':
        retro_reader = IntensiveReadingWithConvolutionNet(clm_model=which_model, hidden_dim=hidden_dim, out_channel=8,
                                                          filter_size=3)
    else:
        raise Exception('Wrong config error')
    retro_reader.train()

    # GPU Config:
    if torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        retro_reader.to(device)
        retro_reader = nn.DataParallel(module=retro_reader)
        print('Use Multi GPUs. Number of GPUs: ', torch.cuda.device_count())
    elif torch.cuda.device_count() == 1:
        device = torch.cuda.current_device()
        retro_reader.to(device)
        print('Use 1 GPU')
    else:
        device = torch.device('cpu')  # CPU
        print("use CPU")

    if torch.cuda.device_count() > 1:
        if config == 'match-attention':
            optimizer = optim.Adam(
                [{'params': retro_reader.module.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.module.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.module.Hq_proj.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.module.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        elif config == 'cross-attention':
            optimizer = optim.Adam(
                [{'params': retro_reader.module.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.module.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.module.attention.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.module.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        elif config == 'cnn-span':
            optimizer = optim.Adam(
                [{'params': retro_reader.module.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.module.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.module.conv.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        else:
            raise Exception('Wrong config error')
    else:
        if config == 'match-attention':
            optimizer = optim.Adam(
                [{'params': retro_reader.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.Hq_proj.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        elif config == 'cross-attention':
            optimizer = optim.Adam(
                [{'params': retro_reader.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.attention.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.span_detect_layer.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        elif config == 'cnn-span':
            optimizer = optim.Adam(
                [{'params': retro_reader.pre_trained_clm.parameters(), 'lr': 1e-4, 'eps': 1e-6},
                 {'params': retro_reader.cls_head.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 {'params': retro_reader.conv.parameters(), 'lr': 1e-3, 'weight_decay': 0.01},
                 ]
            )
        else:
            raise Exception('Wrong config error')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)  # learning rate decay (*0.5)

    cls_loss = nn.BCELoss()  # Binary Cross Entropy Loss
    start_end_loss = nn.CrossEntropyLoss()

    best_score = 0.25
    '''
    for e in range(epoch):
        for i, data in enumerate(iter(dataloader_train)):
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            retro_reader.train()
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
            model_output = retro_reader(batch_encoding['input_ids'].to(device),
                                           attention_mask=batch_encoding['attention_mask'].to(device),
                                           token_type_ids=batch_encoding['token_type_ids'].to(device),
                                           pad_idx=tokenizer.pad_token_id,
                                           max_qus_length=max_qus_len,
                                           max_con_length=max_con_len,
                                           )

            cls_output, start_logits, end_logits = model_output
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            answerable_loss = cls_loss(cls_output, is_impossibles)
            printable = (((start_loss + end_loss) / 2).item(), answerable_loss.item())
            loss = (start_loss + end_loss) / 2 + answerable_loss * multitask_weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                logger.info('Epoch {}, Iteration {}, Span Loss: {:.4f}, Ans Loss {:.4f}'.format(e, i, printable[0],
                                                                                               printable[1]))
                v_loss_intensive, acc, f1 = test(iter(dataloader_valid), retro_reader, device, tokenizer)
                logger.info('Epoch {}, Iteration {}, Intensive valid loss {:.4f}, CLS acc {:.4f}, F1-score {:.4f}'
                            .format(e, i, v_loss_intensive, acc, f1))
                score = acc * f1
                if score >= best_score:  # save the best model
                    best_score = score
                    torch.save(retro_reader.state_dict(), 'retro_reader.pth')
    '''
    model_dict = retro_reader.state_dict()
    previous_dict = torch.load('model_parameters.pth')
    previous_dict = {k: v for k, v in previous_dict.items() if k in model_dict}
    model_dict.update(previous_dict)
    retro_reader.load_state_dict(model_dict)

    # refine our model with cross-attention / match-attention
    logger.info('-------------------------------------------------------------------------')

    for e in range(epoch):
        for i, data in enumerate(iter(dataloader_train)):
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            retro_reader.train()
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
            cls_output, start_logits, end_logits = retro_reader(batch_encoding['input_ids'].to(device),
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

            if i % 1000 == 0:
                logger.info('Epoch {}, Span Loss: {:.4f}, Ans Loss {:.4f}'
                            .format(e, printable[0], printable[1]))
                v_loss_intensive, acc, f1 = test(iter(dataloader_valid), retro_reader, device, tokenizer)
                logger.info('Epoch {}, Intensive valid loss {:.4f}, CLS acc {:.4f}, F1-score {:.4f}'
                            .format(e, v_loss_intensive, acc, f1))
                score = acc * f1
                if score >= best_score:  # save the best model
                    best_score = score
                    torch.save(retro_reader.state_dict(), 'retro_reader.pth')
        scheduler.step()

    # test our model
    logger.info('-------------------------------------------------------------------------')
    retro_reader.load_state_dict(torch.load('retro_reader.pth'))
    test_multi_task_learner_2(iter(dataloader_valid), retro_reader, device, tokenizer)
    torch.save(retro_reader.module.state_dict(), 'single_gpu_model.pth')


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

    CONFIG = ['cross-attention', 'match-attention', 'cnn-span']
    DATASET = ['small', 'normal']

    assert config in CONFIG, 'Given config wrong'
    assert dataset in DATASET, 'Given dataset wrong'
    assert weight > 0, 'Given weight should be larger than zero'
    main(epoch=4, which_config=config, which_dataset=dataset, multitask_weight=weight, seed=seed)
