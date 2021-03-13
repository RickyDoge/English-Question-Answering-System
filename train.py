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
from model.sketchy_reading import SketchyReadingModel
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
            cls_correct_count += torch.sum(cls_out)
            cls_total_count += cls_out.size(0)

    return loss_sum / loss_count, cls_correct_count / cls_total_count


def main(epoch=4, which_config='baseline-small', which_dataset='small', seed=2020):
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
    if which_config == 'cross-attention' or 'match-attention':
        hidden_dim = 256
        which_model = 'google/electra-small-discriminator'
    else:
        raise Exception('Input config wrong.')

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
    sketch_model = SketchyReadingModel(clm_model=which_model, hidden_dim=hidden_dim)
    if which_config == 'cross-attention':
        intensive_model = IntensiveReadingWithCrossAttention(clm_model=which_model, hidden_dim=hidden_dim)
    else:
        intensive_model = IntensiveReadingWithMatchAttention(clm_model=which_model, hidden_dim=hidden_dim)
    sketch_model.train()
    intensive_model.train()

    # GPU Config:
    if torch.cuda.device_count() > 1:
        device = torch.cuda.current_device()
        sketch_model.to(device)
        intensive_model.to(device)
        sketch_model = nn.DataParallel(module=sketch_model)
        intensive_model = nn.DataParallel(module=intensive_model)
        print('Use Multi GPUs. Number of GPUs: ', torch.cuda.device_count())
    elif torch.cuda.device_count() == 1:
        device = torch.cuda.current_device()
        sketch_model.to(device)
        intensive_model.to(device)
        print('Use 1 GPU')
    else:
        device = torch.device('cpu')  # CPU
        print("use CPU")

    if torch.cuda.device_count() > 1:
        optimizer_sketch = optim.Adam(
            [{'params': sketch_model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': sketch_model.module.cls_fc_layer.parameters(), 'lr': 5e-4,
              'weight_decay': 0.01},
             ])
        optimizer_intensive = optim.Adam(
            [{'params': intensive_model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.module.Hq_proj.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             {'params': intensive_model.module.span_detect_layer.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             ] if config == 'match-attention' else
            [{'params': intensive_model.module.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.module.attention.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             {'params': intensive_model.module.span_detect_layer.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             ]
        )
    else:
        optimizer_sketch = optim.Adam([{'params': sketch_model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
                                       {'params': sketch_model.cls_fc_layer.parameters(), 'lr': 3e-4,
                                        'weight_decay': 0.01},
                                       ])
        optimizer_intensive = optim.Adam(
            [{'params': intensive_model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.Hq_proj.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             {'params': intensive_model.span_detect_layer.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             ] if config == 'match-attention' else
            [{'params': intensive_model.pre_trained_clm.parameters(), 'lr': 3e-4, 'eps': 1e-6},
             {'params': intensive_model.attention.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             {'params': intensive_model.span_detect_layer.parameters(), 'lr': 3e-4,
              'weight_decay': 0.01},
             ]
        )

    cls_loss = nn.BCELoss()  # Binary Cross Entropy Loss
    start_end_loss = nn.CrossEntropyLoss()  # type: object

    best_f1 = 0.5
    best_acc = 0.5

    v_loss_intensive, f1 = test_intensive_reader(iter(dataloader_valid), intensive_model, device,
                                                 pad_idx=tokenizer.pad_token_id)
    logger.info('Epoch {}, Iteration {}, Intensive valid loss {:.4f}, F1-score {:.4f}'
                .format(-1, -1, v_loss_intensive, f1))

    if os.path.isfile('model_parameters.pth'):  # load previous best model
        sketch_model.load_state_dict(torch.load('sketch_model_parameters.pth'))
        intensive_model.load_state_dict(torch.load('intensive_model_parameters.pth'))

    for e in range(epoch):
        for i, data in enumerate(iter(dataloader_train)):
            sketch_model.train()
            intensive_model.train()
            batch_encoding, is_impossibles, start_position, end_position, _ = data
            is_impossibles = utils.move_to_device(is_impossibles, device)
            cls_out = sketch_model(batch_encoding['input_ids'].to(device),
                                   attention_mask=batch_encoding['attention_mask'].to(device),
                                   token_type_ids=batch_encoding['token_type_ids'].to(device),
                                   )
            impossible_loss = cls_loss(cls_out, is_impossibles)
            optimizer_sketch.zero_grad()
            impossible_loss.backward()
            optimizer_sketch.step()
            if i % 1000 == 0:
                logger.info('E_FV: Epoch {}, Iteration {}, Train Loss: {:.4f}'.format(e, i, impossible_loss.item()))
                v_loss_sketch, cls_acc = test_sketch_reader(iter(dataloader_valid), sketch_model, device)
                logger.info('Epoch {}, Iteration {}, Sketch valid loss {:.4f}, ClS Acc {:.4f}'
                            .format(e, i, v_loss_sketch, cls_acc))
                if cls_acc >= best_acc:  # save the best model
                    best_acc = cls_acc
                    torch.save(sketch_model.state_dict(), 'sketch_model_parameters.pth')

            start_position = utils.move_to_device(start_position, device)
            end_position = utils.move_to_device(end_position, device)
            start_position = torch.where(start_position > 1, start_position - 1, 0)
            end_position = torch.where(end_position > 1, end_position - 1, 0)  # minus one, because we removed [CLS]

            max_con_len, max_qus_len = utils.find_max_qus_con_length(attention_mask=batch_encoding['attention_mask'],
                                                                     token_type_ids=batch_encoding['token_type_ids'],
                                                                     max_length=batch_encoding['input_ids'].size(1),
                                                                     )
            start_logits, end_logits = intensive_model(batch_encoding['input_ids'].to(device),
                                                       attention_mask=batch_encoding['attention_mask'].to(device),
                                                       token_type_ids=batch_encoding['token_type_ids'].to(device),
                                                       pad_idx=tokenizer.pad_token_id,
                                                       max_qus_length=max_qus_len,
                                                       max_con_length=max_con_len,
                                                       )
            start_loss = start_end_loss(start_logits, start_position)
            end_loss = start_end_loss(end_logits, end_position)
            span_loss = (start_loss + end_loss) / 2
            optimizer_intensive.zero_grad()
            span_loss.backward()
            optimizer_intensive.step()
            if i % 1000 == 0:
                logger.info('I_FV: Epoch {}, Iteration {}, Train Loss: {:.4f}'.format(e, i, span_loss.item()))
                v_loss_intensive, f1 = test_intensive_reader(iter(dataloader_valid), intensive_model, device,
                                                             pad_idx=tokenizer.pad_token_id)
                logger.info('Epoch {}, Iteration {}, Intensive valid loss {:.4f}, F1-score {:.4f}'
                            .format(e, i, v_loss_intensive, f1))
                if f1 >= best_f1:  # save the best model
                    best_f1 = f1
                    torch.save(intensive_model.state_dict(), 'intensive_model_parameters.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='which config')
    parser.add_argument('-d', '--dataset', type=str, help='train on which dataset')
    parser.add_argument('-s', '--seed', type=int, default=2020, help='random seed')
    args = parser.parse_args()

    config = args.config
    dataset = args.dataset
    seed = args.seed

    CONFIG = ['cross-attention', 'match-attention']
    DATASET = ['small', 'normal']

    assert config in CONFIG, 'Given config wrong'
    assert dataset in DATASET, 'Given dataset wrong'
    main(epoch=4, which_config=config, which_dataset=dataset, seed=seed)
