import torch
import torch.nn.utils.rnn as rnn


def move_to_device(data_structure, device):
    if torch.is_tensor(data_structure):
        return data_structure.to(device)
    elif isinstance(data_structure, list):
        return [move_to_device(x, device) for x in data_structure]
    elif isinstance(data_structure, dict):
        return {key: move_to_device(value, device) for key, value in data_structure.items()}
    else:
        return data_structure


def find_idx_of_token(batch_encoding, token_id):
    sep_idx = []
    for sample in batch_encoding['input_ids']:
        for i in range(sample.size()[0]):
            if sample[i] == token_id:
                sep_idx.append(i)
                break
    return torch.tensor(token_id).long()


def generate_question_and_passage_hidden(last_hidden, attention_mask, token_type_ids, pad_idx):
    max_length = last_hidden.size(1)

    sep_token_position = []
    for batch_token_type_ids in token_type_ids:  # a batch of token_type_ids
        batch_sep_token_position = 0
        for i, token_type in enumerate(batch_token_type_ids):  # each token in a batch (sentence)
            if token_type.item() == 1:  # find the index of the first [SEP] token
                batch_sep_token_position = i - 1
                break
        sep_token_position.append(batch_sep_token_position)

    pad_token_position = []
    for batch_token_type_ids in attention_mask:  # a batch of attention mask
        batch_pad_token_position = 0
        for i, token_type in enumerate(batch_token_type_ids):  # each token in a batch (sentence)
            if token_type.item() == 0:  # find the index of the first [PAD] token
                batch_pad_token_position = i - 1
                break
        if batch_pad_token_position == 0:
            batch_pad_token_position = max_length
        pad_token_position.append(batch_pad_token_position)

    question_hidden = []
    passage_hidden = []
    question_length = []
    passage_length = []

    for i, sentence in enumerate(last_hidden):
        passage_hidden.append(sentence[1:sep_token_position[i], :])
        passage_length.append(sep_token_position[i] - 1)
        question_hidden.append(sentence[sep_token_position[i] + 1:pad_token_position[i], :])
        question_length.append(pad_token_position[i] - sep_token_position[i] - 1)

    question_hidden = rnn.pad_sequence(question_hidden, batch_first=True, padding_value=pad_idx)
    passage_hidden = rnn.pad_sequence(passage_hidden, batch_first=True, padding_value=pad_idx)

    passage_pad_mask = length_to_mask(passage_length)
    question_pad_mask = length_to_mask(question_length)

    return question_hidden, passage_hidden, question_pad_mask, passage_pad_mask


def length_to_mask(length):  # length should be one dimensional
    length = torch.tensor(length)
    max_len = length.max().item()
    mask = torch.arange(max_len).expand(len(length), max_len) > length.unsqueeze(1)
    return mask


def calculate_recall(ps, pe, rs, re):
    if ps > re or rs > pe:
        return 0.
    case1 = abs(re - ps)
    case2 = abs(pe - rs)
    case3 = abs(pe - ps)
    case4 = abs(re - rs)
    overlap = min(case1, case2, case3, case4)
    overall = pe - ps
    if overall == 0:
        return 1.
    else:
        return overlap / overall
