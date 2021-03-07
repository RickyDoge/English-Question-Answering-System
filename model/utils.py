import torch


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

def calculate_recall(ps, pe, rs, re):
    if ps > re or rs > pe:
        return 0.
    case1 = abs(re - ps)
    case2 = abs(pe - rs)
    overlap = min(case1, case2)
    overall = pe - ps
    if overall == 0:
        print('warning')
        return 1.
    else:
        return overlap/overall
