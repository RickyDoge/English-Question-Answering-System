import torch
import json
import os


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
                    answer = tokenizer.decode(batch_encoding['input_ids'][i][start: end + 1])
                    question_answer_dict[question_id[i]] = answer
                else:
                    question_answer_dict[question_id[i]] = ''
    with open(os.path.join(os.path.curdir, 'eval.json'), 'w') as file:
        json.dump(question_answer_dict, file)


def test_retro_reader_learner(valid_iterator, sketch_model, intensive_model, device, tokenizer):
    pass
