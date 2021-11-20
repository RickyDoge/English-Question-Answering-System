# English Question Answering System

Span detection style Open-domain QA (Pretrained Model)。机器问答

#### Require:

[PyTorch](https://pytorch.org/get-started/locally/)

```
pip install transformers
conda install -c anaconda importlib-metadata
```

#### Download pre-trained model before running:

```
from transformers import ElectraModel, ElectraTokenizerFast
tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
model = ElectraModel.from_pretrained('google/electra-small-discriminator')
```

#### Data preprocessing:

Create a folder "dataset". Move all files into it. (official SQuAD 2.0 / Quoref / NewsQA / Drop / Medhop / Wikihop) Then run scripts in preprocessing/

#### How to run our baseline:

Run with ELECTRA_small, with only SQuAD 2.0 training set.

```
python train_baseline.py -c baseline-small -d small
```

Run with ELECTRA_base, with all datasets except SQuAD 2.0 dev set.

```
python train_baseline.py -c baseline-base -d normal
```

#### How to run advanced model:

Make sure you run baseline first, so you can see a "model_parameters.pth" file

Run with cross-attention decoder with ELECTRA_small, with only SQuAD 2.0 training set, with random seed 114514:

```
python train.py -c cross-attention -d small -s 114514
```

Run with match-attention decoder  with ELECTRA_small, with all datasets except SQuAD 2.0 dev set, with using regression loss:

```
python train.py -c match-attention -d normal -rl
```

Run with CNN decoder with ELECTRA_base, with only SQuAD 2.0 training set, with using dynamic weight averaging:

```
python train.py -c cnn-span-large -d small -dw
```

#### How to evaluate:

Move unprocessed dev-squad2.0.json into directory: **/evaluate** and **/evaluate/processed_dataset**

Run evaluate.py, and run SQuAD official evaluate script.
