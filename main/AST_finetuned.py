# -*- coding: utf-8 -*-
'''
Finetune AST AudioSet
'''

import sys, os
import yaml
from hyperpyyaml import load_hyperpyyaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import \
    accuracy_score, recall_score, precision_score, f1_score

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from transformers import ASTFeatureExtractor, ASTForAudioClassification
from transformers import Trainer, TrainerCallback
from datasets import Dataset, DatasetDict


def parse_arguments(args):
    yaml_fpath, *override_args = args

    # Parse overrides
    assert len(override_args) % 2 == 0
    overrides = dict()
    for i in np.arange(0, len(override_args), 2):
        param, val = override_args[i].lstrip('--'), override_args[i+1]
        overrides[param] = val

    # Parse config.yaml
    with open(yaml_fpath, 'r+') as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    return hparams


def extract_feats(examples, hparams):
    max_seq_len = int(hparams['max_duration'] * \
        int(hparams['sampling_rate']))

    # X
    sig_s = []
    for fpath in examples['wav_f_path']:
        sig, sr = torchaudio.load(fpath)
        sig = torch.mean(sig, dim=0)

        # Resampling
        if sr != int(hparams['sampling_rate']):
            sig = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=int(hparams['sampling_rate']))(sig)

        # Trim
        sig = sig.squeeze(dim=0).numpy()
        if sig.shape[0] > max_seq_len:
            sig = sig[:max_seq_len]

        # Append
        sig_s.append(sig)

    inputs = hparams['feat_extractor'](sig_s,
            sampling_rate=hparams['sampling_rate'],
            padding="max_length",
            return_tensors="pt")

    examples['input_values'] = inputs['input_values']

    # y
    examples['labels'] = examples["label"]

    return examples

def train(hparams):
    # Read df
    train_eval_df = pd.read_csv(hparams['train_csv'])

    # Split train_df, eval_df
    train_df, eval_df = train_test_split(train_eval_df,
        test_size=0.3,
        random_state=hparams['seed'],
        stratify=train_eval_df['label'])

    # Load Datasets
    dsets = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(eval_df)
    })
    dsets = dsets.remove_columns(
        ['__index_level_0__'])

    # Dset preprocessing
    dsets = dsets \
        .map(lambda x: extract_feats(x, hparams), batched=True)

    # Load pretrained model
    model = ASTForAudioClassification.from_pretrained(
        hparams['ast_model_fpath'],
        num_labels=hparams['num_classes'])

    # Freeze and unfreeze layers
    if hparams['finetuned_layers'] is not None:
        # Freeze and unfreeze layers
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            for finetuned_layer in list(hparams['finetuned_layers']):
                if finetuned_layer in name:
                    param.requires_grad = True
    ## Recheck
    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print(name)

    # Training arg
    training_args = hparams['training_args']

    # optimizer, scheduler, loss function
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=hparams['learning_rate'],
        weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer)

    criterion = hparams['compute_cost']
    class ASTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)

            loss = criterion(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Eval metrics
    def compute_metrics(eval_pred, average='macro'):
        assert average in [None, 'micro', 'macro', 'weighted']

        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids

        # accuracy, precision, recall, f1-score
        acc = accuracy_score(
            y_true=labels, y_pred=predictions,
            normalize=True)
        r = recall_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)
        p = precision_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)
        f1 = f1_score(
            y_true=labels, y_pred=predictions,
            average=average, zero_division=0)

        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1 }

    class PrinterCallback(TrainerCallback):
        def __write_log(self, state):
            train_log, eval_log = [], []

            for e in state.log_history:
                e_keys = set(e)
                if "loss" in e_keys: train_log.append(e)
                elif "eval_loss" in e_keys: eval_log.append(e)
                elif "train_runtime" in e_keys:
                    with open(f"{hparams['exp_dir']}/trainer_info.json", 'w+', encoding='utf-8') as fin:
                        json.dump(e, fin, ensure_ascii=False, indent=4)

            if train_log != []:
                train_log_df = pd.DataFrame.from_dict(train_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                train_log_df.to_csv(f"{hparams['exp_dir']}/log_trainset.csv", index=False)

            if eval_log != []:
                eval_log_df = pd.DataFrame.from_dict(eval_log) \
                    .sort_values("step", ascending=True) \
                    .reset_index(drop=True)
                eval_log_df.to_csv(f"{hparams['exp_dir']}/log_evalset.csv", index=False)

        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            '''Write log after every eval round'''
            self.__write_log(state)
        def on_train_end(self, args, state, control, logs=None, **kwargs):
            '''Write log after training'''
            self.__write_log(state)

    # Finetune
    trainer = ASTTrainer(
        model=model,
        args=training_args,
        train_dataset=dsets['train'],
        eval_dataset=dsets['validation'],
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[
            hparams['early_stopping'],
            PrinterCallback()]
    )
    trainer.train()

    # Save ckpt
    if not os.path.exists(hparams['model_ckpts']):
        os.makedirs(hparams['model_ckpts'])
    trainer.save_model(hparams['model_ckpts'])

    # Eval
    trainer.evaluate()


def test(hparams):
    test_df = pd.read_csv(hparams['test_csv'])

    # Load Datasets
    dsets = DatasetDict({
        'test': Dataset.from_pandas(test_df),
    })

    # Dset preprocessing
    dsets = dsets \
        .map(lambda x: extract_feats(x, hparams), batched=True)

    # Load model
    model = ASTForAudioClassification.from_pretrained(
        hparams['model_ckpts'],
        num_labels=hparams['num_classes'])

    # Infer
    test_trainer = Trainer(model)
    preds, _ , _ = test_trainer.predict(dsets['test'])

    # Predict
    y_preds = np.argmax(preds, axis=1).astype(int)

    # scores
    scores = pd.DataFrame(
        data=preds,
        columns=[ f"Class_{c}_score" for c in (np.arange(hparams['num_classes'])) ])

    # Prediction
    y_preds = pd.DataFrame(
        data=y_preds,
        columns=[ 'Prediction' ])

    # out
    test_df = pd.concat(
        [test_df.reset_index(drop=True), scores, y_preds], axis=1)
    test_df.to_csv(f"{hparams['exp_dir']}/test_scores.csv", index=False)


if __name__ == "__main__":
    torch.cuda.empty_cache()

    # Parse hparams
    hparams = parse_arguments(sys.argv[1:])

    # Set seed
    np.random.seed(hparams['seed'])
    torch.manual_seed(hparams['seed'])
    torch.cuda.manual_seed_all(hparams['seed'])

    # Set Env
    os.environ["CUDA_VISIBLE_DEVICES"]=hparams['GPUs']
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"

    # Create exp dir
    if not os.path.exists(hparams['exp_dir']):
        os.makedirs(hparams['exp_dir'])

    # Save yaml config for reference
    with open(f"{hparams['exp_dir']}/log_exp-conf.yaml", 'w+') as fout:
        yaml.dump(hparams, fout,
            default_flow_style=False)

    # Run exp
    train(hparams)
    test(hparams)
