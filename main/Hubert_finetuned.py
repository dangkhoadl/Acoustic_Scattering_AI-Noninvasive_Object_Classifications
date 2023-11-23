# -*- coding: utf-8 -*-
'''
'''
import sys, os
sys.path.append(os.path.realpath('.'))

import yaml
from hyperpyyaml import load_hyperpyyaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import \
    accuracy_score, recall_score, precision_score, f1_score
import json

import torch
import torchaudio
from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2FeatureExtractor,
    PretrainedConfig,
    HubertForSequenceClassification)
from transformers import Trainer, EarlyStoppingCallback


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

class Hubert_Dset(Dataset):
    def __init__(self, df, hparams):
        super().__init__()
        self._df = df.reset_index(drop=True)
        self._feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            hparams['feat_extractor_fpath'])
        self._hparams = hparams

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        sig, sr = torchaudio.load(self._df.loc[idx, 'wav_f_path'])
        sig = torch.mean(sig, dim=0)

        # Resampling
        if sr != int(self._hparams['sampling_rate']):
            sig = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=int(self._hparams['sampling_rate']))(sig)

        inputs = self._feat_extractor(sig,
            sampling_rate=self._hparams['sampling_rate'],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=int(self._hparams['sampling_rate'] * self._hparams['max_duration']),
        )

        return {
            'wav_fpath': self._df.loc[idx, 'wav_f_path'],
            'input_values': inputs['input_values'].squeeze(dim=0),
            'attention_mask': inputs['attention_mask'],
            'labels': self._df.loc[idx, 'label']
        }


def train(hparams):
    # Read df
    train_eval_df = pd.read_csv(hparams['train_csv'])

    # Split train_df, eval_df
    train_df, eval_df = train_test_split(train_eval_df,
        test_size=0.2,
        random_state=hparams['seed'],
        stratify=train_eval_df['label'])

    # Load Datasets
    train_dset = Hubert_Dset(train_df, hparams=hparams)
    eval_dset = Hubert_Dset(eval_df, hparams=hparams)

    # Load model
    config = PretrainedConfig.from_pretrained(
        hparams['hubert_model_fpath'],
        num_labels=hparams['num_classes'])
    model = HubertForSequenceClassification.from_pretrained(
        hparams['hubert_model_fpath'],
        config=config,
        ignore_mismatched_sizes=True)

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
    class Hubert_Trainer(Trainer):
        pass
        # def compute_loss(self, model, inputs, return_outputs=False):
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)

        #     loss = criterion(outputs.logits, labels)
        #     return (loss, outputs) if return_outputs else loss

    # Finetune
    def compute_metrics(eval_pred, average='macro'):
        assert average in [None, 'micro', 'macro', 'weighted']

        predictions = np.argmax(eval_pred.predictions[0], axis=1)
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

    trainer = Hubert_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=1.0)]
    )
    trainer.train()

    # Save ckpt
    if not os.path.exists(hparams['model_ckpts']):
        os.makedirs(hparams['model_ckpts'])
    trainer.save_model(hparams['model_ckpts'])

    # Eval
    trainer.evaluate()

    # Log
    train_log, eval_log = [], []
    for e in trainer.state.log_history:
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


def test(hparams):
    test_df = pd.read_csv(hparams['test_csv'])

    # Create dset
    test_dset = Hubert_Dset(test_df, hparams=hparams)

    # Load model
    model = HubertForSequenceClassification.from_pretrained(
        hparams['model_ckpts'],
        num_labels=hparams['num_classes'])

    # Infer
    test_trainer = Trainer(model)
    preds, _ , _ = test_trainer.predict(test_dset)

    # Predict
    y_preds = np.argmax(preds[0], axis=1).astype(int)

    # scores
    scores = pd.DataFrame(
        data=preds[0],
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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:1024"

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
