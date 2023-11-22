# -*- coding: utf-8 -*-
'''
'''

import sys, os
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

import torchaudio.transforms as T
from transformers import Trainer, TrainerCallback
from src.custom_Audio_ResNet.custom_model import \
    ResNetAudio_Config, ResNetForAudioClassification

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


class MFCC_Tensor(Dataset):
    def __init__(self, df: pd.DataFrame, hparams):
        super().__init__()
        self._mfcc_transform = T.MFCC(
            sample_rate=hparams['sampling_rate'],
            n_mfcc=hparams['n_mfcc'],
            log_mels=hparams['log_mels'],
            melkwargs=hparams['melkwargs'])

        self._df = df.reset_index(drop=True)
        self._max_seq_len = int(hparams['max_duration'] * \
            int(hparams['sampling_rate']))

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, idx):
        sig, _ = torchaudio.load(self._df.loc[idx, 'wav_f_path'])
        sig = torch.mean(sig, dim=0, keepdim=True)

        # Trim
        if sig.shape[1] > self._max_seq_len:
            sig = sig[:, :self._max_seq_len]

        mfcc = self._mfcc_transform(sig) # (1, n_mfcc, num_frames)
        mfcc = mfcc.squeeze(dim=0) # (n_mfcc, num_frames)

        return {
            'wav_fpath': self._df.loc[idx, 'wav_f_path'],
            'input_values': mfcc.unsqueeze(dim=0),
                # [Cin(1), n_mfcc, num_frames]
            'labels': self._df.loc[idx, 'label'],
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
    train_dset = MFCC_Tensor(train_df, hparams=hparams)
    eval_dset = MFCC_Tensor(eval_df, hparams=hparams)

    # Load model
    config = ResNetAudio_Config(
        resnet_arch=hparams['model_config']['resnet_arch'],
        num_labels=hparams['model_config']['num_labels'])
    model = ResNetForAudioClassification(
        config=config)

    # Training arg
    training_args = hparams['training_args']

    # optimizer, scheduler, loss function
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=hparams['learning_rate'],
        weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer)

    criterion = hparams['compute_cost']
    class CustomTrainer(Trainer):
        pass
        # def compute_loss(self, model, inputs, return_outputs=False):
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)

        #     loss = criterion(outputs['logits'], labels)
        #     return (loss, outputs) if return_outputs else loss

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
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=eval_dset,
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

    # Create dset
    test_dset = MFCC_Tensor(test_df, hparams=hparams)

    # Load model
    config = ResNetAudio_Config(
        resnet_arch=hparams['model_config']['resnet_arch'],
        num_labels=hparams['model_config']['num_labels'])
    model = ResNetForAudioClassification.from_pretrained(
        hparams['model_ckpts'],
        config=config)

    # Infer
    test_trainer = Trainer(model)
    preds, _ , _ = test_trainer.predict(test_dset)

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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256"

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
