# coding=utf-8
import sys
import argparse
import time
import json
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path

import evaluate
import torch
from datasets import Dataset, DatasetDict, load_dataset
from hamison_datasets.UPV.load_upv import load_upv_dataset
from hamison_datasets.preprocessing import clean_text_full
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from transformers import get_linear_schedule_with_warmup, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from accelerate import Accelerator, DistributedType
from torch.distributed import barrier

from transformers import logging as transformers_logging
from datasets import logging as datasets_logging

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from make_train_test_combinations import train_test_combos, train_test_combos_all

train_test_combos = train_test_combos_all

if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable

pd.options.mode.chained_assignment = None

config = {
    'validation': False,
    "model_checkpoint": "dccuchile/bert-base-spanish-wwm-cased",
    "lr": 2e-4,
    "num_epochs": 3,
    "seed": 1234,
    'preprocessing': 'clean_text_full',
    'save_model': False
}
config['batch_size'] = 4 if "deberta" in config['model_checkpoint'] else 8
# config["lr"] = 1e-5 if "deberta" in config['model_checkpoint'] else 2e-5
# config["multilingual"] = any(
#     m in config['model_checkpoint'] for m in ["mdeberta", 'xlm'])
model_checkpoint = config['model_checkpoint']

crop_train_set_to, crop_val_set_to = 0, 0  # dev only, set 0 for full sets

RESULTS_DIR = "../data/results/UPV"
OUTPUT_DIR = "../data/models/UPV"


def get_dataloaders(data, batch_size: int, model_checkpoint: str, accelerator: Accelerator):
    """
    Creates a set of `DataLoader`s for the given dataset,

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    if "deberta" in model_checkpoint:
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_row(row):
        # returns tensors
        tokenized = tokenizer(
            row['text'],
            padding='max_length', max_length=512, truncation=True)
        # TODO should raise exception if split!='test' and 'label' not in row
        if 'label' in row:
            tokenized["labels"] = int(row['label'])

        return tokenized

    # Apply the method we just defined to all the examples in all the splits of the dataset starting with the main process first:
    with accelerator.main_process_first():
        splits = [split for split in data if split in ['train', 'val', 'test']]
        accelerator.print(f'Preprocessing and encoding...\n')
        data_encoded = {}
        for split in splits:
            # apply preprocessing
            data[split]['text'] = data[split]['text'].apply(
                clean_text_full)
            # convert to HT Dataset before encoding
            # data[split] = Dataset.from_pandas(data[split])
            # encode
            data_encoded[split] = Dataset.from_pandas(
                data[split]).map(tokenize_row)
            # data_encoded[split] = Dataset.from_pandas(
            #     data[split].copy().apply(
            #         tokenize_row, axis=1).apply(pd.Series)
            # )
            data_encoded[split].set_format(
                type="torch",
                columns=[col for col in data_encoded[split].column_names
                         if col in ['input_ids', 'token_type_ids', 'attention_mask', 'labels']])

        accelerator.print('Done.\n')

    @dataclass
    class DataCollatorForSequenceClassification:
        def __call__(self, features):
            return {
                'input_ids': torch.stack([feature['input_ids'] for feature in features]).type('torch.LongTensor'),
                'attention_mask': torch.stack([feature['attention_mask'] for feature in features]).type('torch.FloatTensor'),
                # Roberta doesn't have token_type_ids
                'token_type_ids': torch.stack([feature['token_type_ids'] for feature in features]).type('torch.LongTensor'),
                'labels': torch.Tensor([feature["labels"] for feature in features]).type('torch.LongTensor')
            }

    @dataclass
    class DataCollatorForSequenceClassificationTest:
        def __call__(self, features):
            return {
                'input_ids': torch.stack([feature['input_ids'] for feature in features]).type('torch.LongTensor'),
                'attention_mask': torch.stack([feature['attention_mask'] for feature in features]).type('torch.FloatTensor'),
                # Roberta doesn't have token_type_ids
                'token_type_ids': torch.stack([feature['token_type_ids'] for feature in features]).type('torch.LongTensor'),
            }

    # Instantiate dataloaders
    dataloaders = {
        split: DataLoader(
            data_encoded[split],
            shuffle=True,
            collate_fn=DataCollatorForSequenceClassification(),
            batch_size=batch_size,
            drop_last=True)
        for split in ['train', 'val'] if split in splits}

    if 'test' in splits:
        dataloaders['test'] = DataLoader(
            data_encoded['test'],
            shuffle=False,
            collate_fn=DataCollatorForSequenceClassificationTest(),
            batch_size=batch_size,
            drop_last=False)

    return dataloaders


def training_function(data, model_path, results_path, config, args):
    # Initialize accelerator
    accelerator = Accelerator(
        cpu=args.cpu, mixed_precision=args.mixed_precision)
    accelerator.print(config)

    with accelerator.main_process_first():
        # map labels to ids
        # current_labels = set(
        #     data['train']['label'].to_list() + data['test']['label'].to_list())
        current_labels = pd.concat([data['train'], data['test']])[
            'label'].unique()
        num_labels = len(current_labels)
        idx_to_class = {i: j for i, j in enumerate(current_labels)}
        class_to_idx = {value: key for key, value in idx_to_class.items()}
        # print(idx_to_class)
        # print(class_to_idx)
        # print(data['train']['label'])
        data['train']['label'] = data['train']['label'].map(class_to_idx)
        data['test']['label'] = data['test']['label'].map(class_to_idx)
        # print(data['train']['label'])

        # Short-hand hyper-parameters
        lr = config["lr"]
        num_epochs = int(config["num_epochs"])
        seed = int(config["seed"])
        batch_size = int(config["batch_size"])
        model_checkpoint = config['model_checkpoint']

        set_seed(seed)
        start_time = time.time()

    dataloaders = get_dataloaders(
        data, batch_size, model_checkpoint, accelerator)

    # Instantiate the model
    if "deberta" in model_checkpoint:
        model = DebertaV2ForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels)

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(dataloaders['train']) * num_epochs)
    )

    # Prepare everything: we just need to unpack the objects in the same order we gave them to the prepare method.
    # model, optimizer, dataloaders['train'], dataloaders['test'], lr_scheduler = accelerator.prepare(model, optimizer, dataloaders['train'], dataloaders['test'], lr_scheduler)
    # if 'val' in dataloaders:
    #     dataloaders['val'] = accelerator.prepare(dataloaders['val'])

    model, optimizer, dataloaders, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloaders, lr_scheduler)

    # Actual train – eval loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch in tqdm(dataloaders['train']):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation
        # model.eval()

        accelerator.print(
            f"Training time so far: {time.time() - start_time} s")

    # Predict labels of test set (only after last epoch)
    accelerator.print('Evaluating on test set')
    test_preds = []
    for batch in tqdm(dataloaders['test']):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        test_preds.extend(predictions)
    test_preds = [idx_to_class[int(p)] for p in test_preds]
    accelerator.print('preds count:\n', pd.Series(test_preds).value_counts())

    if not isinstance(data['test'], pd.DataFrame):
        data['test'] = pd.DataFrame(data['test'])
    test_indices = data['test'].index.to_list()
    index_name = data['test'].index.name

    output_preds = [{index_name: int(i), "prediction": str(p)}
                    for i, p in zip(test_indices, test_preds)]

    # Output predictions of test set
    # test_output_name = 'test_preds_trainedonval' if config['validation'] else 'preds'
    # with open(Path(results_path, f'{test_output_name}.txt'), 'w+') as f:
    #     f.write("\n".join(map(str, test_preds)))
    # with open(Path(results_path, f'{test_output_name}.json'), 'w+') as f:
    #     print('[', file=f)
    #     json_strings = ['    ' + json.dumps(
    #         {index_name: int(i), "prediction": str(p)})
    #         for i, p in zip(test_indices, test_preds)]
    #     print(*json_strings, sep=',\n', file=f)
    #     print(']', file=f)

    accelerator.print(
        f"Total training time: {time.time() - start_time} s")

    barrier()
    if config['save_model']:
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                model_path,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(unwrapped_model),
            )
            try:
                # append config parameters to config.json
                with open(f"{model_path}/config.json") as f:
                    output_config = json.load(f)
                output_config.update(config)
                with open(f"{model_path}/config.json", 'w+') as f:
                    json.dump(output_config, f, indent=2)
            except:
                accelerator.print(
                    'Could not append training parameters to config.json')

            accelerator.print(
                f"Done saving {model_checkpoint} on {model_path}.")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return output_preds


def main():
    parser = argparse.ArgumentParser(
        description="Training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true",
                        help="If passed, will train on the CPU.")
    args = parser.parse_args()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    datasets_logging.set_verbosity_error()
    datasets_logging.disable_progress_bar()

    print(f'Loading dataset.\n')
    upv = load_upv_dataset()
    upv = upv[upv['text'].notnull()].drop(
        columns=['html', 'article'])

    upv_features = [
        # '1. Espacio',
        # '2. Texto', '3. Multimedia', '4. Elemento desinformador',
        # '5.  Tipo multimedia', '6. Alteración multimedia',
        # '7. Cuenta emisora', '8. Fuente',
        # '9. Protagonista',
        # '9. Protagonista reduced',
        # '10. Atributo',
        '11. Macro tema', '12. Populismo',
        '13. Ataque', '14. Tipo de ataque',
        '14. Tipo de ataque reduced'
    ]
    output_fname = 'results_BETO.json'
    # with open(output_fname, 'w+') as f:
    #     print("[", file=f)
    results = []
    for anotation in upv_features:
        selection = upv[upv[anotation].notnull()].copy()
        selection['label'] = selection.loc[:, anotation]
        for cs_train in train_test_combos:
            data = {}
            cs_train_list = [cs_train] if isinstance(
                cs_train, str) else cs_train
            data['train'] = selection[selection['¿Qué campaña?'].isin(
                cs_train_list)]

            # Output dirs
            cs_train_list_repr = '_'.join(cs_train_list)
            model_path = f"{
                OUTPUT_DIR}/{model_checkpoint.split('/')[-1]}/{cs_train_list_repr}__{anotation}/"
            model_path += '-validation' if config['validation'] else ''

            cs_test = train_test_combos[cs_train]
            for c_test in cs_test:
                data['test'] = selection[selection['¿Qué campaña?'].isin(
                    [c_test])]

                # Output dirs
                results_path = f"{RESULTS_DIR}/{model_checkpoint.split(
                    '/')[-1]}/{cs_train_list_repr}__{c_test}__{anotation}/"

                print(
                    f'Training {model_checkpoint} on {cs_train_list}. Will test on {c_test}')

                combo_results = training_function(
                    data, model_path, results_path, config, args)

                preds = [record['prediction'] for record in combo_results]
                accuracy = np.mean(preds == data['test'][anotation])
                try:
                    f1 = f1_score(y_true=data['test']
                                  [anotation], y_pred=preds)
                except:
                    f1 = f1_score(y_true=data['test'][anotation],
                                  y_pred=preds, average='macro')

                cf = confusion_matrix(
                    y_true=data['test'][anotation], y_pred=preds)

                record = {
                    'feat': anotation,
                    'train': cs_train,
                    'test': c_test,
                    'accuracy': accuracy, 'f1': f1, 'cf': cf.tolist(),
                    'preds': combo_results}

                # with open(output_fname, 'a+') as f:
                # print(f"    {json.dumps(record)},", file=f)
                results.append(record)

    # with open(output_fname, 'a+') as f:
    #     print("]", file=f)

    with open(output_fname, 'w+') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
