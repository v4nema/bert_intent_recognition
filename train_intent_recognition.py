import json
import os
import time
from typing import List

from transformers import AutoTokenizer, pipeline, Pipeline
from datasets import load_dataset, DatasetDict, Dataset
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
import evaluate

import numpy as np

import config


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load(config.eval_metric)
    return accuracy.compute(predictions=predictions, references=labels)


def map_id_label(dd: DatasetDict):
    id2label = {i: v for i, v in enumerate(dd['train'].info.features['label'].names)}
    label2id = {v: i for i, v in id2label.items()}
    # num_labels = len(id2label)
    return id2label, label2id


def load_split_dataset(files: List[str], path_to_save: str = None) -> DatasetDict:
    # ds = load_dataset('csv', delimiter="\t", data_files=['dataset/part1.tsv', 'dataset/part2.tsv'])
    ds = load_dataset('csv', delimiter="\t", data_files=files)
    ds_train = ds['train'].add_column(name='label', column=ds['train']['intent'])
    ds_train = ds_train.class_encode_column("label")

    dd = ds_train.train_test_split(seed=42,
                                   test_size=0.2,
                                   stratify_by_column='label')
    if path_to_save:
        dd['train'].to_csv(f'{path_to_save}/train.csv', sep='\t')
        dd['test'].to_csv(f'{path_to_save}/test.csv', sep='\t')

    return dd


def define_callbacks(tf_validation_set) -> List[tf.keras.callbacks.Callback]:
    callbacks = []

    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    callbacks.append(metric_callback)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='ckpt/best_model',
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True)
    callbacks.append(checkpoint_callback)

    return callbacks


def prepare_datasets(model, tokenizer, tokenized_ds):
    tf_train_set = model.prepare_tf_dataset(
        tokenized_ds["train"],
        shuffle=True,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_ds["test"],
        shuffle=False,
        batch_size=config.batch_size,
        tokenizer=tokenizer,
    )

    return tf_train_set, tf_validation_set


def create_model(id2label, label2id):
    model = TFAutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id)

    batches_per_epoch = len(dataset_dict["train"]) // config.batch_size
    total_train_steps = int(batches_per_epoch * config.num_epochs)

    optimizer, schedule = create_optimizer(init_lr=config.learning_rate,
                                           num_warmup_steps=0,
                                           num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer)

    return model


def train(dataset_dict: DatasetDict, model) -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized_ds = dataset_dict.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)
    tf_train_set, tf_validation_set = prepare_datasets(model, tokenizer, tokenized_ds)
    callbacks = define_callbacks(tf_validation_set)
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=config.num_epochs, callbacks=callbacks)

    model.load_weights('ckpt/best_model')
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe


def test(dataset: Dataset, label2id, pipe: Pipeline):
    task_evaluator = evaluate.evaluator("text-classification")
    print('Computing evaluation results... It takes a lot, be patient...')
    result = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        label_mapping=label2id
    )
    print('Evaluation completed.')
    file = os.path.join(config.results_path, f'{str(int(time.time()))}_evaluation.json')
    with open(file, "w") as f:
        json.dump(result, f)
    return result

# def save_last_model(model):
#     import os
#     cmd = 'zip -r intent_bert.zip intent_bert'
#     model.save_pretrained('intent_bert', from_pt=True)
#     os.system(cmd)
#     last_model = TFAutoModelForSequenceClassification.from_pretrained('intent_bert')


if __name__ == '__main__':
    dataset_dict = load_split_dataset(config.dataset_files, config.train_test_split_path)
    id2label, label2id = map_id_label(dataset_dict)
    model = create_model(id2label, label2id)
    pipe = train(dataset_dict, model)

    ds_test = Dataset.from_csv(f'{config.train_test_split_path}/test.csv', sep='\t')
    result = test(ds_test, label2id, pipe)
    print(result)
    # save_last_model(model)
