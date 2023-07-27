from typing import List, Any

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset, DatasetDict, Dataset
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
import evaluate
import numpy as np

import config

accuracy = evaluate.load(config.eval_metric)
model_name = config.model_name


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def map_id_label(dd: DatasetDict):
    id2label = {i: v for i, v in enumerate(dd['train'].info.features['label'].names)}
    label2id = {v: i for i, v in id2label.items()}
    # num_labels = len(id2label)
    return id2label, label2id


def load_split_dataset(files: List[str]) -> DatasetDict:
    # ds = load_dataset('csv', delimiter="\t", data_files=['dataset/part1.tsv', 'dataset/part2.tsv'])
    ds = load_dataset('csv', delimiter="\t", data_files=files)
    ds_train = ds['train'].rename_column('intent', 'label').class_encode_column("label")

    dd = ds_train.train_test_split(seed=42,
                                   test_size=0.2,
                                   stratify_by_column='label')
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


def train(dataset_dict: DatasetDict):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_ds = dataset_dict.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)

    id2label, label2id = map_id_label(dataset_dict)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    batches_per_epoch = len(dataset_dict["train"]) // config.batch_size
    total_train_steps = int(batches_per_epoch * config.num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    tf_train_set, tf_validation_set = prepare_datasets(model, tokenizer, tokenized_ds)

    model.compile(optimizer=optimizer)

    callbacks = define_callbacks(tf_validation_set)

    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=config.num_epochs, callbacks=callbacks)


if __name__ == '__main__':
    dataset_dict = load_split_dataset(config.dataset_files)
    train(dataset_dict)
