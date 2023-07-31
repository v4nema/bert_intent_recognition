from typing import List

import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import KerasMetricCallback, TFAutoModelForSequenceClassification, create_optimizer
import tensorflow as tf


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


def create_model(id2label, label2id, len_dataset_dict_train):
    model = TFAutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id)

    batches_per_epoch = len_dataset_dict_train // config.batch_size
    total_train_steps = int(batches_per_epoch * config.num_epochs)

    optimizer, schedule = create_optimizer(init_lr=config.learning_rate,
                                           num_warmup_steps=0,
                                           num_train_steps=total_train_steps)
    model.compile(optimizer=optimizer)

    return model
