from datasets import DatasetDict
from transformers import AutoTokenizer, pipeline, Pipeline

import config
import model_utils


def train(dataset_dict: DatasetDict, model, ckpt_path) -> Pipeline:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized_ds = dataset_dict.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)
    tf_train_set, tf_validation_set = model_utils.prepare_datasets(model, tokenizer, tokenized_ds)
    callbacks = model_utils.define_callbacks(tf_validation_set)
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=config.num_epochs, callbacks=callbacks)

    model.load_weights(ckpt_path)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return pipe


if __name__ == '__main__':
    # define paths
    data_path = 'dataset'
    pipeline_path = 'pipe'
    ckpt_path = 'ckpt/best_model'

    # Load dataset and save train test split
    from dataset_utils import load_split_save

    dataset_files = [f'{data_path}/part1.tsv', f'{data_path}/part2.tsv']
    dataset_dict = load_split_save(input=dataset_files, path_to_save=data_path)

    # Create model
    id2label, label2id = model_utils.map_id_label(dataset_dict)
    len_dataset_dict_train = len(dataset_dict["train"])
    model = model_utils.create_model(id2label, label2id, len_dataset_dict_train)

    # Run training and save pipeline
    pipe = train(dataset_dict, model, ckpt_path)
    pipe.save_pretrained(pipeline_path)

