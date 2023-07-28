import json
import os
import time

from datasets import Dataset
from evaluate import evaluator
from transformers import pipeline, Pipeline


def test(dataset: Dataset, pipe: Pipeline, results_path: str):
    task_evaluator = evaluator("text-classification")
    print('Computing evaluation results... It takes a lot, be patient...')
    result = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=dataset,
        label_mapping=pipe.model.config.label2id
    )
    print('Evaluation completed.')
    file = os.path.join(results_path, f'{str(int(time.time()))}_evaluation.json')
    with open(file, "w") as f:
        json.dump(result, f)
    return result


if __name__ == '__main__':
    # define paths
    data_path = 'dataset'
    pipeline_path = 'pipe'
    results_path = 'results'

    # Load dataset and pipeline
    ds_test = Dataset.from_csv(f'{data_path}/test.csv', sep='\t')
    pipe = pipeline("text-classification", model=pipeline_path)

    # Run test code
    result = test(ds_test, pipe, results_path)
    print(result)
