# coding=utf-8
import json
from typing import List, Tuple
from bottle import route, run, request, response
from transformers import pipeline

pipe = None


def parse_element(d_elem: dict) -> Tuple:
    label = d_elem['label']
    score = d_elem['score']
    return label, score


def get_probabilities(pipe_result: List[dict]) -> dict:
    probabilities = {}
    for pr in pipe_result:
        label, score = parse_element(pr)
        probabilities[label] = score
    return probabilities


def get_winner(pipe_result: List[dict]) -> str:
    return pipe_result[0]["label"]


@route('/intent', method='POST')
def post_intent():
    try:
        data = request.json
    except:
        raise ValueError

    if data is None:
        print('data is None')
        raise ValueError

    stmt = data['stmt']
    print(stmt)
    pipe_result = pipe(stmt, top_k=101)

    out_dict = {
        'winner': get_winner(pipe_result),
        'probabilities': get_probabilities(pipe_result)
    }

    print("result: " + str(out_dict))

    response.headers['Content-Type'] = 'application/json'
    res = json.dumps(out_dict)
    return res


if __name__ == '__main__':
    pipeline_path = 'pipe'
    pipe = pipeline("text-classification", model=pipeline_path)

    # bottle.debug(True)

    run(host='0.0.0.0', port=9002)
