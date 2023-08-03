Intent Recognition with BERT italian uncased from HuggingFace 


https://huggingface.co/osiria/bert-base-italian-uncased

## Train
Training data are in tsv format, with columns being [text, intent].

When dataset is loaded and prepared for training, new files are created with format csv, but still with tab separator, and with columns [text, intent, labels].

To start the training, review the config file to set the learning parameters, and run the script train_intent_recognition.py.
At the end of the process, the script will save the pipeline with the trained model, so specify the path in which the pipeline will be saved.

## Test
Run the test_intent_recognition.py script. This script can evaluate the model on the whole test set or only on specific statements.
It needs the pipeline path.

## Intent Recognition Service
Run the script run_intent_recognition.py in order to use intent recognition as a REST API.

Here is request example:

    headers = {'Content-Type': 'application/json'}
    { 
        "stmt": "this is a test statement" 
    }

Here is a response example:

    {
        "winner": "INTENT_42",
        "probabilities": {
            "INTENT_42": 0.7136337161064148,
            "INTENT_56": 0.02917424589395523,
            "INTENT_8": 0.013974654488265514,
            ...
        }
    }