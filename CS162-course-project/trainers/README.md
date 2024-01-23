# Training Pipeline

## Milestone 1: Evaluation Setup

Please finish the `TODO`s in the two functions `evaluate_standard` and `pairwise_accuracy` in `train_utils.py` file.

You can execute the following commands to test your implementations:
```bash
python3 train_utils.py

# Or, at the root directory.
python3 -m trainers.train_utils
```

If it prints out:

```bash
Your `evaluate_standard` function is correct!
Your `pairwise_accuracy` function is correct!
```
Then you should be all set!  
Otherwise, please check your implementation again as there is something wrong something wrong.

## Milestone 3: Model Training and Selection

For model training, you are expected to complete two TODO blocks named `Training Loop` and `Evaluation Loop` in `train.py` file.
For model selection, you need to complete the TODO block name `Model Selection` in `train.py` file.
Please make sure to read through most of the codes in `train.py` so as to get familiar with the major training and evaluation paradigm of this standard NLP training.

In order to test your implementation, simply run:
```bash
bash scripts/train_dummy.sh
```

If your implementations are correct, then the training and evaluation loops should run fine. You should be able to obtain 100% accuracy on the dev and test sets.