# TF Estimator template

This is a template to easily build ML systems using Tensorflow's Estimator and Dataset APIs.

## Files to be changed

To get the system up and running, you basically need to change the following files:

- `model_fn.py` contains the model definition of the network. The input is stored in the dict `features` and can be accessed as shown in the example model.
- `data.py` should store code that loads the data and defines two generator functions that deliver examples for the dataset.

The rest of the code should mainly work without changes. To start the training and evaluation, just run

```python3 run.py```

