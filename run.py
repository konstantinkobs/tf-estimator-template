import tensorflow as tf
from config import config
from model_fn import model_fn
from input_fn import train_input_fn, test_input_fn

params = {
    "learning_rate": config["learning_rate"]
}

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir=config["model_dir"])

print("Training model...")
model.train(input_fn=train_input_fn)

print("Testing model...")
model.evaluate(input_fn=test_input_fn)

print("Predicting...")
pred = model.predict(input_fn=test_input_fn)
