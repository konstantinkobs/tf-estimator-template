"""
Builds the datasets and creates the input_fns for training and testing.
"""
import tensorflow as tf
from config import config
from data import training_generator, test_generator, input_types, input_shapes, output_shapes

print("Building datasets...")

train_dataset = tf.data.Dataset.from_generator(
    training_generator,
    input_types,
    output_shapes=(tf.TensorShape(list(input_shapes)), tf.TensorShape(list(output_shapes))))
test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    input_types,
    output_shapes=(tf.TensorShape(list(input_shapes)), tf.TensorShape(list(output_shapes))))


def input_fn(dataset, perform_shuffle=True, repeat_count=None):
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=int(5.4 * config["batch_size"]))

    dataset = dataset.prefetch(config["max_buffer"])
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(config["batch_size"])

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return {"x": batch_features}, batch_labels


train_input_fn = lambda: input_fn(train_dataset, repeat_count=config["training_epochs"])
test_input_fn = lambda: input_fn(test_dataset, perform_shuffle=False, repeat_count=1)
