"""
Contains the model function that implements the tensorflow model.
"""
import tensorflow as tf


def model_fn(features, labels, mode, params):

    ###########################
    # REPLACE WITH YOUR MODEL #
    ###########################

    conv1 = tf.layers.conv1d(
        inputs=features["x"],
        filters=16,
        kernel_size=5,
        padding="valid",
        activation=tf.nn.elu,
        name="conv1"
    )

    pool1 = tf.layers.max_pooling1d(
        conv1,
        2,
        1,
        "valid",
        name="pool1"
    )

    flat = tf.contrib.layers.flatten(pool1)

    dense1 = tf.layers.dense(
        inputs=flat,
        units=32,
        activation=tf.nn.elu,
        name="dense1"
    )

    out = tf.layers.dense(
        inputs=dense1,
        units=1,
        activation=tf.nn.relu,
        name="out"
    )

    # PREDICTION (NO NEED TO BE CHANGED)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=out)

    #############################
    # REPLACE WITH YOUR METRICS #
    #############################
    mse = tf.reduce_mean(tf.squared_difference(labels, out))
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels, out)))

    tf.summary.scalar("rmse", rmse)

    optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])\
                        .minimize(mse, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=mse, train_op=optimizer)
