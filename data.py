"""
This file reads the data and generates new training and test examples.
"""
import tensorflow as tf

print("Reading data...")

##################################################
# READ YOUR DATA AND CREATE TRAIN AND TEST SPLIT #
##################################################

###################################################
# REPLACE WITH YOUR TRAIN AND TEST DATA GENERATOR #
###################################################
# Tip: You can also combine both generators using a parameters.
# Then, however, you have to wrap the dataset generation into a lambda.

input_shapes = (1, 1)
input_types = (tf.float32, tf.float32)
output_shapes = ()


def training_generator():
    for i in range(100):
        yield i, i+1


def test_generator():
    for i in range(100):
        yield i, i+1

