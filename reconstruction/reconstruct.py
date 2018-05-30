import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import segment from segmentation.py


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required = True)
parser.add_argument('--data_path', required = True)
FLAGS = parser.parse_args()

MODEL_PATH = FLAGS.model_path
DATA_PATH = FLAGS.data_path



def evaluate(model_path, data_path):
     with tf.Session() as sess:
        #restore metagraph
        saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'final.meta' )
        #restore vars
        saver.restore(sess, os.path.join(MODEL_PATH, 'final'))


        #TODO Load data and feed into graph
        #TODO Write output to disk




if __name__ == "__main__":
    evaluate()
