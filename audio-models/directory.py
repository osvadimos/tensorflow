import tensorflow as tf

filenames = tf.train.match_filenames_once('./audio_dataset/*.wav')