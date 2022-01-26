import tensorflow as tf
from tensorflow.contrib import lookup

src_filename = 'eng_sentences.txt'
target_filename = 'ger_sentences.txt'
src_data = tf.data.TextLineDataset(src_filename)
target_data = tf.data.TextLineDataset(target_filename)
# split the sentences to words
src_data = src_data.map(lambda string: tf.string_split([string]).values)
target_data = target_data.map(lambda string: tf.string_split([string]).values)
# lookup tables
src_words = 'words/eng_words.txt'
target_words = 'words/ger_words.txt'
src_table = lookup.index_table_from_file(src_words, num_oov_buckets=1, name='sourcce_dict')
target_table = lookup.index_table_from_file(target_words, num_oov_buckets=1, name='target_words')
# convert words to ids
src_data = src_data.map(lambda tokens: src_table.lookup(tokens))
target_data = target_data.map(lambda tokens: target_table.lookup(tokens))
# sentences and lengths
src_data = src_data.map(lambda words: (words, tf.size(words)))
target_data = target_data.map(lambda words: (words, tf.size(words)))

dataset = tf.data.Dataset.zip((src_data, target_data))
datashape = (tf.TensorShape([None]), tf.TensorShape([])),\
            (tf.TensorShape([None]), tf.TensorShape([]))
data_value = (src_table.lookup('<GO>'), 0), (target_table.lookup('<GO>'), 0)
dataset = dataset.repeat()
dataset = dataset.padded_batch(5, datashape, data_value)
