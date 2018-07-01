import datetime
import os
import sys
import csv
import pickle

from shutil import copyfile


import keras
import numpy as np

from keras_text.corpus import imdb
from keras_text.data import Dataset
from keras_text.models import AlexCNN, AttentionRNN, StackedRNN, TokenModelFactory, YoonKimCNN, BasicLSTM
from keras_text.preprocessing import SimpleTokenizer

import pathlib


import vis

from config import *

max_len = 50


def build_dataset():
    X = []
    y = []

    with open(path_data) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                new_x = row['comment_text']
                X.append(new_x)
                y.append(int(row['bin']))
            except Exception as e:
                print(e)

    tokenizer = SimpleTokenizer()

    tokenizer.build_vocab(X)

    # tokenizer.apply_encoding_options(limit_top_tokens=5000)

    print(list(tokenizer.token_index)[:50])
    user_token = tokenizer.token_index['<user>']
    print(user_token)

    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    y_cat = keras.utils.to_categorical(y, num_classes=2)

    print(y_cat[:10])

    ds = Dataset(X_padded, y_cat, tokenizer=tokenizer)
    ds.update_test_indices(test_size=0.1)
    ds.save(path_for_proc_data)


def train():
    exp_path = os.path.join(
        'experiments', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    pathlib.Path(exp_path).mkdir(parents=True)

    copyfile('./run.py', exp_path + '/run.py')

    ds = Dataset.load(path_for_proc_data)
    X_train, _, y_train, _ = ds.train_val_split()

    print(ds.tokenizer.decode_texts(X_train[:10]))

    print(y_train[:10])

    # RNN models can use `max_tokens=None` to indicate variable length words per mini-batch.
    factory = TokenModelFactory(
        2, ds.tokenizer.token_index, max_tokens=max_len, embedding_path=path_embedding, embedding_dims=50)
    # 2, ds.tokenizer.token_index, max_tokens=max_len, embedding_type='fasttext.simple')

    # word_encoder_model = YoonKimCNN()
    # word_encoder_model = AlexCNN(dropout_rate=[0, 0])
    # word_encoder_model = AttentionRNN()
    # word_encoder_model = StackedRNN()
    word_encoder_model = BasicLSTM()
    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=2,
                        batch_size=32, validation_split=0.1)
    print(history.history)
    with open(exp_path + '/training_history.bin', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    vis.plot_history(history, exp_path)


def main():
    if len(sys.argv) != 2:
        raise 'error'
    if sys.argv[1] == 'build':
        build_dataset()
    if sys.argv[1] == 'train':
        train()


if __name__ == '__main__':
    main()
