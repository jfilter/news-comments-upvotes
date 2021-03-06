import json
import os
import pathlib
import random
import sys
from shutil import copyfile, move

import keras
import numpy as np

import util
from config import *
from keras_text.data import Dataset
from keras_text.models import AlexCNN, AttentionRNN, BasicLSTM, StackedRNN, TokenModelFactory, YoonKimCNN
from keras_text.preprocessing import SimpleTokenizer

# init random
random.seed()

max_len = 400
embedding_dims = 300

epochs = 400
patience = 10

batch_size = 10000


def train_stacked(lr=0.001, batch_size=batch_size, dropout_rate=0.5, hidden_dims=[10, 5], rnn_class=keras.layers.GRU, e_dir=None):
    word_encoder_model = StackedRNN(
        hidden_dims=hidden_dims, dropout_rate=dropout_rate, rnn_class=rnn_class)
    train(word_encoder_model, lr, batch_size, e_dir=e_dir)


def train_cnn(lr=0.001, batch_size=batch_size, dropout_rate=0.5, filter_sizes=[3, 4, 5], num_filters=20, e_dir=None):
    word_encoder_model = YoonKimCNN(
        filter_sizes=filter_sizes, num_filters=num_filters, dropout_rate=dropout_rate)
    train(word_encoder_model, lr, batch_size, e_dir=e_dir)


def train_attention(lr=0.001, batch_size=50, dropout_rate=0.5, encoder_dims=50, rnn_class=keras.layers.GRU, e_dir=None):
    word_encoder_model = AttentionRNN(
        encoder_dims=encoder_dims, dropout_rate=dropout_rate, rnn_class=rnn_class)
    train(word_encoder_model, lr, batch_size, e_dir=e_dir)


def train(word_encoder_model, lr, batch_size, e_dir=None):
    optimizer = keras.optimizers.adam(lr=lr)

    X_train, y_train, X_val, y_val, tokenizer = util.load_train_val()

    factory = TokenModelFactory(
        2, tokenizer.token_index, max_tokens=max_len, embedding_path=path_embedding, embedding_dims=embedding_dims)

    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    exp_path = util.create_exp_dir(path_data, str(
        word_encoder_model), lr=lr, bs=batch_size, prefix=e_dir)
    copyfile('./run.py', exp_path + '/run.py')

    print(exp_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        exp_path + "/best.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience)
    csv_logger = keras.callbacks.CSVLogger(
        exp_path + '/log.csv', append=True, separator=';')
    callbacks_list = [checkpoint, early_stop, csv_logger]

    model.summary()

    with open(exp_path + '/config.txt', 'a') as the_file:
        the_file.write(
            '\n'.join([str(x) for x in [lr, word_encoder_model.dropout_rate, batch_size]]))

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)

    best_acc = str(max(history.history['val_acc']))[:6]

    # append best acc
    move(exp_path, exp_path + '_' + best_acc)


def test_data():
    path = 'my_model.h5'
    model = keras.models.load_model(path)

    _, _, _, _, tokenizer = util.load_train_val()

    X, y = util.load_test(path_data, tokenizer)

    res = model.eval(x=X, y=y, batch_size=200)
    print(res)


def search_hyper_cnn():
    e_dir = os.path.join(base_experiment_folder,
                         'cnn_' + path_data.split('/')[-1])
    while(True):
        filter_sizes = None
        while(True):
            num_filter_sizes = random.randint(2, 4)
            filter_sizes = random.sample(range(1, 10), num_filter_sizes)
            if 2 in filter_sizes or 3 in filter_sizes:
                break
        num_filters = random.randint(10, 30)
        lr = random.uniform(0.0001, 0.01)
        do = random.uniform(0.3, 0.8)
        batch_size = random.randint(500, 6000)
        train_cnn(filter_sizes=filter_sizes, num_filters=num_filters,
                  lr=lr, dropout_rate=do, batch_size=batch_size, e_dir=e_dir)


def search_hyper_stacked():
    e_dir = os.path.join(base_experiment_folder,
                         'stacked_' + path_data.split('/')[-1])
    while(True):
        h1 = random.randint(1, 10)
        h2 = random.randint(1, 10)
        lr = random.uniform(0.0001, 0.01)
        do = random.uniform(0.3, 0.6)
        batch_size = random.randint(500, 2000)
        # batch_size = 2 ** random.randint(9, 12)
        # batch_size = 64
        train_stacked(hidden_dims=[h1, h2], lr=lr,
                      dropout_rate=do, batch_size=batch_size, e_dir=e_dir)


def search_hyper_simple():
    e_dir = os.path.join(base_experiment_folder,
                         'simple_lstm' + path_data.split('/')[-1])
    while(True):
        h1 = random.randint(1, 10)
        h2 = random.randint(1, 10)
        lr = random.uniform(0.0001, 0.01)
        do = random.uniform(0.3, 0.6)
        batch_size = random.randint(500, 2000)
        # batch_size = 2 ** random.randint(9, 12)
        # batch_size = 64
        train_stacked(hidden_dims=[h1, h2], lr=lr,
                      dropout_rate=do, batch_size=batch_size, e_dir=e_dir)


def build_dataset():
    pathlib.Path(dir_proc_data).mkdir(parents=True)

    X_train, y_train = util.load_data(path_data + '/train.csv')

    tokenizer = SimpleTokenizer()

    # onyl build vocab on training data
    tokenizer.build_vocab(X_train)
#    tokenizer.apply_encoding_options(limit_top_tokens=20000)
    util.build_save_data(X_train, y_train, tokenizer,
                         dir_proc_data + '/train.bin', max_len)

    X_val, y_val = util.load_data(path_data + '/val.csv')
    util.build_save_data(X_val, y_val, tokenizer,
                         dir_proc_data + '/val.bin', max_len)

    X_test, y_test = util.load_data(path_data + '/test.csv')
    util.build_save_data(X_test, y_test, tokenizer,
                         dir_proc_data + '/test.bin', max_len)


def main():
    if len(sys.argv) != 2:
        raise ValueError('You have to specify a positional command!')
    if sys.argv[1] == 'build':
        build_dataset()
    if sys.argv[1] == 'traincnn':
        train_cnn()
    if sys.argv[1] == 'trainstacked':
        train_stacked()
    if sys.argv[1] == 'trainatt':
        train_attention()
    if sys.argv[1] == 'searchcnn':
        search_hyper_cnn()
    if sys.argv[1] == 'searchstacked':
        search_hyper_stacked()
    if sys.argv[1] == 'test':
        test_data()


if __name__ == '__main__':
    main()
