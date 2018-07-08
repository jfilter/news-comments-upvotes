import json
import pathlib
import sys
from shutil import copyfile

import keras
import numpy as np

import util
import vis
from config import *
from keras_text.data import Dataset
from keras_text.models import AlexCNN, AttentionRNN, BasicLSTM, StackedRNN, TokenModelFactory, YoonKimCNN
from keras_text.preprocessing import SimpleTokenizer

max_len = 400

epochs = 10
batch_size = 32
lr = 0.001


def train():
    optimizer = keras.optimizers.adam(lr=lr)
    word_encoder_model = YoonKimCNN()
    # word_encoder_model = AlexCNN(dropout_rate=[0, 0])
    # word_encoder_model = AttentionRNN()
    # word_encoder_model = StackedRNN()
    # word_encoder_model = BasicLSTM()

    exp_path = util.create_exp_dir()
    copyfile('./run.py', exp_path + '/run.py')

    ds_train = Dataset.load(dir_proc_data + '/train.bin')
    X_train, y_train = ds_train.X, ds_train.y

    ds_val = Dataset.load(dir_proc_data + '/val.bin')
    X_val, y_val = ds_val.X, ds_val.y

    print(ds_train.tokenizer.decode_texts(X_train[:10]))
    print(y_train[:10])

    print(ds_train.tokenizer.decode_texts(X_val[:10]))
    print(y_val[:10])

    factory = TokenModelFactory(
        2, ds_train.tokenizer.token_index, max_tokens=max_len, embedding_path=path_embedding, embedding_dims=50)

    model = factory.build_model(
        token_encoder_model=word_encoder_model, trainable_embeddings=False)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = keras.callbacks.ModelCheckpoint(
        exp_path + "/best.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = keras.callbacks.CSVLogger(
        exp_path + '/log.csv', append=True, separator=';')
    callbacks_list = [checkpoint, early_stop, csv_logger]

    model.summary()

    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks_list)

    vis.plot_history(history, exp_path)


def test_data():
    # TODO
    pass


def build_dataset():
    pathlib.Path(dir_proc_data).mkdir(parents=True)

    X_train, y_train = util.load_data(path_data + '/train.csv')

    tokenizer = SimpleTokenizer()

    # onyl build vocab on training data
    tokenizer.build_vocab(X_train)

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
    if sys.argv[1] == 'train':
        train()


if __name__ == '__main__':
    main()
