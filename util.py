import csv
import datetime
import os
import pathlib

import keras

from config import *
from keras_text.data import Dataset


# date=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
def create_exp_dir(path_data, model_str, lr, bs, prefix=None, exp_folder='experiments',):

    # asume the last name is the data name
    data_name = path_data.split('/')[-1]
    exp_dir = os.path.join(prefix, exp_folder)
    pathlib.Path(exp_dir).mkdir(parents=True, exist_ok=True)
    num_folders = len(next(os.walk(exp_dir))[1])

    # 4 digits
    epx_id_prefix = "%05d" % num_folders

    filename = epx_id_prefix + '_' + data_name + "_" + model_str + "_lr_" + \
        str(round(lr, 3)) + "_bs_" + str(round(bs, 3))

    filename = filename.replace('.', '_')

    exp_path = os.path.join(prefix, exp_folder, filename)
    pathlib.Path(exp_path).mkdir(parents=True)
    return exp_path


def load_data(path):
    X = []
    y = []

    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                new_x = row['comment_text']
                X.append(new_x)
                y.append(int(float(row['class'])))
            except Exception as e:
                print(e)

    return X, y


def build_save_data(X, y, tokenizer, path, max_len):
    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    y_cat = keras.utils.to_categorical(y, num_classes=2)

    print(y_cat[:10])

    ds = Dataset(X_padded,
                 y_cat, tokenizer=tokenizer)
    ds.save(path)


def load_train_val():
    ds_train = Dataset.load(dir_proc_data + '/train.bin')
    X_train, y_train = ds_train.X, ds_train.y

    ds_val = Dataset.load(dir_proc_data + '/val.bin')
    X_val, y_val = ds_val.X, ds_val.y
    return X_train, y_train, X_val, y_val, ds_train.tokenizer


def load_test(path_data, tokenizer):
    max_len = 400
    X, y = load_data(path_data + '/test.csv')

    X_encoded = tokenizer.encode_texts(X)
    X_padded = tokenizer.pad_sequences(
        X_encoded, fixed_token_seq_length=max_len)

    y_cat = keras.utils.to_categorical(y, num_classes=2)

    return X_padded, y_cat
