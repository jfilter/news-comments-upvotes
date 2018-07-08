import csv
import datetime
import os
import pathlib

import keras

from keras_text.data import Dataset

exp_folder = 'experiments'


def create_exp_dir():
    num_folders = len(next(os.walk(exp_folder))[1])

    # 4 digits
    epx_id_prefix = "%04d_" % num_folders

    exp_path = os.path.join(
        exp_folder, epx_id_prefix + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
                y.append(int(row['class']))
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
