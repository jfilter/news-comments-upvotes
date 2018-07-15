import os
import json

# https://www.kaggle.com/danbrice/keras-plot-history-full-report-and-grid-search
import matplotlib

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def _plot_history(history, path):
    loss_list = [s for s in history.keys(
    ) if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.keys()
                     if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.keys(
    ) if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys()
                    if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    # As loss always exists
    epochs = range(1, len(history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    l = loss_list[-1]
    plt.plot(epochs, history[l], 'b', label='Training loss (' + str(
            str(format(history[l][-1], '.5f'))+')'))
    l = val_loss_list[-1]
    plt.plot(epochs, history[l], 'g', label='Validation loss (' + str(
            str(format(history[l][-1], '.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(path + '/loss.png')

    # Accuracy
    plt.figure(2)
    l = acc_list[-1]
    plt.plot(epochs, history[l], 'b', label='Training accuracy (' + str(
            format(history[l][-1], '.5f'))+')')
    l = val_acc_list[-1]
    plt.plot(epochs, history[l], 'g', label='Validation accuracy (' + str(
            format(history[l][-1], '.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path + '/accuracy.png')


def plot_history(history_dict, path):
    _plot_history(history_dict.history, path)


def plot_history_json_file(json_path, path):
    with open(json_path) as handle:
        dictdump = json.loads(handle.read())
        _plot_history(dictdump, path)
