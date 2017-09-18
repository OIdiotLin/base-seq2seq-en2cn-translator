import torch
import torch.nn as nn
from torch.autograd import Variable
import seq2seq.data as data
from seq2seq.data import src_lang, tar_lang, prepare
from seq2seq.models import EncoderRNN, DecoderRNN
from seq2seq.train import train_iters

HIDDEN_SIZE = 256

if __name__ == '__main__':
    train_pairs = prepare()

    encoder = EncoderRNN(
        input_size=src_lang.n_words,
        hidden_size=HIDDEN_SIZE,
    )
    decoder = DecoderRNN(
        hidden_size=HIDDEN_SIZE,
        output_size=tar_lang.n_words,
        n_layers=1,
        dropout_p=0.1
    )

    train_iters(
        encoder=encoder,
        decoder=decoder,
        data_pairs=train_pairs
    )

