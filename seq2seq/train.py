import random

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import ticker

from seq2seq.models import EncoderRNN, DecoderRNN
from torch.autograd import Variable
from seq2seq import use_cuda
import seq2seq.data as data

import matplotlib.pyplot as plt
import matplotlib.animation as anim


def train(input_variable, target_variable,
          encoder, decoder,
          encoder_optim, decoder_optim,
          criterion, max_len=15):
    encoder_hidden = encoder.init_hidden()

    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    input_len = input_variable.size()[0]
    target_len = target_variable.size()[0]

    encoder_outs = Variable(torch.zeros(max_len, encoder.hidden_size))
    encoder_outs = encoder_outs.cuda() if use_cuda else encoder_outs

    for it in range(input_len):
        encoder_out, encoder_hidden = encoder(input_variable[it], encoder_hidden)
        encoder_outs[it] = encoder_out[0][0]

    decoder_in = Variable(torch.LongTensor([[0]]))
    decoder_in = decoder_in.cuda() if use_cuda else decoder_in
    decoder_hidden = encoder_hidden

    loss = 0

    if random.random() < 0.5:
        for it in range(target_len):
            decoder_out, decoder_hidden, decoder_attn = decoder(
                decoder_in, decoder_hidden, encoder_out, encoder_outs
            )
            loss += criterion(decoder_out, target_variable[it])
            decoder_in = target_variable[it]
    else:
        for it in range(target_len):
            decoder_out, decoder_hidden, decoder_attn = decoder(
                decoder_in, decoder_hidden, encoder_out, encoder_outs
            )
            topv, topi = decoder_out.data.topk(1)

            decoder_in = Variable(torch.LongTensor([[topi[0][0]]]))
            decoder_in = decoder_in.cuda() if use_cuda else decoder_in

            loss += criterion(decoder_out, target_variable[it])
            if topi[0][0] == 0:
                break

    loss.backward()

    encoder_optim.step()
    decoder_optim.step()

    return loss.data[0] / target_len


def train_iters(encoder, decoder, data_pairs):
    sum_loss = 0
    losses = []

    print('There are totally %d pairs of data.' % len(data_pairs))
    print('********* Start Training *********')

    encoder_optim = torch.optim.SGD(params=encoder.parameters(), lr=0.05)
    decoder_optim = torch.optim.SGD(params=decoder.parameters(), lr=0.05)

    criterion = torch.nn.NLLLoss()

    for i in range(len(data_pairs)):
        in_variable, out_variable = data_pairs[i][0], data_pairs[i][1]

        loss = train(
            input_variable=in_variable, target_variable=out_variable,
            encoder=encoder, decoder=decoder,
            encoder_optim=encoder_optim, decoder_optim=decoder_optim,
            criterion=criterion
        )
        sum_loss = sum_loss + loss

        if i % 100 == 0:
            print('trained: %d (%d%%) - loss: %.4f' % (i, i*100/len(data_pairs), sum_loss/20))
            losses.append(sum_loss/20)
            sum_loss = 0

        if i % 5000 == 0 and i != 0:
            torch.save(encoder, 'encoder-%d%%.pkl' % (i*100 / len(data_pairs)))
            torch.save(decoder, 'decoder-%d%%.pkl' % (i*100 / len(data_pairs)))

    plt.figure()
    fig, ax = plt.subplot()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(losses)

