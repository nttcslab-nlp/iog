#coding: utf-8

#learn variational lstm proposed in https://arxiv.org/pdf/1512.05287.pdf for language model
#input: train and valid corpus (replaced to npz file by utils.py), vocab file
#output: trained language model


import sys
import numpy as np
import numpy as xp
import time
import copy
import argparse
from six.moves import cPickle as pickle
from six.moves import range as range
import chainer
import chainer.functions as F
import chainer.optimizers as O
import chainer.links as L
from chainer import cuda
import chainer.serializers as S
import chainer.initializers as I

from learnLSTMLM import LSTMLM
from learnLSTMLM import SettingData
from learnLSTMLM import read_vocab
from learnLSTMLM import make_batch
from learnLSTMLM import valid
from learnLSTMLM import valid_with_batch


class VariationalLSTM(LSTMLM, object):
    def __init__(self, dim, vocab, layerNum, depth=1, modelType='VariationalLSTMLM'):
        super(VariationalLSTM, self).__init__(dim, vocab, layerNum, depth, modelType)


    def compute_forward4train_variationalLM(self, wordIndex, prevHidden, embedMask=None, inputMaskList=[], prevHiddenMaskList=[], outputMask=None):
        h = self.lmNet.Embed(wordIndex)
        mask = F.broadcast_to(F.reshape(embedMask, (len(wordIndex), 1)), h.shape)
        h = h * mask
        for i in range(self.layerNum):
            lstm = self.lmNet.__getitem__('LSTM%i'%i)
            if prevHidden is None:
                lstm.reset_state()
            else:
                lstm.h = lstm.h * prevHiddenMaskList[i]
            h = h * inputMaskList[i]
            lstm(h)
            h = lstm.h
        y = self.lmNet.Output(h * outputMask)
        return y, h


def make_dropout_mask(current_words, input_dropout, embed_dropout, hidden_dropout, output_dropout, lmWithRNN):
    batchsize = current_words.shape[0]
    step = current_words.shape[1]
    embedMask = np.random.binomial(1, 1 - embed_dropout, (batchsize, step)).astype(np.float32) / (1 - embed_dropout)
    current_word_list = current_words.data.tolist() #to be fast random access
    for i in range(batchsize):
        for j in range(step):
            for k in range(j + 1, step):
                #if int(current_words.data[i][k]) == int(current_words.data[i][j]):
                if current_word_list[i][k] == current_word_list[i][j]:
                    embedMask[i][k] = embedMask[i][j]
                    break
    embedMask = chainer.Variable(xp.array(embedMask))
    inputMaskList = [chainer.Variable(xp.array(np.random.binomial(1, 1 - input_dropout, (batchsize, lmWithRNN.dim)).astype(np.float32) / (1 - input_dropout))) for _ in range(lmWithRNN.layerNum)]
    prevHiddenMaskList = [chainer.Variable(xp.array(np.random.binomial(1, 1 - hidden_dropout, (batchsize, lmWithRNN.dim)).astype(np.float32) / (1 - hidden_dropout))) for _ in range(lmWithRNN.layerNum)]
    outputMask = chainer.Variable(xp.array(np.random.binomial(1, 1 - output_dropout, (batchsize, lmWithRNN.dim)).astype(np.float32) / (1 - output_dropout)))
    return embedMask, inputMaskList, prevHiddenMaskList, outputMask


def train_with_batch(current_words, next_words, lmWithRNN, args, prevHidden):
    #current words and next words are index matrix (batch * step)
    loss = 0
    embedMask, inputMaskList, prevHiddenMaskList, outputMask = make_dropout_mask(current_words, args.input_dropout, args.embed_dropout, args.hidden_dropout, args.output_dropout, lmWithRNN)
    for index in range(current_words.shape[1]):
        wordIndex = current_words[:, index]
        y, prevHidden = lmWithRNN.compute_forward4train_variationalLM(wordIndex, prevHidden, embedMask[:, index], inputMaskList, prevHiddenMaskList, outputMask)
        loss += F.softmax_cross_entropy(y, next_words[:, index])
    return loss, prevHidden


def train(lmWithRNN, args, trainData, validData):
    #trainData and validData is chainer Variable
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
    if args.WT:
        lmWithRNN.lmNet.Output.W.data = lmWithRNN.lmNet.Embed.W.data
    opt = O.SGD(args.lr)
    opt.setup(lmWithRNN.lmNet)
    opt.add_hook(chainer.optimizer.GradientClipping(args.maxGrad))
    opt.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    prevvalidperp = np.inf
    prevModel = None
    for epoch in range(args.epoch):
        epochStart = time.time()
        lr_decay = args.decay ** max(epoch + 1 - args.decayEpoch, 0.0)
        opt.lr = args.lr * lr_decay
        sys.stdout.write('Learning rate: %.6f\n'%(opt.lr))
        totalloss = 0
        finishnum = 0
        prevHidden = None
        for current_words, next_words in make_batch(trainData, args.batch, args.step):
            lmWithRNN.lmNet.cleargrads()
            loss, prevHidden = train_with_batch(current_words, next_words, lmWithRNN, args, prevHidden)
            loss.backward()
            loss.unchain_backward()
            opt.update()
            totalloss += float(F.sum(loss).data) * current_words.shape[0]
            finishnum += current_words.shape[0] * current_words.shape[1]
            sys.stderr.write('\r Finished %s'%finishnum)
        sys.stderr.write('\n')
        epochEnd = time.time()
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            if args.valid_with_batch:
                validloss, validperp = valid_with_batch(validData, lmWithRNN)
            else:
                validloss, validperp = valid(validData, lmWithRNN)
        sys.stdout.write('Train time is %s\tValid time is %s\n'%(epochEnd - epochStart, time.time() - epochEnd))
        sys.stdout.write('Epoch: %s\tTrain loss: %.6f\tValid loss: %.6f\tValid perplexity: %.6f\n'%(epoch, totalloss / finishnum, validloss, validperp))
        sys.stdout.flush()
        if prevvalidperp < validperp:
            lmOutputFile = args.output + '.epoch%s'%(epoch) + '.bin'
            S.save_npz(lmOutputFile, prevModel)
        prevModel = copy.deepcopy(lmWithRNN.lmNet).to_cpu()
        prevvalidperp = validperp
    lmOutputFile = args.output + '.epoch%s_fin'%(epoch+1) + '.bin'
    S.save_npz(lmOutputFile, prevModel)


def main(args):
    vocab = read_vocab(args.vocab)
    trainData = chainer.Variable(xp.array(np.load(args.train)['arr_0'], dtype=np.int32))
    validData = chainer.Variable(xp.array(np.load(args.valid)['arr_0'], dtype=np.int32))
    lmWithRNN = VariationalLSTM(args.dim, vocab, args.layerNum)
    lmWithRNN.make_network(args.scale)
    settingData = SettingData(lmWithRNN)
    outputFile = open(args.output + '.setting', 'w')
    pickle.dump(settingData, outputFile)
    outputFile.close()
    train(lmWithRNN, args, trainData, validData)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vocab', dest='vocab', default='',
        help='vocabulary file constructed by utils.py')
    parser.add_argument('--train', dest='train', default='',
        help='training data (.npz)')
    parser.add_argument('--valid', dest='valid', default='',
        help='validation data (.npz)')
    parser.add_argument('--output', dest='output', default='', required=True,
        help='output file name')
    parser.add_argument('--scale', dest='scale', default=0.05, type=float,
        help='scale value for initialization')
    parser.add_argument('--lr', dest='lr', default=1.0, type=float,
        help='initial learining rate')
    parser.add_argument('--maxGrad', dest='maxGrad', default=5, type=float,
        help='max gradient norm')
    parser.add_argument('--layer', dest='layerNum', default=2, type=int,
        help='the number of RNN layers')
    parser.add_argument('--step', dest='step', default=35, type=int,
        help='the number of steps to update parameters')
    parser.add_argument('-d', '--dim', dest='dim', default=650, type=int,
        help='the number of dimensions')
    parser.add_argument('--decayEpoch', dest='decayEpoch', default=6, type=int,
        help='the epoch to keep initial learning rate')
    parser.add_argument('-e', '--epoch', dest='epoch', default=39, type=int,
        help='the number of epoch')
    parser.add_argument('--input_dropout', dest='input_dropout', default=0.35, type=float,
        help='dropout rate for rnn input')
    parser.add_argument('--hidden_dropout', dest='hidden_dropout', default=0.2, type=float,
        help='dropout rate for previous hidden state (in time step)')
    parser.add_argument('--embed_dropout', dest='embed_dropout', default=0.2, type=float,
        help='dropout rate for embedding')
    parser.add_argument('--output_dropout', dest='output_dropout', default=0.35, type=float,
        help='dropout rate for output from rnn')
    parser.add_argument('--decay', dest='decay', default=0.833333, type=float,
        help='the value for epoch decay')
    parser.add_argument('-b', '--batch', dest='batch', default=20, type=int,
        help='batch size')
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
        help='seed value')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-7, type=float,
        help='weight decay')
    parser.add_argument('--WT', dest='WT', default=False, action='store_true',
        help='whether to share embedding matrix with output or not')
    parser.add_argument('--valid_with_batch', dest='valid_with_batch', default=False, action='store_true',
        help='whether valid with batch or not')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    main(args)

"""
default setting is the medium setting.

the large setting is:
--scale 0.04 \\
--maxGrad 10 \\
-d 1500 \\
--decayEpoch 14 \\
-e 55 \\
--decay 0.87 \\ 
--embed_dropout 0.3 \\
--input_dropout 0.5 \\
--hidden_dropout 0.3 \\
--output_dropout 0.5
"""
