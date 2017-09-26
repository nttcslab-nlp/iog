#coding: utf-8

#learn LSTM language model introduced by Zaremba et al., https://arxiv.org/abs/1409.2329
#input: train and valid corpus (replaced to npz file by utils.py), vocab file
#output: trained language model

#use chainer version 1.24.0
#This program cannot run on chainer v2


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


class SettingData:
    def __init__(self, learnedModel):
        self.vocab = learnedModel.vocab
        self.modelType = learnedModel.modelType
        self.dim = learnedModel.dim
        self.layerNum = learnedModel.layerNum
        self.depth = learnedModel.depth


class LSTMLM:
    def __init__(self, dim, vocab, layerNum, depth=1, modelType='LSTMLM'):
        self.dim = dim
        self.vocab = vocab
        self.layerNum = layerNum
        self.modelType = modelType
        self.depth = depth


    def make_network(self, scale):
        dim = self.dim
        self.lmNet = chainer.Chain()
        self.lmNet.add_link('Embed', L.EmbedID(len(self.vocab), dim, initialW=I.Uniform(scale)))
        for j in range(self.layerNum):
            self.lmNet.add_link('LSTM%s'%(j), L.LSTM(dim, dim, lateral_init=I.Uniform(scale), upward_init=I.Uniform(scale)))
        self.lmNet.add_link('Output', L.Linear(dim, len(self.vocab), initialW=I.Uniform(scale)))


    def compute_forward(self, wordIndex, prevHidden, dropout=0.0):
        #input: input word indices (chainer Variable)
        #prev hidden: previous hidden state
        train = dropout > 0.0
        embed = self.lmNet.Embed(wordIndex)
        h = F.dropout(embed, ratio=dropout, train=train)
        for j in range(self.layerNum):
            lstm = self.lmNet.__getitem__('LSTM%s'%(j))
            if prevHidden is None:
                lstm.reset_state()
            lstm(h)
            h = F.dropout(lstm.h, ratio=dropout, train=train)
        y = self.lmNet.Output(h)
        return y, h


def read_vocab(vocabFile):
    d = {}
    for line in open(vocabFile):
        line = line.strip()
        d[line] = len(d)
    return d


def make_batch(data, batchsize, step):
    #data is chainer Variable representing word indices
    dataLength = len(data)
    batchLength = dataLength // batchsize
    data4batch = F.reshape(data[0 : batchsize * batchLength], (batchsize, batchLength))
    for i in range(0, batchLength - 1, step):
        current_words = data4batch[:, i : min(i+step, batchLength - 1)]
        next_words = data4batch[:, i+1 : min(i+1+step, batchLength)]
        yield current_words, next_words


def train_with_batch(current_words, next_words, lmWithRNN, args, prevHidden=None):
    #current words and next words are index matrix (batch * step)
    loss = 0
    for index in range(current_words.shape[1]):
        wordIndex = current_words[:, index]
        y, prevHidden = lmWithRNN.compute_forward(wordIndex, prevHidden, args.dropout)
        loss += F.softmax_cross_entropy(y, next_words[:, index])
    return loss, prevHidden


def valid(validData, lmWithRNN):
    #valid data is chainer variable
    prevHidden = None
    loss = 0
    for i in range(len(validData) - 1):
        y, prevHidden = lmWithRNN.compute_forward(validData[i : i+1], prevHidden)
        loss += F.softmax_cross_entropy(y, validData[i+1 : i+2])
    loss = float(F.sum(loss).data) / (len(validData) - 1)
    perp = np.exp(loss)
    return loss, perp


def valid_with_batch(validData, lmWithRNN):
    batchsize = 50
    prevHidden = None
    totalins = len(validData) - 1
    loss = 0
    for current_words, next_words in make_batch(validData, batchsize, 100000):
        for index in range(current_words.shape[1]):
            wordIndex = current_words[:, index]
            y, prevHidden = lmWithRNN.compute_forward(wordIndex, prevHidden)
            loss += F.softmax_cross_entropy(y, next_words[:, index]) * batchsize
    loss = float(F.sum(loss).data) / totalins
    perp = np.exp(loss)
    return loss, perp


def train(lmWithRNN, args, trainData, validData):
    #trainData and validData is chainer Variable
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
    if args.WT:
        lmWithRNN.lmNet.Output.W.data = lmWithRNN.lmNet.Embed.W.data
    opt = O.SGD(args.lr)
    opt.setup(lmWithRNN.lmNet)
    opt.add_hook(chainer.optimizer.GradientClipping(args.maxGrad))
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
    validData = chainer.Variable(xp.array(np.load(args.valid)['arr_0'], dtype=np.int32), volatile='on')
    lmWithRNN = LSTMLM(args.dim, vocab, args.layerNum)
    lmWithRNN.make_network(args.scale)
    settingData = SettingData(lmWithRNN)
    outputFile = open(args.output + '.setting', 'w')
    pickle.dump(settingData, outputFile)
    outputFile.close()
    train(lmWithRNN, args, trainData, validData)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #default hyper parameter is the same as medium model implemented by tensorflow (https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)
    parser.add_argument('-v', '--vocab', dest='vocab', default='',
        help='vocabulary file constructed by utils.py')
    parser.add_argument('--train', dest='train', default='', required=True,
        help='training data (.npz)')
    parser.add_argument('--valid', dest='valid', default='', required=True,
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
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float,
        help='dropout rate')
    parser.add_argument('--decay', dest='decay', default=0.833333, type=float,
        help='the value for epoch decay')
    parser.add_argument('-b', '--batch', dest='batch', default=20, type=int,
        help='batch size')
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
        help='seed value')
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
default parameter is the medium setting.

the large setting is:
--maxGrad 10 \\
--scale 0.04 \\
-d 1500 \\
--decayEpoch 14 \\
-e 55 \\
--dropout 0.65 \\
--decay 0.87
"""

