#coding: utf-8

#learn recurrent highway network proposed in https://arxiv.org/pdf/1607.03474.pdf for language model
#input: train and valid corpus (replaced to npz file by utils.py)
#output: trained language model


import sys
from six.moves import cPickle as pickle
from six.moves import range as range
import numpy as np
import numpy as xp
import time
import copy
import argparse
import chainer
import chainer.functions as F
import chainer.optimizers as O
import chainer.links as L
from chainer import cuda
import chainer.serializers as S
import chainer.initializers as I

from learnLSTMLM import SettingData
from learnLSTMLM import read_vocab
from learnLSTMLM import make_batch


class RHN:
    def __init__(self, dim, vocab, depth, layerNum=1, modelType='RHN'):
        self.dim = dim
        self.vocab = vocab
        self.depth = depth
        self.layerNum = layerNum
        self.modelType = modelType


    def make_network(self, scale, bias=0):
        dim = self.dim
        self.lmNet = chainer.Chain()
        self.lmNet.add_link('Embed', L.EmbedID(len(self.vocab), self.dim, initialW=I.Uniform(scale)))
        for i in range(self.layerNum):
            self.lmNet.add_link('x2h%s_0'%(i), L.Linear(dim, dim, initialW=I.Uniform(scale), nobias=True))
            self.lmNet.add_link('x2t%s_0'%(i), L.Linear(dim, dim, initialW=I.Uniform(scale), nobias=True))
            for j in range(self.depth):
                self.lmNet.add_link('h2h%s_%s'%(i, j), L.Linear(dim, dim, initialW=I.Uniform(scale)))
                self.lmNet.add_link('h2t%s_%s'%(i, j), L.Linear(dim, dim, initialW=I.Uniform(scale), initial_bias=bias))
        self.lmNet.add_link('Output', L.Linear(dim, len(self.vocab), initialW=I.Uniform(scale)))


    def compute_forward(self, wordIndex, prevHiddenList, embedMask=None, inputMaskList=[], prevHiddenMaskList=[], outputMask=None, dropout=0.0):
        #prevHiddenList is a list containing previous hidden states
        x = self.lmNet.Embed(wordIndex)
        if embedMask is not None:
            mask = F.broadcast_to(F.reshape(embedMask, (len(wordIndex), 1)), x.shape)
            x = x * mask
        for i in range(self.layerNum):
            x2h = self.lmNet.__getitem__('x2h%s_0'%(i))
            x2t = self.lmNet.__getitem__('x2t%s_0'%(i))
            for j in range(self.depth):
                h2h = self.lmNet.__getitem__('h2h%s_%s'%(i, j))
                h2t = self.lmNet.__getitem__('h2t%s_%s'%(i, j))
                if len(prevHiddenMaskList) != 0:
                    h = prevHiddenList[i] * prevHiddenMaskList[i]
                else:
                    h = prevHiddenList[i]
                if j == 0:
                    if len(inputMaskList) != 0:
                        x = x * inputMaskList[i]
                    h = F.tanh(x2h(x) + h2h(h))
                    t = F.sigmoid(x2t(x) + h2t(h))
                else:
                    h = F.tanh(h2h(h))
                    t = F.sigmoid(h2t(h))
                prevHiddenList[i] = (h - prevHiddenList[i]) * t + prevHiddenList[i]
            x = prevHiddenList[i]
        if outputMask is None:
            y = self.lmNet.Output(prevHiddenList[-1])
        else:
            y = self.lmNet.Output(prevHiddenList[-1] * outputMask)
        return y, prevHiddenList


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


def valid_with_batch(validData, lmWithRNN):
    batchsize = 50
    prevHiddenList = [chainer.Variable(xp.zeros((batchsize, lmWithRNN.dim)).astype(np.float32)) for _ in range(lmWithRNN.layerNum)]
    loss = 0
    totalins = len(validData) - 1
    for current_words, next_words in make_batch(validData, batchsize, 1000000):
        for index in range(current_words.shape[1]):
            wordIndex = current_words[:, index]
            y, prevHiddenList = lmWithRNN.compute_forward(wordIndex, prevHiddenList)
            loss += F.softmax_cross_entropy(y, next_words[:, index]) * batchsize
    loss = float(F.sum(loss).data) / totalins
    perp = np.exp(loss)
    return loss, perp


def train_with_batch(current_words, next_words, lmWithRNN, args, prevHiddenList):
    loss = 0
    embedMask, inputMaskList, prevHiddenMaskList, outputMask = make_dropout_mask(current_words, args.input_dropout, args.embed_dropout, args.hidden_dropout, args.output_dropout, lmWithRNN)
    for index in range(current_words.shape[1]):
        wordIndex = current_words[:, index]
        y, prevHiddenList = lmWithRNN.compute_forward(wordIndex, prevHiddenList, embedMask[:, index], inputMaskList, prevHiddenMaskList, outputMask)
        loss += F.softmax_cross_entropy(y, next_words[:, index])
    return loss, prevHiddenList


def train(lmWithRNN, args, trainData, validData):
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
    if args.WT:
        lmWithRNN.lmNet.Output.W.data = lmWithRNN.lmNet.Embed.W.data
        #assign the same id to output and embedding
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
        prevHiddenList = [chainer.Variable(xp.zeros((args.batch, args.dim)).astype(np.float32)) for _ in range(lmWithRNN.layerNum)]
        for current_words, next_words in make_batch(trainData, args.batch, args.step):
            lmWithRNN.lmNet.cleargrads()
            loss, prevHiddenList = train_with_batch(current_words, next_words, lmWithRNN, args, prevHiddenList)
            loss.backward()
            loss.unchain_backward()
            opt.update()
            totalloss += float(F.sum(loss).data) * current_words.shape[0]
            finishnum += current_words.shape[0] * current_words.shape[1]
            sys.stderr.write('\r Finished %s'%finishnum)
        sys.stderr.write('\n')
        epochEnd = time.time()
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            validloss, validperp = valid_with_batch(validData, lmWithRNN)
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
    lmWithRNN = RHN(args.dim, vocab, args.depth)
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
    parser.add_argument('--scale', dest='scale', default=0.04, type=float,
        help='scale value for initialization')
    parser.add_argument('--bias', dest='bias', default=-2.0, type=float,
        help='initial bias value')
    parser.add_argument('--lr', dest='lr', default=0.2, type=float,
        help='initial learining rate')
    parser.add_argument('--maxGrad', dest='maxGrad', default=10, type=float,
        help='max gradient norm')
    parser.add_argument('--layer', dest='layerNum', default=1, type=int,
        help='the number of RNN layers')
    parser.add_argument('--step', dest='step', default=35, type=int,
        help='the number of steps to update parameters')
    parser.add_argument('-d', '--dim', dest='dim', default=890, type=int,
        help='the number of dimensions')
    parser.add_argument('--decayEpoch', dest='decayEpoch', default=20, type=int,
        help='the epoch to keep initial learning rate')
    parser.add_argument('-e', '--epoch', dest='epoch', default=200, type=int,
        help='the number of epoch')
    parser.add_argument('--input_dropout', dest='input_dropout', default=0.75, type=float,
        help='dropout rate for rnn input')
    parser.add_argument('--hidden_dropout', dest='hidden_dropout', default=0.25, type=float,
        help='dropout rate for previous hidden state (in time step)')
    parser.add_argument('--embed_dropout', dest='embed_dropout', default=0.25, type=float,
        help='dropout rate for embedding')
    parser.add_argument('--output_dropout', dest='output_dropout', default=0.75, type=float,
        help='dropout rate for output from rnn')
    parser.add_argument('--decay', dest='decay', default=0.98039, type=float,
        help='the value for epoch decay')
    parser.add_argument('-b', '--batch', dest='batch', default=20, type=int,
        help='batch size')
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
        help='seed value')
    parser.add_argument('--weight_decay', dest='weight_decay', default=1e-7, type=float,
        help='weight decay')
    parser.add_argument('--depth', dest='depth', default=8, type=int,
        help='specify the number of depth')
    parser.add_argument('--WT', dest='WT', default=False, action='store_true',
        help='whether to share embedding matrix with output or not')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    main(args)


