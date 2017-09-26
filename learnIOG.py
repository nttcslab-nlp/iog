#coding: utf-8

#learn input-to-output gate to enhance the language model
#input: learned language model (this program did not train the language model), train corpus, and validation corpus
#output: trained input-to-output gate

#use chainer version 1.24.0

import sys
from six.moves import cPickle as pickle
from six.moves import range as range
import numpy as np
import numpy as xp
import argparse
import time
import copy
import chainer
from chainer import cuda
import chainer.serializers as S
import chainer.functions as F
import chainer.optimizers as O
import chainer.links as L
import chainer.initializers as I

from learnLSTMLM import read_vocab
from learnLSTMLM import make_batch
from learnLSTMLM import SettingData
from learnLSTMLM import LSTMLM
from learnRecurrentHighwayNet import RHN


class Setting4Gate:
    def __init__(self, gateModel):
        self.vocab = gateModel.vocab
        self.dim = gateModel.dim


    def set_lmmodel_file(self, settingFile, modelFile):
        self.settingFile = settingFile
        self.modelFile = modelFile


class GateModel:
    def __init__(self, vocab, dim):
        self.vocab = vocab
        self.dim = dim


    def make_network(self, scale):
        dim = self.dim
        self.gateModel = chainer.Chain(
            Embed = L.EmbedID(len(self.vocab), dim, initialW=I.Uniform(scale)),
            Weight = L.Linear(dim, len(self.vocab), initialW=I.Uniform(scale))
            )


    def compute_gate(self, input_word, dropout=0.0):
        train = dropout > 0.0
        embed = self.gateModel.Embed(input_word)
        out = self.gateModel.Weight(F.dropout(embed, ratio=dropout, train=train))
        gate = F.sigmoid(out)
        return gate


def train_with_batch(current_words, next_words, lmWithRNN, gateModel, args, prevHidden=None):
    loss = 0
    for index in range(current_words.shape[1]):
        wordIndex = current_words[:, index]
        rnn_out, prevHidden = lmWithRNN.compute_forward(wordIndex, prevHidden, dropout=0.0)
        gate = gateModel.compute_gate(wordIndex, args.dropout)
        y = gate * rnn_out
        loss += F.softmax_cross_entropy(y, next_words[:, index])
    return loss, prevHidden


def valid_with_batch(validData, lmWithRNN, gateModel):
    batchsize = 64
    totalins = len(validData) - 1
    loss = 0
    if lmWithRNN.modelType == 'RHN':
        prevHidden = [chainer.Variable(xp.zeros((batchsize, lmWithRNN.dim)).astype(np.float32), volatile='on') for _ in range(lmWithRNN.layerNum)]
    else:
        prevHidden = None
    for current_words, next_words in make_batch(validData, batchsize, 100000):
        for index in range(current_words.shape[1]):
            wordIndex = current_words[:, index]
            rnn_out, prevHidden = lmWithRNN.compute_forward(wordIndex, prevHidden)
            gate = gateModel.compute_gate(wordIndex)
            y = gate * rnn_out
            loss += F.softmax_cross_entropy(y, next_words[:, index]) * batchsize
    loss = float(F.sum(loss).data) / totalins
    perp = np.exp(loss)
    return loss, perp


def train(lmWithRNN, gateModel, args, trainData, validData):
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
        gateModel.gateModel.to_gpu()
    opt = O.Adam(alpha=0.001)
    opt.setup(gateModel.gateModel)
    opt.add_hook(chainer.optimizer.GradientClipping(args.maxGrad))
    bestperp = np.inf
    for epoch in range(args.epoch):
        epochStart = time.time()
        totalloss = 0
        finishnum = 0
        lr_decay = np.sqrt(epoch + 1)
        opt.alpha = 0.001 / lr_decay
        print 'Learning rate: %.6f'%(opt.alpha)
        if lmWithRNN.modelType == 'RHN':
            prevHidden = [chainer.Variable(xp.zeros((args.batch, lmWithRNN.dim)).astype(np.float32)) for _ in range(lmWithRNN.layerNum)]
        else:
            prevHidden = None
        for current_words, next_words in make_batch(trainData, args.batch, args.step):
            lmWithRNN.lmNet.cleargrads()
            gateModel.gateModel.cleargrads()
            loss, prevHidden = train_with_batch(current_words, next_words, lmWithRNN, gateModel, args, prevHidden)
            loss.backward()
            loss.unchain_backward()
            opt.update()
            totalloss += float(F.sum(loss).data) * current_words.shape[0]
            finishnum += current_words.shape[0] * current_words.shape[1]
            sys.stderr.write('\r Finished %s'%finishnum)
        sys.stderr.write('\n')
        epochEnd = time.time()
        validloss, validperp = valid_with_batch(validData, lmWithRNN, gateModel)
        sys.stderr.write('Train time is %s\tValid time is %s\n'%(epochEnd - epochStart, time.time() - epochEnd))
        sys.stdout.write('Epoch: %s\tTrain loss: %.6f\tValid loss: %.6f\tValid perplexity: %.6f\n'%(epoch, totalloss / finishnum, validloss, validperp))
        sys.stdout.flush()
        if validperp < bestperp:
            gateOutputFile = args.output + '.bin'
            S.save_npz(gateOutputFile, copy.deepcopy(gateModel.gateModel).to_cpu())
            bestperp = validperp


def main(args):
    trainData = chainer.Variable(xp.array(np.load(args.train)['arr_0'], dtype=np.int32))
    validData = chainer.Variable(xp.array(np.load(args.valid)['arr_0'], dtype=np.int32), volatile='on')
    lmModelData = pickle.load(open(args.setting))
    if 'RHN' in lmModelData.modelType:
        lmWithRNN = RHN(lmModelData.dim, lmModelData.vocab, lmModelData.depth)
    else:
        lmWithRNN = LSTMLM(lmModelData.dim, lmModelData.vocab, lmModelData.layerNum)
    lmWithRNN.make_network(1.0)
    S.load_npz(args.model, lmWithRNN.lmNet)
    gateModel = GateModel(lmWithRNN.vocab, args.dim)
    gateModel.make_network(args.scale)
    settingData = Setting4Gate(gateModel)
    settingData.set_lmmodel_file(args.setting, args.model)
    outputFile = open(args.output + '.setting', 'w')
    pickle.dump(settingData, outputFile)
    outputFile.close()
    train(lmWithRNN, gateModel, args, trainData, validData)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--train', dest='train', default='', required=True,
        help='training data (.npz)')
    parser.add_argument('--valid', dest='valid', default='', required=True,
        help='validation data (.npz)')
    parser.add_argument('--output', dest='output', default='', required=True,
        help='output file name')
    parser.add_argument('--maxGrad', dest='maxGrad', default=0.1, type=float,
        help='max gradient norm')
    parser.add_argument('--step', dest='step', default=35, type=int,
        help='the number of steps to update parameters')
    parser.add_argument('-e', '--epoch', dest='epoch', default=5, type=int,
        help='the number of epoch')
    parser.add_argument('--dropout', dest='dropout', default=0.5, type=float,
        help='dropout rate')
    parser.add_argument('-d', '--dim', dest='dim', default=300, type=int,
        help='the number of dimensions')
    parser.add_argument('-b', '--batch', dest='batch', default=20, type=int,
        help='batch size')
    parser.add_argument('--scale', dest='scale', default=0.01, type=float,
        help='scale value for initialization')
    parser.add_argument('-s', '--setting', dest='setting', default='', required=True,
        help='specify the setting file of trained language model')
    parser.add_argument('-m', '--model', dest='model', default='', required=True,
        help='specify the name of trained language model file')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np
    main(args)


