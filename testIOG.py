#coding: utf-8

#test trained input-to-output gate model
#input: test corpus (replaced to npz file by utils.py), IOG setting file, and IOG model file
#output: loss value and perplexity

#use chainer version 1.24.0

import sys
from six.moves import cPickle as pickle
from six.moves import range as range
import numpy as np
import numpy as xp
import argparse
import chainer
from chainer import cuda
import chainer.serializers as S
import chainer.functions as F

from learnLSTMLM import SettingData
from learnLSTMLM import LSTMLM
from learnRecurrentHighwayNet import RHN
from learnIOG import Setting4Gate
from learnIOG import GateModel


def valid(validData, lmWithRNN, gateModel, prev=None):
    loss = 0
    for i in xrange(len(validData) - 1):
        y, prev = lmWithRNN.compute_forward(validData[i : i+1], prev)
        gate = gateModel.compute_gate(validData[i : i+1])
        loss += F.softmax_cross_entropy(y * gate, validData[i+1 : i+2])
    loss = float(F.sum(loss).data) / (len(validData) - 1)
    perp = np.exp(loss)
    return loss, perp


def valid_with_cachemodel(validData, lmWithRNN, gateModel, cachesize, prev):
    #computing neural cache model, with lambda 0.2, theta 0.3
    lam = 0.2
    theta = 0.3
    loss = 0
    cache = []
    for i in xrange(len(validData) - 1):
        sys.stderr.write('\r%s'%i)
        if i != 0:
            z = xp.zeros((min(cachesize, i), len(lmWithRNN.vocab)), dtype=np.float32)
            z[xp.arange(min(cachesize, i)), validData.data[max(1, i - cachesize + 1) : i+1]] = 1
            one_hot = chainer.Variable(z)
        y, prev = lmWithRNN.compute_forward(validData[i : i+1], prev)
        if type(prev) is list:
            h = prev[-1]
        else:
            h = prev
        if i != 0:
            c = F.vstack(cache)
            cdot = F.exp(theta * F.sum(c * F.broadcast_to(h, c.shape), axis=1, keepdims=True))
            cdotm = F.broadcast_to(cdot, one_hot.shape)
            cdotsum = F.sum(cdotm * one_hot, axis=0, keepdims=True)
            cacheprob = cdotsum / F.sum(cdotsum).data
        cache.append(h)
        cache = cache[-cachesize:]
        gate = gateModel.compute_gate(validData[i : i+1])
        lingprob = F.softmax(y * gate)
        if i == 0:
            p = lingprob
        else:
            p = lam * F.reshape(cacheprob, (1, len(lmWithRNN.vocab))) + (1 - lam) * lingprob
        logp = -F.log(p)
        loss += logp[0, validData[i+1 : i+2].data[0]]
    loss = float(F.sum(loss).data) / (len(validData) - 1)
    perp = np.exp(loss)
    sys.stderr.write('/n')
    return loss, perp


def main(args):
    modelData = pickle.load(open(args.setting))
    gateModel = GateModel(modelData.vocab, modelData.dim)
    gateModel.make_network(1.0)
    S.load_npz(args.model, gateModel.gateModel)
    lmModelData = pickle.load(open(modelData.settingFile))
    if 'RHN' in lmModelData.modelType:
        lmWithRNN = RHN(lmModelData.dim, lmModelData.vocab, lmModelData.depth)
    else:
        lmWithRNN = LSTMLM(lmModelData.dim, lmModelData.vocab, lmModelData.layerNum)
    lmWithRNN.make_network(1.0)
    S.load_npz(modelData.modelFile, lmWithRNN.lmNet)
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
        gateModel.gateModel.to_gpu()
    testData = chainer.Variable(xp.array(np.load(args.test)['arr_0'], dtype=np.int32))
    if 'RHN' in lmModelData.modelType:
        prev = [chainer.Variable(xp.zeros((1, lmModelData.dim)).astype(np.float32)) for _ in range(lmModelData.layerNum)]
    else:
        prev = None
    if args.cachesize > 0:
        testloss, testperp = valid_with_cachemodel(testData, lmWithRNN, gateModel, args.cachesize, prev)
    else:
        testloss, testperp = valid(testData, lmWithRNN, gateModel, prev)
    print 'Loss: %.6f\tPerplexity: %.6f'%(testloss, testperp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', dest='test', default='',
       help='test data (.npz)')
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cachesize', dest='cachesize', default=0, type=int,
        help='cache size')
    parser.add_argument('-s', '--setting', dest='setting', default='', required=True,
        help='specify the setting file of trained IOG (including the path of the used language model')
    parser.add_argument('-m', '--model', dest='model', default='', required=True,
        help='specify the name of trained IOG')
    args = parser.parse_args()
    if args.gpu >= 0:
        cuda.check_cuda_available()
        cuda.get_device(args.gpu).use()
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        xp = cuda.cupy if args.gpu >= 0 else np

    main(args)


