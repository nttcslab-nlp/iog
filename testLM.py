#coding: utf-8

#test trained language model
#input: test corpus (replaced to npz file by utils.py), setting file, and model file
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


def valid(validData, lmWithRNN, prev=None):
    loss = 0
    for i in xrange(len(validData) - 1):
        y, prev = lmWithRNN.compute_forward(validData[i : i+1], prev)
        loss += F.softmax_cross_entropy(y, validData[i+1 : i+2])
    loss = float(F.sum(loss).data) / (len(validData) - 1)
    perp = np.exp(loss)
    return loss, perp


def valid_with_cachemodel(validData, lmWithRNN, cachesize, prev=None):
    #computing neural cache model, with lambda 0.2, theta 0.3
    lam = 0.2
    theta = 0.3
    loss = 0
    cache = []
    max_float32_log = np.log(np.finfo(np.float32).max) - 1.0
    for i in xrange(len(validData) - 1):
        sys.stderr.write('\r%s'%i)
        if i != 0:
            z = xp.zeros((min(cachesize, i), len(lmWithRNN.vocab)), dtype=np.float32)
            z[xp.arange(min(cachesize, i)), validData.data[max(1, i - cachesize + 1) : i+1]] = 1
            one_hot = chainer.Variable(z, volatile='on')
        y, prev = lmWithRNN.compute_forward(validData[i : i+1], prev)
        if type(prev) is list:
            h = prev[-1]
        else:
            h = prev
        if i != 0:
            c = F.vstack(cache)
            cdot = theta * F.sum(c * F.broadcast_to(h, c.shape), axis=1, keepdims=True)
            cdot_lim = F.minimum(chainer.Variable(xp.full(cdot.shape, max_float32_log).astype(np.float32), volatile='on'), cdot)
            cdot_exp = F.exp(cdot_lim)
            cdotm = F.broadcast_to(cdot_exp, one_hot.shape)
            cdotsum = F.sum(cdotm * one_hot, axis=0, keepdims=True)
            cacheprob = cdotsum / F.sum(cdotsum).data
        cache.append(h)
        cache = cache[-cachesize:]
        lingprob = F.softmax(y)
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
    if 'RHN' in modelData.modelType:
        lmWithRNN = RHN(modelData.dim, modelData.vocab, modelData.depth)
    else:
        lmWithRNN = LSTMLM(modelData.dim, modelData.vocab, modelData.layerNum)
    lmWithRNN.make_network(1.0)
    S.load_npz(args.model, lmWithRNN.lmNet)
    if args.gpu >= 0:
        lmWithRNN.lmNet.to_gpu()
    testData = chainer.Variable(xp.array(np.load(args.test)['arr_0'], dtype=np.int32), volatile='on')
    if 'RHN' in modelData.modelType:
        prev = [chainer.Variable(xp.zeros((1, modelData.dim)).astype(np.float32), volatile='on') for _ in range(modelData.layerNum)]
    else:
        prev = None
    if args.cachesize > 0:
        testloss, testperp = valid_with_cachemodel(testData, lmWithRNN, args.cachesize, prev)
    else:
        testloss, testperp = valid(testData, lmWithRNN, prev)
    print 'Loss: %.6f\tPerplexity: %.6f'%(testloss, testperp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', dest='test', default='',
       help='specify the test data')
    parser.add_argument('-g', '--gpu', dest='gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--cachesize', dest='cachesize', default=0, type=int,
        help='specify the size of cache')
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


