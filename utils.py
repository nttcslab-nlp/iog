#coding: utf-8

#input: the directory containing training data, valid data, test data
#output: vocabulary, each data replaced to .npz file


import sys
import numpy as np
import argparse


def make_vocab(trainFile):
    vocablist = []
    vocab = {}
    for line in open(trainFile):
        line = line.strip() + ' <eos>'
        for word in line.split():
            if not word in vocab:
                vocablist.append(word)
                vocab[word] = len(vocab)
    return vocablist, vocab


def replace_word2index(vocab, inputFile, outputFile):
    #replace word to index and output the numpy file
    indexList = [vocab[word] for line in open(inputFile) for word in (line.strip() + ' <eos>').split()]
    indexList = np.array(indexList, dtype=np.int32)
    np.savez_compressed(outputFile, indexList)


def main(args):
    vocablist, vocab = make_vocab(args.inputDirectory + '/%s.train.txt'%(args.prefix))
    print '\n'.join(vocablist)
    replace_word2index(vocab, args.inputDirectory + '/%s.train.txt'%(args.prefix), args.outputDirectory + '/%s.train.npz'%(args.prefix))
    replace_word2index(vocab, args.inputDirectory + '/%s.test.txt'%(args.prefix), args.outputDirectory + '/%s.test.npz'%(args.prefix))
    replace_word2index(vocab, args.inputDirectory + '/%s.valid.txt'%(args.prefix), args.outputDirectory + '/%s.valid.npz'%(args.prefix))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDirectory', dest='inputDirectory', default='', required=True,
        help='specify the directory including training, valid, test dataset')
    parser.add_argument('-o', '--outputDirectory', dest='outputDirectory', default='', required=True,
        help='psecify the directory for output files')
    parser.add_argument('-p', '--prefix', dest='prefix', default='ptb',
        help='specify the prefix of input files')
    args = parser.parse_args()
    if not args.prefix in ['ptb', 'wiki']:
        sys.stderr.write('Please specify correct prefix of input files\n')
        sys.exit(1)
    main(args)


