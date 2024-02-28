# -*- coding: utf-8 -*-
import math
import torch
import pickle
from lm.model import RNNModel
import numpy as np
from lm.data import Dictionary


class LMProb():
    def __init__(self, model_path, dict_path):
        with open(dict_path, 'rb') as f:
            self.dictionary = pickle.load(f)
        with open(model_path, 'rb') as f:
            self.model = RNNModel(len(self.dictionary), 300, 300, 3)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.model = self.model.cuda()
        print(len(self.dictionary))

    def get_prob(self, words, verbose=False):
        pad_words = ['<sos>'] + words + ['<eos>']
        indxs = [self.dictionary.getid(w) for w in pad_words]
        # print (indxs, self.dictionary.getid('_UNK'))
        input = torch.LongTensor([int(indxs[0])]).unsqueeze(0).cuda()
        input.requires_grad = False

        if verbose:
            print('words =', pad_words)
            print('indxs =', indxs)

        with torch.no_grad():
            hidden = self.model.init_hidden(1)
            log_probs = []
            for i in range(1, len(pad_words)):
                output, hidden = self.model(input, hidden)
                # print (output.data.max(), output.data.exp())
                word_weights = output.squeeze().double().exp()
                # print (i, pad_words[i])
                prob = word_weights[indxs[i]] / word_weights.sum()
                log_probs.append(math.log(prob))
                # print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(pad_words[i], indxs[i], log_probs[-1]))
                input.fill_(int(indxs[i]))

        if verbose:
            for i in range(len(log_probs)):
                print('  {} => {:d},\tlogP(w|s)={:.4f}'.format(
                    pad_words[i + 1], indxs[i + 1], log_probs[i]))
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        # return sum(log_probs) / math.sqrt(len(log_probs))
        return sum(log_probs) / len(log_probs)
