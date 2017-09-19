import jieba
import re
import torch
from torch.autograd import Variable

from seq2seq import use_cuda


class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def parse(s, lang):
        if lang == 'english' or lang == 'french':
            s = s.lower()
            s = re.sub(r"([.!?])", r" \1", s).split(' ')
        elif lang == 'chinese':
            s = jieba.lcut(s)
        return s


class Language:
    def __init__(self, name):
        self.name = name
        self.word2token = {}
        self.word2count = {}
        self.token2word = {}
        self.n_words = 0

    def add_word(self, word):
        if word not in self.word2token:
            self.word2token[word] = self.n_words
            self.token2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, s):
        for word in Tokenizer.parse(s, self.name):
            self.add_word(word)


src_lang = Language('english')
tar_lang = Language('french')

src_lang.add_word('<EOS>')
src_lang.add_word('<PAD>')
tar_lang.add_word('<EOS>')
tar_lang.add_word('<PAD>')


def get_dataset():
    """
    get data set from corpus.txt and calculate their tokens
    :return: list of tokens of source string and
             list of tokens of target string
    """

    src = []
    tar = []

    with open('%s-%s.txt' % (src_lang.name, tar_lang.name), mode='r') as f:
        lines = f.readlines()

        for line in lines:
            src_s, tar_s = line[0:-1].split('\t')
            # src.clear()
            # tar.clear()

            src_lang.add_sentence(src_s)
            tar_lang.add_sentence(tar_s)

            src_tokens = [src_lang.word2token[w] for w in Tokenizer.parse(src_s, src_lang.name)]
            tar_tokens = [tar_lang.word2token[w] for w in Tokenizer.parse(tar_s, tar_lang.name)]

            src_tokens.append(src_lang.word2token['<EOS>'])
            tar_tokens.append(tar_lang.word2token['<EOS>'])

            src_variable = Variable(torch.LongTensor(src_tokens)).view(-1, 1)
            tar_variable = Variable(torch.LongTensor(tar_tokens)).view(-1, 1)

            if use_cuda:
                src_variable = src_variable.cuda()
                tar_variable = tar_variable.cuda()

            src.append(src_variable)
            tar.append(tar_variable)

    return src, tar


def prepare():
    src, tar = get_dataset()
    return [(src[i], tar[i]) for i in range(len(src))]


