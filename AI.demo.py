# -*- coding: UTF-8 -*-

import itertools
import pickle
import nltk
import sys
import sklearn
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures


def bag_of_words(words):
    return dict([(word, True) for word in words])


def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  # 把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n)  # 使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)


def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  # 所有词和（信息量大的）双词搭配一起作为特征


def create_word_scores():
    posWords = pickle.load(
        open('./pos_words.pkl', 'rb'))
    negWords = pickle.load(
        open('./neg_words.pkl', 'rb'))

    posWords = list(itertools.chain(*posWords))  # 把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords))  # 同理

    word_fd = nltk.FreqDist()  # 可统计所有词的词频
    cond_word_fd = nltk.ConditionalFreqDist()  # 可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        # help(FreqDist)
        word_fd[word] += 1  # word_fd.inc(word)
        cond_word_fd['pos'][word] += 1  # cond_word_fd['pos'].inc(word)
    for word in negWords:
        word_fd[word] += 1  # word_fd.inc(word)
        cond_word_fd['neg'][word] += 1  # cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()  # 积极词的数量
    neg_word_count = cond_word_fd['neg'].N()  # 消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count),
                                               total_word_count)  # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count),
                                               total_word_count)  # 同理
        word_scores[word] = pos_score + neg_score  # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores  # 包括了每个词和这个词的信息量


# 计算整个语料里面每个词和双词搭配的信息量
def create_word_bigram_scores():
    posdata = pickle.load(
        open('./data/pos_words.pkl', 'rb'))
    negdata = pickle.load(
        open('./data/neg_words.pkl', 'rb'))

    poswords = list(itertools.chain(*posdata))
    negwords = list(itertools.chain(*negdata))

    pos_bigram_finder = BigramCollocationFinder.from_words(poswords)
    neg_bigram_finder = BigramCollocationFinder.from_words(negwords)
    posbigrams = pos_bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negbigrams = neg_bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = poswords + posbigrams  # 词和双词搭配
    neg = negwords + negbigrams

    word_fd = nltk.FreqDist()
    cond_word_fd = nltk.ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1  # word_fd.inc(word)
        cond_word_fd['pos'][word] += 1  # cond_word_fd['pos'].inc(word)
    for word in neg:
        word_fd[word] += 1  # word_fd.inc(word)
        cond_word_fd['neg'][word] += 1  # cond_word_fd['neg'].inc(word)

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


# 根据信息量进行倒序排序，选择排名靠前的信息量的词
def find_best_words(word_scores, number):
    # 把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_vals = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    word_scores = create_word_bigram_scores()
    best_words = find_best_words(word_scores, 2000)  # 2000 is best and svc and logicstic is best 1.
    return dict([(word, True) for word in words if word in best_words])


test = sys.argv[1]


def extract_features(data):
    feat = []
    for i in data:
        feat.append(best_word_features(i))
        return feat


moto_features = best_word_features(test)  # 把文本转化为特征表示的形式

clf = pickle.load(open('./classifier/classifier_LSVC.pkl', 'rb'))
pred = clf.prob_classify_many(moto_features)

# p_file = open('./test_socre.txt', 'w')

for i in pred:
    print("【积极】" + str(i.prob('pos')) + "  【消极】" + str(i.prob('neg')))
#    p_file.write(str(i.prob('pos'))+' ' + str(i.prob('neg'))+'\n')

# p_file.close()
