import nltk
import csv
import itertools
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from RNN.RNNnumpy import RNNnumpy
vocabulary_size = 8000
unkown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# # 只需要运行一次
# nltk.download("book")

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("reading csv file")
with open('data/reddit-comments-2015-08.csv', 'rt',encoding='UTF-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    next(reader)
    # 文章分成句子
    sentence = itertools.chain(*[nltk.sent_tokenize(x[0].encode('utf-8').decode('utf-8').lower()) for x in reader])
    # 句子收尾加上token
    sentence = ["%s %s %s" % (sentence_start_token,x[0],sentence_end_token) for x in sentence]
print("Spared %d sentences" % len(sentence))

# 将句子标记为单词
tokenize_sentence = [nltk.word_tokenize(x) for x in sentence]

# 计算字频
word_freq = nltk.FreqDist(itertools.chain(*tokenize_sentence))
print("Found %d unique words tokens." % len(word_freq.items()))

# 得到最常出现的一些字和build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unkown_token)
word_to_index = dict([(w,i)for i, w in enumerate(index_to_word)] )
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d  times." % (vocab[-1][0], vocab[-1][1]))

# 用unknow_token替代所有不在字典中的字
for i,sent in enumerate(tokenize_sentence):
    tokenize_sentence[i] = [w if w in word_to_index else unkown_token for w in sent]

print("\nExample sentence: '%s'" % sentence[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenize_sentence[0])
# 构造训练数据
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenize_sentence])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenize_sentence])


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases

            if (len(losses) > 1 and losses[-1][1] >losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

np.random.seed(10)
# Train on a small subset of the data to see what happens
model = RNNnumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)