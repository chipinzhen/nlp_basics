import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F

# test corpus
test_sentences = ["i like dog", "i like cat", "i like animal",
                  "dog cat animal", "apple cat dog like", "cat like fish",
                  "dog like meat", "i like apple", "i hate apple",
                  "i like movie book music apple", "dog like bark", "dog friend cat"]

corpus_list = list(set(' '.join(test_sentences).split()))
corpus_dictionary = {word: i for i, word in enumerate(corpus_list)}

BATCH_SIZE = 4
VOCABULARY_SIZE = len(corpus_dictionary)
EPOCH = 100
LEARNING_RATE = 0.01

def random_batch(corpus, batchSize=1, n_skipgram=1):
    random_index = np.random.choice(range(len(corpus)), batchSize, replace=False)
    random_inputs = []
    random_labels = []
    for index in random_index:
        sentence_list = corpus[index].split()
        mid_word_index = np.random.choice(range(len(sentence_list)), 1, replace=False)[0]
        random_inputs.append(corpus_dictionary[sentence_list[mid_word_index]])
        target_index = np.random.choice(
            list(range(max(0, mid_word_index - n_skipgram), mid_word_index)) +
            list(range(mid_word_index + 1, min(mid_word_index + n_skipgram + 1, len(sentence_list)))),
            1, replace=False)[0]
        random_labels.append(corpus_dictionary[sentence_list[target_index]])

    return random_inputs, random_labels


class Word2Vec(nn.Module):
    def __init__(self, dict_size, vector_size=128):
        super().__init__()
        self.linear1 = nn.Linear(in_features=dict_size, out_features=vector_size)
        self.linear2 = nn.Linear(in_features=vector_size, out_features=dict_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def parameter_initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)


model = Word2Vec(dict_size=len(corpus_dictionary), vector_size=128)
model.parameter_initialize()
Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for epoch in range(EPOCH):
    inputs, labels = random_batch(test_sentences, batchSize=4, n_skipgram=1)
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    inputs = F.one_hot(inputs, num_classes=len(corpus_dictionary)).float()
    output = model(inputs)
    # print(output)
    # break
    loss = Loss(input=output, target=labels)
    loss.backward()
    print(loss)
    optimizer.step()
    optimizer.zero_grad()
