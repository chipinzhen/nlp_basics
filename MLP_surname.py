from torch.utils.data import Dataset, DataLoader
import random
from torch import nn
import torch
from torch import optim
from torch.nn import functional as F

surnames_list = []
country_list = []
with open('./surnames.csv', encoding='utf-8') as f:
    headline = f.readline()
    while True:
        line = f.readline().strip()
        if not line:
            break
        line_list = line.split(',')
        surnames_list.append(line_list[0])
        country_list.append(line_list[1])

label_set = list(set(country_list))
surnames_set = list(set(surnames_list))

label_dict = {label: i for i, label in enumerate(label_set)}
surnames_dict = {surname: i for i, surname in enumerate(surnames_set)}
# print(label_dict)
# print(surnames_set)

total_data = list(zip(surnames_list, country_list))
data_length = len(total_data)
random.shuffle(total_data)

# split training and test dataset.
training_data, test_data = total_data[:int(0.7 * data_length)], total_data[int(0.7 * data_length):]


class SurnameDataset(Dataset):

    def __init__(self, label_dict, surname_dict, dataList):
        self.label_dict = label_dict
        self.surname_dict = surname_dict
        self.data = self.data_convert(dataList)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)

    def data_convert(self, dataList):
        data = []
        for surname, label in dataList:
            surname_id = self.surname_dict[surname]
            label_id = self.label_dict[label]
            data.append((surname_id, label_id))
        return data


trainingDataSet = SurnameDataset(label_dict, surnames_dict, training_data)
testDataSet = SurnameDataset(label_dict, surnames_dict, test_data)

BATCHSIZE = 128
MAX_EPOCH = 100
LR = 0.1
MOMENTUM = 0.9

trainingDataLoader = DataLoader(dataset=trainingDataSet, batch_size=BATCHSIZE, shuffle=True)
testDataLoader = DataLoader(dataset=testDataSet, batch_size=128, shuffle=True)


class MLP(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(MLP, self).__init__()
        self.Linear1 = nn.Linear(inputsize, 200)
        self.Linear2 = nn.Linear(200, outputsize)
        nn.init.xavier_normal_(self.Linear1.weight)
        nn.init.xavier_normal_(self.Linear2.weight)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.Linear2(x)
        return x


mlpNet = MLP(len(surnames_dict), len(label_dict))

Loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlpNet.parameters(), lr=LR, momentum=MOMENTUM)

for epoch in range(MAX_EPOCH):
    for i, data in enumerate(trainingDataLoader):
        # data = torch.
        surnames, label = data
        surnames = F.one_hot(surnames, len(surnames_dict)).float()
        output = mlpNet(surnames)
        loss = Loss(output, label)
        # print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 5 == 0:
        num_correct = 0
        for i, data in enumerate(trainingDataLoader):
            surnames, label = data
            surnames = F.one_hot(surnames, len(surnames_dict)).float()
            output = mlpNet(surnames)
            output = F.softmax(output)
            _, predicted = torch.max(output.data, 1)
            # print(predicted)
            num_correct += torch.eq(predicted, label).sum().float().item()
        print("epoch:", epoch, "Accuracy:",num_correct/len(trainingDataLoader.dataset))