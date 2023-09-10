# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'ERNIE'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 20
        self.batch_size = 16
        self.pad_size = 128
        self.learning_rate = 5e-5
        self.hidden_size = 768
        self.bert_path = './ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        print(self.tokenizer)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)  #
        self.num_filters = 256
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2
        self.hidden_size2 = 64


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.rnn_hidden * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.rnn_hidden * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        H, _ = self.lstm(encoder_out)

        M = self.tanh1(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)
        out = H * alpha

        out = torch.sum(out, 1)
        out = F.relu(out)

        out = self.fc1(out)
        out = self.fc(out)
        return out


