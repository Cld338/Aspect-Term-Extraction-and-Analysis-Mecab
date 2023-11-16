from transformers import BertModel
from torch.nn import BatchNorm1d
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau


class bert_ATE(torch.nn.Module):
    def __init__(self, pretrain_model):
        super(bert_ATE, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model, return_dict=False)
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, 3)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    def forward(self, ids_tensors, tags_tensors, masks_tensors):
        bert_outputs, _ = self.bert(input_ids=ids_tensors, attention_mask=masks_tensors, return_dict=False)

        linear_outputs = self.linear(bert_outputs)
        # print(linear_outputs.size())

        if tags_tensors is not None:
            tags_tensors = tags_tensors.view(-1)
            linear_outputs = linear_outputs.view(-1, 3)
            # print(linear_outputs.size())
            # print(tags_tensors.size())
            loss = self.loss_fn(linear_outputs, tags_tensors)

            return loss
        else:
            return linear_outputs

    def train_model(self, loader, epochs):
        all_data = len(loader)
        for epoch in range(epochs):
            finish_data = 0
            losses = []
            current_times = []
            correct_predictions = 0

            for data in loader:
                t0 = time.time()
                ids_tensors, tags_tensors, _, masks_tensors = data
                # ids_tensors = torch.LongTensor([[math.ceil(j) for j in i] for i in ids_tensors])
                # print(ids_tensors)

                ids_tensors = ids_tensors.to(DEVICE)
                tags_tensors = tags_tensors.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                loss = self(ids_tensors=ids_tensors, tags_tensors=tags_tensors, masks_tensors=masks_tensors)
                losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                finish_data += 1
                current_times.append(round(time.time() - t0, 3))
                current = np.mean(current_times)
                hr, min, sec = evl_time(current * (all_data - finish_data) + current * all_data * (epochs - epoch - 1))
                print('epoch:', epoch, " batch:", finish_data, "/", all_data, " loss:", np.mean(losses), " hr:", hr,
                      " min:", min, " sec:", sec)

            # Learning rate scheduler step
            self.scheduler.step(np.mean(losses))

    def test_model(self, loader):
        pred = []
        trueth = []
        with torch.no_grad():
            for data in loader:
                ids_tensors, tags_tensors, _, masks_tensors = data

                ids_tensors = ids_tensors.to(DEVICE)
                tags_tensors = tags_tensors.to(DEVICE)
                masks_tensors = masks_tensors.to(DEVICE)

                outputs = self(ids_tensors=ids_tensors, tags_tensors=None, masks_tensors=masks_tensors)

                _, predictions = torch.max(outputs, dim=2)

                pred += list([int(j) for i in predictions for j in i])
                trueth += list([int(j) for i in tags_tensors for j in i])

        return trueth, pred