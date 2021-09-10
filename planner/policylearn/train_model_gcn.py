#!/usr/bin/env python3
import argparse
import os
import pickle
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tools import ProgressBar
from torch.nn import Linear, MSELoss
from torch.special import expit
from torch_geometric.data import DataLoader
from torch_geometric.nn import (ChebConv, GCNConv, global_add_pool,
                                global_max_pool, global_mean_pool)

LOSSFN = MSELoss()


# src: https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
class GCN(torch.nn.Module):
    def __init__(self, conv_channels, class_channels, num_node_features):
        super(GCN, self).__init__()
        torch.manual_seed(0)
        self.conv1 = GCNConv(num_node_features, conv_channels)
        self.conv2 = GCNConv(conv_channels, conv_channels)
        self.lin = Linear(conv_channels*3, class_channels)
        self.lin2 = Linear(class_channels, 1)

    def forward(self, x, edge_index, pos, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = torch.cat((
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
            global_add_pool(x, batch)
        ), 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.lin(x)
        x = x.relu()
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.lin2(x)
        x = expit(x)  # logistics function

        return x


def train(model, datas, optimizer):
    model.train()
    dl = DataLoader(dataset=datas, batch_size=50)
    accuracy = torch.zeros(len(dl))
    losss = torch.zeros(len(dl))
    # Iterate in batches over the training/test dataset.
    for i_d, data in enumerate(dl):
        out = model(data.x, data.edge_index, data.pos, data.batch)
        accuracy[i_d] = torch.mean(1 - torch.abs(torch.round(out) - data.y))
        loss = LOSSFN(out.flatten(), data.y)
        losss[i_d] = loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    loss_overall = torch.mean(losss)
    return float(torch.mean(accuracy)), float(loss_overall)


def test(model, datas):
    model.eval()
    dl = DataLoader(dataset=datas, batch_size=1)
    accuracy = torch.zeros(len(dl))
    losss = torch.zeros(len(dl))
    # Iterate in batches over the training/test dataset.
    for i_d, data in enumerate(dl):
        out = model(data.x, data.edge_index, data.pos, data.batch)
        accuracy[i_d] = 1 - torch.abs(torch.round(out) - data.y)
        losss[i_d] = LOSSFN(out.flatten(), data.y)
    return float(torch.mean(accuracy)), float(torch.mean(losss))


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_fname', type=str, default="my_model.torch", )
    # parser.add_argument(
    #     '-t', '--model_type', choices=[
    #         CLASSIFICATION_STR,
    #         CONVRNN_STR
    #     ])
    parser.add_argument(
        'fnames_read_pkl', type=str, nargs='+')
    args = parser.parse_args()
    fnames_read_pkl: List[str] = args.fnames_read_pkl
    print(f'fnames_read_pkl: {fnames_read_pkl}')
    model_fname: str = args.model_fname
    print(f'model_fname: {model_fname}')
    # model_type: str = args.model_type
    # print(f'model_type: {model_type}')

    # meta params
    validation_split: float = .1  # of first file
    test_split: float = .1
    epochs = 100

    # data
    pb = ProgressBar("epochs * files", len(fnames_read_pkl)*epochs, 1)
    for i_e in range(epochs):
        for fname_read_pkl in fnames_read_pkl:
            # print("~"*60)
            # print(f"epoch {i_e+1} of {epochs}")
            # print(
            #     f"reading file {fnames_read_pkl.index(fname_read_pkl) + 1} of " +
            #     f"{len(fnames_read_pkl)} : " +
            #     f"{fname_read_pkl}")
            with open(fname_read_pkl, 'rb') as f:
                d = pickle.load(f)
            n = len(d)
            n_val = int(n * validation_split)
            # on first file only
            if fname_read_pkl == fnames_read_pkl[0] and i_e == 0:
                print(f'n: {n}')
                print(f'n_val: {n_val}')
                n_test = int(n*test_split)
                print(f'n_test: {n_test}')
                n_train = n - n_val - n_test
                print(f'n_train: {n_train}')
                train_graphs = [d[i] for i in range(n_train)]
                assert len(train_graphs) == n_train, "We must have all data."
                test_graphs = [d[i] for i in range(n_train, n_train+n_test)]
                val_graphs = [d[i] for i in range(n_train+n_test, n)]
                assert len(d) == len(train_graphs) + \
                    len(test_graphs) + len(val_graphs)
            else:
                n_train = n - n_val
                # print(f'n_train: {n_train}')
                train_graphs = [d[i] for i in range(n_train)]
                val_graphs = [d[i] for i in range(n_train, n)]
                assert len(d) == len(train_graphs) + len(val_graphs)

            # on first file only
            if fname_read_pkl == fnames_read_pkl[0] and i_e == 0:
                # info on data shape
                num_nodes = train_graphs[0].num_nodes
                num_edges = train_graphs[0].num_edges
                num_node_features = train_graphs[0].num_node_features
                print(f"train_graphs[0]: {train_graphs[0]}")
                print(f"len(train_graphs): {len(train_graphs)}")
                print(f"num_nodes: {num_nodes}")
                print(f"num_edges: {num_edges}")
                print(f"num_node_features: {num_node_features}")

                # create model
                model = GCN(
                    conv_channels=50,
                    class_channels=100,
                    num_node_features=num_node_features
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                # train
                training_accuracy: List[float] = []
                val_accuracy: Optional[List[float]] = []
                test_accuracy: List[float] = []
                loss: List[float] = []
                val_loss: Optional[List[float]] = []
                test_loss: List[float] = []
                test_x: List[float] = []

                # evaluating untrained model
                pretrain_test_accuracy, pretrain_test_loss = test(
                    model, test_graphs)
                print(f"pretrain_test_loss: {pretrain_test_loss}")
                print(f"pretrain_test_accuracy: {pretrain_test_accuracy}")
                test_x.append(0)
                test_accuracy.append(pretrain_test_accuracy)
                test_loss.append(pretrain_test_loss)
            # (if) on first file only
            random.shuffle(train_graphs)
            one_training_accuracy, one_training_loss = train(
                model, train_graphs, optimizer)
            training_accuracy.append(one_training_accuracy)
            loss.append(one_training_loss)
            pb.progress()

        # test after each epoch
        one_test_accuracy, one_test_loss = test(
            model, test_graphs)
        print(f"one_test_loss: {one_test_loss}")
        print(f"one_test_accuracy: {one_test_accuracy}")
        test_x.append(len(training_accuracy)-1)
        test_accuracy.append(one_test_accuracy)
        test_loss.append(one_test_loss)

    val_accuracy, val_loss = test(
        model, val_graphs)
    print(f"val_loss: {val_loss}")
    print(f"val_accuracy: {val_accuracy}")
    torch.save(model.state_dict(), model_fname)
    pb.end()

    # print history
    fig, axs = plt.subplots(2)
    axs[0].plot(training_accuracy, label="training_accuracy")
    axs[1].plot(loss, label="training_loss")
    if val_accuracy is not None:
        axs[0].plot(
            [min(test_x), max(test_x)],
            [val_accuracy]*2,
            "--",
            label="val_accuracy")
    if val_loss is not None:
        axs[1].plot(
            [min(test_x), max(test_x)],
            [val_loss]*2,
            "--",
            label="val_loss")
    axs[0].plot(test_x, test_accuracy, label="test_accuracy")
    axs[1].plot(test_x, test_loss, label="test_loss")
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Batch')
    axs[1].legend(loc='upper right')
    axs[1].set_xlabel('Batch')
    fig.savefig("training_history.png")
    plt.show()
