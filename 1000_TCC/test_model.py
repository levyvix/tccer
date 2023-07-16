from model.MLP import MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_metric_learning import losses, miners, distances, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from utils.common import get_all_embeddings, get_accuracy, log_to_file
import os

# import json
import random
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# load saved model (state dict)

model = MLP(embedding_size=20)

model.load_state_dict(torch.load("saved_history/last_model.pth"))

model = model.to("cuda")

# load test data

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(MEAN, STD), # TODO: discover
    ]
)

# train_dataset = MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = MNIST(root="data", train=False, download=True, transform=transform)

# reshape and normalize
# train_dataset.data = train_dataset.data.reshape(-1, 28 * 28).float() / 255.0

test_dataset.data = test_dataset.data.reshape(-1, 28 * 28).float() / 255.0

# train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# get embeddings

model = model.eval()

with torch.no_grad():
    embeddings = []
    labels = []

    for x, y in test_dataloader:
        x = x.to("cuda")

        out = model(x)

        embeddings.append(out.cpu().numpy())
        labels.append(y.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)


# Visualize embeddings
# PCA

pca = PCA(n_components=2)
pca.fit(embeddings)
embeddings = pca.transform(embeddings)

# plot
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab10")
plt.title("PCA projection of the digits dataset")
plt.colorbar()
plt.show()


# UMAP

# import umap

# reducer = umap.UMAP()

# embedding = reducer.fit_transform(embeddings)

# plt.add_subplot(1, 3, 2)
# plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10")
# plt.set_title("UMAP projection of the digits dataset")


# TSNE

# from sklearn.manifold import TSNE

# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

# tsne_results = tsne.fit_transform(embeddings)

# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap="tab10")
# plt.title("TSNE projection of the digits dataset")
# plt.colorbar()

# plt.show()
