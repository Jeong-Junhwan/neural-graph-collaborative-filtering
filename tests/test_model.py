import pytest
from model import NGCF, BPR
from preprocess import load_train_data

import torch


@pytest.fixture()
def train_data():
    return load_train_data()


def test_NGCF(train_data):
    train_data, data_info, laplacian_matrix = train_data
    model = NGCF(
        n_users=data_info["users"],
        n_items=data_info["items"],
        embedding_dim=16,
        laplacian_matrix=laplacian_matrix,
    )
    data = torch.LongTensor([[1, 5], [5, 7], [5, 1], [4, 2]])
    u = data[:, 0]
    i = data[:, 1]

    assert len(model.forward(u, i)) == 4


def test_BPR(train_data):
    train_data, data_info, laplacian_matrix = train_data
    model = BPR(
        n_users=data_info["users"],
        n_items=data_info["items"],
        embedding_dim=16,
        laplacian_matrix=laplacian_matrix,
    )
    data = torch.LongTensor([[1, 5, 6], [2, 5, 7], [3, 5, 1], [4, 5, 2]])

    loss = model.forward(data)
    loss.backward()
