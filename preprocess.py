import torch
import pandas as pd
import numpy as np
from typing import Tuple, Dict


col_names = ["user_id", "item_id", "rating", "timestamp"]
dtypes = {
    "user_id": np.int64,
    "item_id": np.int64,
    "rating": np.int64,
    "timestamp": np.int64,
}


def load_raw_data() -> Tuple[pd.DataFrame, Dict[str, int]]:
    # read raw data
    inter_data = pd.read_csv("ml-100k/u.data", sep="\t", names=col_names, dtype=dtypes)
    data_info = dict()
    with open("ml-100k/u.info", "r") as file:
        for line in file:
            n, what = line.split()
            data_info[what] = int(n)

    # convert to implicit feedback
    inter_data = inter_data[["user_id", "item_id"]]
    return inter_data, data_info


def user_item_2_node(
    inter_data: pd.DataFrame, data_info: Dict[str, int]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    inter_data["user_id"] -= 1
    inter_data["item_id"] += data_info["users"] - 1

    return inter_data, data_info


def generate_negative_samples(
    inter_data: pd.DataFrame, data_info: Dict[str, int]
) -> Tuple[pd.Series, pd.Series]:
    # find negative samples (don't have interaction with each user)
    inter_data = inter_data.groupby("user_id")["item_id"].apply(lambda x: x.to_numpy())
    non_inter_data = inter_data.copy()
    non_inter_data = non_inter_data.apply(
        lambda x: np.array(
            list(set(range(1, data_info["items"] + 1)) - set(x)), dtype=np.int64
        )
    )

    # u, [i1, i2, ,,,]
    # u, [j1, j2, ,,,]
    return inter_data, non_inter_data


def convert2uij(
    inter_data: pd.Series, non_inter_data: pd.Series, data_info: Dict[str, int]
) -> pd.DataFrame:
    inter_data.name = "positive"
    non_inter_data.name = "negative"
    combined_data = pd.concat([inter_data, non_inter_data], axis=1).reset_index()
    combined_data = combined_data.explode("positive").reset_index(drop=True)

    # u1, i1, [j1, j2, j3, ,,,]
    # u1, i2, [j1, j2, j3, ,,,]
    return combined_data


def get_adj_matrix(train_data: pd.DataFrame, data_info: Dict[str, int]) -> torch.Tensor:
    adj_matrix = torch.zeros(
        (
            data_info["users"] + data_info["items"],
            data_info["users"] + data_info["items"],
        )
    )

    for row in train_data.itertuples():
        adj_matrix[row[1], row[2]] = 1
        adj_matrix[row[2], row[1]] = 1

    return adj_matrix


def adj_to_laplacian(matrix: torch.Tensor) -> torch.Tensor:
    N_t_inverse = torch.pow(matrix.sum(dim=0), -0.5)
    diag_matrix_inverse = torch.mul(torch.eye(matrix.shape[1]), N_t_inverse)

    return torch.mm(torch.mm(diag_matrix_inverse, matrix), diag_matrix_inverse)


def load_train_data() -> Tuple[pd.DataFrame, Dict[str, int], torch.Tensor]:
    inter_data, data_info = load_raw_data()
    inter_data, data_info = user_item_2_node(inter_data, data_info)
    inter_data, non_inter_data = generate_negative_samples(inter_data, data_info)
    train_data = convert2uij(inter_data, non_inter_data, data_info)
    adj_matrix = get_adj_matrix(train_data, data_info)
    laplacian_matrix = adj_to_laplacian(adj_matrix)

    return train_data, data_info, laplacian_matrix
