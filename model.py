import torch
from torch import nn
from torch.nn import functional as F


class NGCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        laplacian_matrix: torch.Tensor,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.initial_embedding = nn.Parameter(
            nn.init.kaiming_normal_(torch.zeros((n_users + n_items, embedding_dim)))
        )

        self.embedding_dim = embedding_dim
        self.register_buffer("laplacian_matrix", laplacian_matrix)
        # self.laplacian_matrix = laplacian_matrix
        self.n_layers = n_layers

        self.leaky_relu = nn.LeakyReLU()
        self.embedding_propagation_layers = self._make_layers()

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        # (Batch) * 2

        # (n_layers + 1, n_users + n_items, embedding_dim)
        embedding_per_layer = self._embedding_propagation()

        # (n_layers + 1, batch, embedding)
        user_vectors = embedding_per_layer[:, u, :]
        item_vectors = embedding_per_layer[:, i, :]

        # reshape for inner product
        # (batch, others)
        user_vectors = user_vectors.view(-1, (self.n_layers + 1) * self.embedding_dim)
        item_vectors = item_vectors.view(-1, (self.n_layers + 1) * self.embedding_dim)

        # (batch)
        inner_product = torch.mul(user_vectors, item_vectors).sum(dim=-1)

        return inner_product

    def _make_layers(self) -> nn.ModuleDict:
        embedding_propagation_layers = nn.ModuleDict()
        for i in range(1, self.n_layers + 1):
            temp_dict = dict()
            temp_dict[f"W1_{i}"] = nn.Linear(self.embedding_dim, self.embedding_dim)
            temp_dict[f"W2_{i}"] = nn.Linear(self.embedding_dim, self.embedding_dim)

            embedding_propagation_layers.update(temp_dict)

        return embedding_propagation_layers

    def _embedding_propagation(self) -> torch.Tensor:
        E_prev = self.initial_embedding
        embeddings = [E_prev]

        for i in range(1, self.n_layers + 1):
            W1 = self.embedding_propagation_layers[f"W1_{i}"]
            W2 = self.embedding_propagation_layers[f"W2_{i}"]

            L_I = self.laplacian_matrix + torch.eye(
                self.laplacian_matrix.shape[0], device=self.laplacian_matrix.device
            )

            left = W1(torch.mm(L_I, E_prev))
            right = torch.mm(self.laplacian_matrix, E_prev) * E_prev
            right = W2(right)

            E_next = self.leaky_relu(left + right)
            embeddings.append(E_next)

            E_prev = E_next

        embeddings = torch.stack(embeddings)

        return embeddings

    def _init_weight(self):
        pass


class BPR(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        laplacian_matrix: torch.Tensor,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.NGCF = NGCF(n_users, n_items, embedding_dim, laplacian_matrix, n_layers)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, 3)

        u = x[:, 0]
        i = x[:, 1]
        j = x[:, 2]

        xui = self.NGCF(u, i)
        xuj = self.NGCF(u, j)

        xuij = xui - xuj
        output = self.log_sigmoid(xuij)
        # to maximize BPR-OPT
        return -torch.mean(output)
