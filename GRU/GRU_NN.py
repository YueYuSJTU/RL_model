from typing import Optional
import gymnasium as gym

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Embedder(nn.Module):
    def __init__(self, input_length, embed_dim, activation=nn.ReLU):
        super(Embedder, self).__init__()
        if activation:
            self.embedder = nn.Sequential(
                nn.Linear(input_length, embed_dim),
                activation(),
                # nn.Linear(embed_dim, embed_dim),
                # activation(),
                nn.LayerNorm(embed_dim),
            )
        else:
            self.embedder = nn.Sequential(
                nn.Linear(input_length, embed_dim),
                nn.LayerNorm(embed_dim),
            )

    def forward(self, x):
        return self.embedder(x)

class RNNEncoder(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=128,
        embed_dim=128,
        rnn_hidden_size=128,
        num_layers=1,
        dropout=None,
        encoder="GRU",
        obs_length=15,
    ):
        super(RNNEncoder, self).__init__(observation_space, features_dim)
        self.observation_space = observation_space
        # self.features_dim = features_dim
        self.embed_dim = embed_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout if dropout is not None else 0.0
        self.encoder = encoder
        self.obs_length = obs_length

        self.obs_encoder = Embedder(obs_length, embed_dim)
        if encoder == "GRU":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers,
                dropout=self.dropout,
                batch_first=True,
            )
        elif encoder == "LSTM":
            self.rnn = nn.LSTM(
                input_size=embed_dim,
                hidden_size=rnn_hidden_size,
                num_layers=num_layers,
                dropout=self.dropout,
                batch_first=True,
            )
        else:
            raise NotImplementedError
        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.final_encoder = Embedder(rnn_hidden_size, features_dim, activation=None)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch_size, sequence_size, obs_length)
        # print(f"observations: {observations.shape}")

        embedded_obs = self.obs_encoder(observations)
        # print(f"embedded_obs: {embedded_obs.shape}")
        # embedded_obs: (batch_size, sequence_size, embed_dim)

        rnn_output, _ = self.rnn(embedded_obs)
        # print(f"rnn_output: {rnn_output.shape}")
        # rnn_output: (batch_size, sequence_size, rnn_hidden_size)

        final_state = rnn_output[:, -1, :]
        # print(f"final_state: {final_state.shape}")
        # final_state: (batch_size, rnn_hidden_size)

        result = self.final_encoder(final_state)
        # result: (batch_size, features_dim)

        return result


if __name__ == "__main__":
    batch_size = 32
    sequence_size = 5
    obs_length = 15
    embed_dim = 128
    rnn_hidden_size = 128

    obs = torch.rand(batch_size, sequence_size, obs_length)
    print(obs.shape)
    rnn = RNNEncoder(
        observation_space=gym.spaces.Box(low=-1, high=1, shape=(batch_size, sequence_size, obs_length)),
        features_dim=128,
        embed_dim=128,
        rnn_hidden_size=128,
        num_layers=1,
        dropout=None,
        encoder="GRU",
        obs_length=15,
    )
    result = rnn(obs)
    print(result.shape)