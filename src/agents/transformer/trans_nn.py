import torch
import torch.nn as nn
import math
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PositionalEncoding(nn.Module):
    """
    Helper class for adding positional encoding to the input embeddings.
    This allows the Transformer to understand the sequence order.
    Implementation is based on the PyTorch tutorials.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Transpose to (1, max_len, d_model) for easier addition with batch_first=True tensors
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using a Transformer architecture based on the provided diagram.

    It consists of:
    1. An Embedding Network (MLP) to project input features.
    2. A Positional Encoding layer to inject sequence information.
    3. A Transformer with 2 Encoder layers followed by 2 Decoder layers.
    4. A final Linear layer to produce the feature vector of desired size.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        # The output dimension of the final features vector is specified by features_dim.
        super().__init__(observation_space, features_dim)

        # Ensure the observation space is a 2D Box (sequence, features)
        assert len(observation_space.shape) == 2, (
            f"Expected a 2D observation space, but got {observation_space.shape}"
        )
        
        seq_len, in_features = observation_space.shape

        # --- Model Hyperparameters from the user's description ---
        d_model = 128  # The dimension of the transformer embeddings
        nhead = 4      # Number of heads in the multi-head attention
        d_ff = 128     # Hidden dimension of the feed-forward network
        n_encoders = 2 # Number of encoder layers
        n_decoders = 2 # Number of decoder layers
        dropout = 0.1  # A standard dropout rate

        # --- 1. Embedding Network ---
        # As described: two hidden layers with 64 neurons each (ReLU activation)
        # Input matches the observation space (in_features=58)
        # Output matches the transformer's model dimension (d_model=128)
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        # --- Positional Encoding ---
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)

        # --- 2. Encoder Network ---
        # PyTorch's TransformerEncoderLayer encapsulates one full encoder block
        # (Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Important for SB3: (Batch, Sequence, Features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoders
        )

        # --- 3. Decoder Network ---
        # PyTorch's TransformerDecoderLayer encapsulates one full decoder block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_decoders
        )
        
        # --- 4. Final Output Layer ---
        # The transformer's output will be a sequence of embeddings (batch, seq_len, d_model).
        # We flatten this sequence and project it to the final desired `features_dim`.
        flattened_dim = seq_len * d_model
        self.linear_out = nn.Linear(flattened_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Main forward pass for feature extraction.
        
        Args:
            observations: The input tensor from the environment.
                          Shape: (batch_size, seq_len, in_features) -> (N, 10, 58)
        
        Returns:
            A flattened feature tensor.
            Shape: (batch_size, features_dim)
        """
        
        # 1. Apply embedding network to each time step's features
        # Input: (N, 10, 58) -> Output: (N, 10, 128)
        embedded_obs = self.embedding_net(observations)
        
        # Add positional encoding to give the model sequence information
        # Input: (N, 10, 128) -> Output: (N, 10, 128)
        pos_encoded_obs = self.pos_encoder(embedded_obs)

        # 2. Pass through the stack of 2 Encoder layers
        # Input: (N, 10, 128) -> Output: (N, 10, 128)
        encoder_output = self.transformer_encoder(pos_encoded_obs)

        # 3. Pass through the stack of 2 Decoder layers
        # For this feature extraction task, the encoder's output serves as both
        # the target sequence (`tgt`) and the memory context (`memory`) for the decoder.
        # Input: tgt=(N, 10, 128), memory=(N, 10, 128) -> Output: (N, 10, 128)
        decoder_output = self.transformer_decoder(tgt=encoder_output, memory=encoder_output)
        
        # 4. Flatten the final sequence and project to the desired feature dimension
        # Flatten: (N, 10, 128) -> (N, 1280)
        flattened_output = torch.flatten(decoder_output, start_dim=1)
        
        # Final linear projection: (N, 1280) -> (N, features_dim)
        features = self.linear_out(flattened_output)
        
        return features

# --- Example Usage ---
if __name__ == '__main__':
    # Define an observation space that matches your environment
    # Box(-1, 1, (10, 58)) means a sequence of 10 items, each with 58 features.
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(10, 58))
    
    # Instantiate the feature extractor
    # You can choose the final output dimension, e.g., 256
    transformer_extractor = TransformerFeatureExtractor(observation_space=obs_space, features_dim=256)
    
    # Print the model to inspect its architecture
    print(transformer_extractor)
    
    # Create a dummy batch of observations (e.g., batch size of 4)
    dummy_obs = torch.randn(4, 10, 58)
    
    # Pass the dummy data through the extractor
    extracted_features = transformer_extractor(dummy_obs)
    
    # Check the output shape
    # It should be (batch_size, features_dim) -> (4, 256)
    print(f"\nInput shape: {dummy_obs.shape}")
    print(f"Output features shape: {extracted_features.shape}")

    # You would then pass this class to your SB3 policy definition, for example:
    # policy_kwargs = dict(
    #     features_extractor_class=TransformerFeatureExtractor,
    #     features_extractor_kwargs=dict(features_dim=256),
    # )
    # model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)