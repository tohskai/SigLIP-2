# --------------------  baseline model (same as in prompt) --------------------
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

torch._inductor.config.realize_opcount_threshold = 500


def create_document_ids(seq_sizes: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Create a tensor of document indices [0, 1, ..., n-1]
    doc_ids = torch.arange(seq_sizes.size(0), device=device)
    # Repeat each document id according to its corresponding sequence size
    repeated = torch.repeat_interleave(doc_ids, seq_sizes)

    # Determine current length and the next multiple of 128
    orig_length = repeated.numel()
    padded_length = (
        (orig_length + 127) // 128
    ) * 128  # ceiling to nearest multiple of 128

    # If no padding is needed, return the result directly
    if padded_length == orig_length:
        return repeated

    # Create padding values.
    # We choose values from orig_length to padded_length - 1,
    # which will be unique and will not conflict with the original doc_ids.
    pad = torch.arange(
        orig_length, padded_length, dtype=repeated.dtype, device=repeated.device
    )

    # Concatenate the repeated ids with the padding
    return torch.cat([repeated, pad])


@dataclass
class Siglip2VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int = 16
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.1
    num_patches: int = 256


class Siglip2SequenceEmbeddings(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    def forward(
        self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Forward pass through the embeddings layer.

        Args:
            packed_seq_patches: Tuple containing:
                - seq_patches: [seq_len, patch_embed] - All patches from all images concatenated
                - seq_sizes: [num_images] - Number of patches for each image
                - token_grids: [num_images, 1, 1] - Grid of tokens for each image

        Returns:
            torch.Tensor: The embedded patches with positional embeddings added.
        """
        seq_patches, seq_sizes, spatial_shapes = packed_seq_patches

        # Apply patch embeddings
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(seq_patches.to(dtype=target_dtype))

        # Prepare positional embeddings grid: (1, embed_dim, h, w)
        positional_embeddings = (
            self.position_embedding.weight.reshape(
                self.position_embedding_size, self.position_embedding_size, -1
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        # Build a list of positional embeddings for each image
        pos_embeds_list = []
        for i in range(len(seq_sizes)):
            height, width = spatial_shapes[i]
            resized_pos_embed = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # Reshape from (1, embed_dim, height, width) to (height*width, embed_dim)
            resized_pos_embed = resized_pos_embed.reshape(
                self.embed_dim, height * width
            ).transpose(0, 1)
            pos_embeds_list.append(resized_pos_embed)

        # Concatenate all positional embeddings along the sequence dimension
        pos_embeds = torch.cat(pos_embeds_list, dim=0)

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + pos_embeds
        return embeddings


class Siglip2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # Adapted from Siglip2Attention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(self, hidden_states, document_ids=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.size()
        # 1. Linear projections
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        # 2. Reshape into [B, heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply FlexAttention with document mask
        def document_mask(b, h, q_idx, kv_idx):
            return document_ids[q_idx] == document_ids[kv_idx]

        block_mask = torch.compile(create_block_mask)(
            document_mask, 1, 1, seq_len, seq_len
        )

        attn_per_head = flex_attention(Q, K, V, block_mask=block_mask, scale=self.scale)
        # (Dropout on attn weights handled via mask_mod or by post-dropout below)
        # 4. (Optional) Dropout on output
        if self.training and self.dropout > 0:
            attn_per_head = nn.functional.dropout(
                attn_per_head, p=self.dropout, training=True
            )
        # 5. Merge heads and project
        attn_output = attn_per_head.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        # 6. Return output and None (since attention probs are not explicitly returned)
        return attn_output, None


class Siglip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2EncoderLayer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = Siglip2Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            document_ids=document_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2Encoder(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Siglip2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        document_ids: torch.Tensor | None = None,
    ):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                document_ids,
            )

        return hidden_states


class Siglip2SequenceVisionTransformer(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2SequenceEmbeddings(config)
        self.encoder = Siglip2Encoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, packed_seq_patches: tuple[torch.Tensor, torch.Tensor]):
        seq_patches, token_grids = packed_seq_patches
        seq_sizes = torch.prod(token_grids, dim=-1)

        # seq_patches: [seq_len, patch_embed]
        # seq_sizes: [block] such that sum(blocks) == seq_len where seq_sizes[0:1]
        # represent the image patches from the first image.

        # Get embeddings from packed sequence
        hidden_states = self.embeddings((seq_patches, seq_sizes, token_grids))

        # The encoder expects a batch dimension
        # Add a pseudo batch dimension for the encoder
        hidden_states = hidden_states.unsqueeze(0)

        # Generate the appropriate attention mask for packed sequence
        # Ensure mask has same dtype as hidden_states for compatibility with SDPA
        document_ids = create_document_ids(seq_sizes, hidden_states.device)

        # Pass through encoder
        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            document_ids=document_ids,
        )

        # Apply final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Remove the pseudo batch dimension we added earlier
        last_hidden_state = last_hidden_state.squeeze(0)

        # Return the full sequence of embeddings
        return last_hidden_state


# -----------------------------------------------------------------------------
