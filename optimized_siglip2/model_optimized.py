# --------------------  baseline model (same as in prompt) --------------------
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .ops import LigerLayerNormFunction, mlp_func

torch._inductor.config.realize_opcount_threshold = 500
torch.backends.cuda.matmul.allow_tf32 = True

torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}


class LayerNormImproved(nn.Module):
    def __init__(self, hidden_size, eps, bias=True, init_fn="ones"):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.bias = nn.Parameter(
            torch.randn(hidden_size) if bias else torch.zeros(hidden_size)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return LigerLayerNormFunction.apply(
            hidden_states, self.weight, self.bias, self.variance_epsilon
        )

    def extra_repr(self):
        return f"{self.hidden_size}, eps={self.eps}"


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


class Siglip2SequenceEmbeddingsImproved(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = torch.compile(
            nn.Linear(
                in_features=config.num_channels * self.patch_size * self.patch_size,
                out_features=self.embed_dim,
            ),
            options=torch_compile_options,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = torch.compile(
            nn.Embedding(self.num_patches, self.embed_dim),
            options=torch_compile_options,
        )

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


class Siglip2AttentionImproved(nn.Module):
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

        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # Adapted from Siglip2Attention.forward and transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward
    def forward(self, hidden_states, block_mask=None, output_attentions=False):
        batch_size, seq_len, _ = hidden_states.size()
        # 1. Linear projections
        qkv = self.qkv_proj(hidden_states)
        Q, K, V = qkv.chunk(3, dim=-1)

        # 2. Reshape into [B, heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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


class MLPImproved(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return mlp_func(hidden_states, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)


class Siglip2EncoderLayerImproved(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = torch.compile(
            Siglip2AttentionImproved(config), options=torch_compile_options
        )
        self.layer_norm1 = torch.compile(
            LayerNormImproved(self.embed_dim, eps=config.layer_norm_eps),
            options=torch_compile_options,
        )
        self.mlp = torch.compile(MLPImproved(config), options=torch_compile_options)
        self.layer_norm2 = torch.compile(
            LayerNormImproved(self.embed_dim, eps=config.layer_norm_eps),
            options=torch_compile_options,
        )

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask,
    ) -> tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            block_mask=block_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2EncoderImproved(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                Siglip2EncoderLayerImproved(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds,
        document_ids: torch.Tensor | None = None,
    ):
        hidden_states = inputs_embeds

        def document_mask(b, h, q_idx, kv_idx):
            return document_ids[q_idx] == document_ids[kv_idx]

        _, seq_len, _ = hidden_states.size()

        block_mask = torch.compile(create_block_mask, options=torch_compile_options)(
            document_mask, 1, 1, seq_len, seq_len
        )

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                block_mask,
            )

        return hidden_states


class Siglip2SequenceVisionTransformerOptimized(nn.Module):
    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = torch.compile(
            Siglip2SequenceEmbeddingsImproved(config), options=torch_compile_options
        )
        self.encoder = torch.compile(
            Siglip2EncoderImproved(config), options=torch_compile_options
        )
        self.post_layernorm = torch.compile(
            LayerNormImproved(config.hidden_size, eps=config.layer_norm_eps),
            options=torch_compile_options,
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


# -----------------------------------------------------------------------------
