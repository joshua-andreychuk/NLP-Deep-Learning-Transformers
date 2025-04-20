"""
Originally forked from Andrej Karpathy's minGPT.

XCS224N : Homework 5

John Hewitt <johnhew@stanford.edu>
Ansh Khurana <anshk@stanford.edu>
Soumya Chatterjee <soumyac@stanford.edu>
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def precompute_rotary_emb(dim, max_positions):
    """
    RoPE uses the following sinusoidal functions to encode positions:

    cos(t theta_i) and sin(t theta_i)
        where t is the position and
              theta_i = 1/10000^(2(i-1)/dim) for i in [1, dim/2]

    Since the maximum length of sequences is known, we can precompute
    these values to speed up training.

    Implement the precompute_rotary_emb function that returns a tensor of
    shape (max_positions, dim/2, 2) where the last dimension contains
    the cos and sin values for each position and each dimension of
    the embedding.
    """

    rope_cache = None

    ### TODO:
    ### [part h]
    ### START CODE HERE
    half_dim = dim // 2
    rope_cache = torch.zeros(max_positions, half_dim, 2)

    # Compute frequencies for each dimension
    # This matches the typical formula 1 / (10000^(2i/dim)) for i in [0 .. half_dim-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float) * 2.0 / dim))

    # Positions from [0..max_positions-1]
    positions = torch.arange(0, max_positions, dtype=torch.float).unsqueeze(1)  # [max_positions, 1]

    # Compute angles [max_positions, half_dim]
    angles = positions * inv_freq

    # rope_cache has shape (max_positions, half_dim, 2),
    # storing cos in the last dim's 0 index, sin in index 1
    rope_cache[:, :, 0] = torch.cos(angles)
    rope_cache[:, :, 1] = torch.sin(angles)
    ### END CODE HERE
    return rope_cache


def apply_rotary_emb(x, rope_cache):
    """Apply the RoPE to the input tensor x."""
    # [part h]
    # You might find the following functions useful to convert
    # between real and complex numbers:
    #
    # torch.view_as_real - https://pytorch.org/docs/stable/generated/torch.view_as_real.html
    # torch.view_as_complex - https://pytorch.org/docs/stable/generated/torch.view_as_complex.html
    #
    # Note that during inference, the length of the sequence might be different
    # from the length of the precomputed values. In this case, you should
    # truncate the precomputed values to match the length of the sequence.

    rotated_x = None
    ### TODO:
    ### [part h]
    ### START CODE HERE
    B, nH, T, hs = x.shape  # e.g. (batch, num_heads, seq_len, head_size)
    # Rope cache shape: (max_positions, hs, 2) => we only need up to T for positions
    # also ensure we only use up to hs//2 in case each head is smaller than dim//2
    rope_slice = rope_cache[:T, : (hs // 2), :]  # shape => (T, hs//2, 2)

    # Convert x into real+imag form
    # Reshape x to (B, nH, T, hs//2, 2) so it can be viewed as a complex tensor
    x_reshaped = x.view(B, nH, T, hs // 2, 2)
    x_complex = torch.view_as_complex(x_reshaped)

    # Convert rope_slice into a complex representation
    # shape => (T, hs//2, 2) -> unsqueeze to (1, 1, T, hs//2, 2) for broadcast
    rope_slice = rope_slice.unsqueeze(0).unsqueeze(0)  # shape => (1, 1, T, hs//2, 2)
    rope_complex = torch.view_as_complex(rope_slice)

    # Apply RoPE by multiplying the complex representations
    rotated_x_complex = x_complex * rope_complex

    # Convert back to real form => shape (B, nH, T, hs//2, 2)
    rotated_x_reshaped = torch.view_as_real(rotated_x_complex)

    # Finally reshape back to (B, nH, T, hs)
    rotated_x = rotated_x_reshaped.view(B, nH, T, hs)
    ### END CODE HERE
    return rotated_x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.rope = config.rope
        if self.rope:
            assert (config.n_embd % config.n_head) % 2 == 0

            ### TODO:
            # [part h] Precompute the cos and sin values for RoPE and
            # store them in rope_cache.
            # Hint: The maximum sequence length is given by config.block_size.
            rope_cache = None
            ### START CODE HERE
            rope_cache = precompute_rotary_emb(config.n_embd // config.n_head, config.block_size)
            ### END CODE HERE

            self.register_buffer("rope_cache", rope_cache)

        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        # (B x T x C) is of dimension (batch x block_size x n_embd) which is (batch x l x d) in the handout.
        # nh should be number_of_heads, and hs would then stand for n_embed (or "dimensionality" d in the handout) per head

        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.rope:
            pass
            ### TODO:
            # [part h] Apply RoPE to the query and key.
            ### START CODE HERE
            q = apply_rotary_emb(q, self.rope_cache)
            k = apply_rotary_emb(k, self.rope_cache)
            ### END CODE HERE

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e10)
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class CausalCrossAttention(nn.Module):
    """
    Modifications over the self-attention layer to handle two inputs and perform
    cross-attention between them.
    This follows the implementation of the self attention module with
    auto-regressive masking on (key).
    Manipulation of batch-size to allow for different batch size between the 
    two inputs, with broadcasting over to the higher batch size value.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x_kv, x_q):
        Bk, Tk, Ck = x_kv.size()
        Bq, Tq, Cq = x_q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        
        # keys of x1
        k = self.key(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)
        
        # query with x2
        q = self.query(x_q).view(Bq, Tq, self.n_head, Cq // self.n_head).transpose(1, 2) # (B, nh, Tq, hs)
        
        # values from x1
        v = self.value(x_kv).view(Bk, Tk, self.n_head, Ck // self.n_head).transpose(1, 2) # (B, nh, Tk, hs)

        # causal self-attention;  (B, nh, Tk, hs) x (B, nh, hs, Tq) -> (B, nh, Tq, Tk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        B = max(Bk, Bq)
        
        att = att.masked_fill(self.mask[:,:,:Tq,:Tk] == 0, -1e10) 
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, Tq, Tk) x (B, nh, Tk, hs) -> (B, nh, Tq, hs)
        y = y.transpose(1, 2).contiguous().view(B, Tq, Cq) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
