"""
model.py — Faithful DeepSeek Engram implementation (Phase 1)
============================================================
Implements the official DeepSeek Engram architecture from
arXiv:2601.07372 and engram_demo_v1.py.

Key components:
  - CompressedTokenizer: vocabulary normalization + dedup
  - NgramHashMapping: multi-head, multi-order, prime-sized hashing
  - MultiHeadEmbedding: offset-based flat embedding table
  - ShortConv: depthwise causal conv with dilation=max_ngram_size
  - Engram: the full module with per-branch similarity gating
  - EngramEnhancedModel: wraps base LM and registers read hooks
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from tokenizers import normalizers, Regex
from transformers import AutoTokenizer


# ----------------------------------------------------------------------
# 1. Compressed Tokenizer
# ----------------------------------------------------------------------
class CompressedTokenizer:
    """
    Normalizes and deduplicates the tokenizer vocabulary.
    Maps all tokens that normalize alike to a single compressed ID.
    """
    def __init__(self, tokenizer_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []
        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text
            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]
        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


# ----------------------------------------------------------------------
# 2. N-gram Hash Mapping
# ----------------------------------------------------------------------
class NgramHashMapping:
    """
    Multi-head, multi-order N-gram hashing with layer-specific multipliers
    and prime-sized disjoint bucket ranges.
    """
    def __init__(self, engram_vocab_size, max_ngram_size, n_embed_per_ngram,
                 n_head_per_ngram, layer_ids, tokenizer_name_or_path,
                 pad_id, seed):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound,
                           size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _find_next_prime(self, start, seen_primes):
        candidate = start + 1
        while True:
            if isprime(candidate) and candidate not in seen_primes:
                return candidate
            candidate += 1

    def _calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                for _ in range(num_head):
                    found_prime = self._find_next_prime(
                        current_prime_search_start, seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids, layer_id):
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k):
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)), mode="constant",
                constant_values=self.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers


# ----------------------------------------------------------------------
# 3. Multi-Head Embedding
# ----------------------------------------------------------------------
class MultiHeadEmbedding(nn.Module):
    """
    Single flat embedding table with offset-based indexing across all heads.
    """
    def __init__(self, list_of_N, D):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids):
        shifted_input_ids = input_ids + self.offsets
        return self.embedding(shifted_input_ids)


# ----------------------------------------------------------------------
# 4. Short Convolution
# ----------------------------------------------------------------------
class ShortConv(nn.Module):
    """
    Depthwise causal conv1d with dilation = max_ngram_size.
    Operates over all hyper-connection branches jointly.
    """
    def __init__(self, hidden_size, kernel_size=4, dilation=1,
                 norm_eps=1e-5, hc_mult=4, activation=True):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        # Conv zero-init for identity graft
        nn.init.zeros_(self.conv.weight)
        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)
        ])
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x):
        """x: (B, T, HC_MULT, D)"""
        B, T, G, C = x.shape
        assert G == self.hc_mult
        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y


# ----------------------------------------------------------------------
# 5. Engram Module
# ----------------------------------------------------------------------
class Engram(nn.Module):
    """
    The full Engram module for a single layer.
    """
    def __init__(self, layer_id, hidden_size, engram_vocab_size,
                 max_ngram_size=3, n_embed_per_ngram=512,
                 n_head_per_ngram=8, tokenizer_name_or_path="Qwen/Qwen2.5-1.5B",
                 pad_id=0, seed=0, hc_mult=1, kernel_size=4):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.disabled = False

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=[layer_id],
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
        )

        # Flattened list of all head bucket sizes
        flat_N = []
        for head_sizes in self.hash_mapping.vocab_size_across_layers[layer_id]:
            flat_N.extend(head_sizes)

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=flat_N,
            D=n_embed_per_ngram // n_head_per_ngram,
        )

        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            dilation=max_ngram_size,
            hc_mult=hc_mult,
        )

        engram_hidden_size = (max_ngram_size - 1) * n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, hidden_size) for _ in range(hc_mult)
        ])
        # Zero-init for identity graft at step 0
        nn.init.zeros_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        for proj in self.key_projs:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(hidden_size) for _ in range(hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(hidden_size) for _ in range(hc_mult)
        ])

    def forward(self, hidden_states, input_ids):
        """
        hidden_states: (B, L, D)  — if hc_mult=1, else (B, L, HC_MULT, D)
        input_ids: (B, L)
        """
        if self.disabled:
            return torch.zeros_like(hidden_states)     # ← add these 2 lines
        # Ensure 4D for hyper-connection handling
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.unsqueeze(2)  # (B, L, 1, D)

        B, L = input_ids.shape
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids.detach().cpu().numpy())[self.layer_id]
        ).to(hidden_states.device)

        # Lookup and flatten
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

        # Per-branch gating
        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)  # (B, L, HC_MULT, 1)
        self.last_gate = gates.detach().squeeze(-1).mean(dim=-1)  # (B, L)

        value = gates * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)

        # Squeeze back to 3D if hc_mult=1
        if self.hc_mult == 1:
            output = output.squeeze(2)

        return output


# ----------------------------------------------------------------------
# 6. Engram-Enhanced Model Wrapper
# ----------------------------------------------------------------------
class EngramEnhancedModel(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size=1536,
        engram_vocab_size=None,
        max_ngram_size=3,
        n_embed_per_ngram=512,
        n_head_per_ngram=8,
        read_layers=(2, 5),
        tokenizer_name_or_path="Qwen/Qwen2.5-1.5B",
        pad_id=0,
        seed=0,
    ):
        super().__init__()
        self.base_model = base_model
        self.read_layers = read_layers
        self._current_input_ids = None
        self._hooks = []

        if engram_vocab_size is None:
            # Default: ~5x compressed vocab per n-gram order
            # Qwen2.5-1.5B vocab ≈ 151,936
            engram_vocab_size = [151936 * 5, 151936 * 5]

        self.memory_layers = nn.ModuleDict()
        for layer_idx in read_layers:
            self.memory_layers[str(layer_idx)] = Engram(
                layer_id=layer_idx,
                hidden_size=hidden_size,
                engram_vocab_size=engram_vocab_size,
                max_ngram_size=max_ngram_size,
                n_embed_per_ngram=n_embed_per_ngram,
                n_head_per_ngram=n_head_per_ngram,
                tokenizer_name_or_path=tokenizer_name_or_path,
                pad_id=pad_id,
                seed=seed,
                hc_mult=1,  # Qwen has no hyper-connections
                kernel_size=4,
            )

        self.setup_hooks()

    def setup_hooks(self):
        def make_hook(mem):
            def hook(module, args, output):
                if self._current_input_ids is None:
                    return output
                states = output[0] if isinstance(output, tuple) else output
                mem_out = mem(states, self._current_input_ids)
                mixed = states + mem_out
                return (mixed,) + output[1:] if isinstance(output, tuple) else mixed
            return hook

        # Handle PeftModel wrapper
        try:
            from peft import PeftModel
            raw = self.base_model
            if isinstance(raw, PeftModel):
                raw = raw.base_model.model
        except ImportError:
            raw = self.base_model

        layers = raw.model.layers
        for layer_idx, mem in self.memory_layers.items():
            idx = int(layer_idx)
            if idx < len(layers):
                self._hooks.append(
                    layers[idx].register_forward_hook(make_hook(mem))
                )
            else:
                print(f"  [WARNING] Read layer {idx} out of range, skipping.")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self._current_input_ids = input_ids
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def reset_cache(self):
        pass

    def clear_step_cache(self):
        pass
