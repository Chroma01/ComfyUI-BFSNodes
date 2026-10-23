# Extracted from official ComfyUI bef11fc nodes_sparse_attention.py for loader/closure compatibility tests.
# SPDX-License-Identifier: GPL-3.0-only
class SparseAttnPatch:
    """Options plus runtime state for one patched model; an ON_CLEANUP callback
    resets the state when the sampling run ends."""

    def __init__(self, tau, topk_ratio, vsa, sigma_start, sigma_end, min_tokens, dense_blocks, sink_conditioning, extra_tokens, verbose):
        self.tau = tau
        self.topk_ratio = topk_ratio
        self.extra_tokens = extra_tokens
        self.vsa = vsa
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.min_tokens = min_tokens
        self.dense_blocks = dense_blocks
        self.sink_conditioning = sink_conditioning
        self.verbose = verbose
        self.installed = set()
        self.reset()

    def reset(self):
        self.pooled = {}
        self.vsa_plans = {}
        self.vsa_rope = None
        self._logged = set()

    def log_once(self, key, message):
        if self.verbose and key not in self._logged:
            self._logged.add(key)
            logging.info(f'BlockSparseAttention: {message}')

    def dense_reason(self, transformer_options, tokens, block_index):
        """Why this call stays dense regardless of its tensors, or None."""
        sigmas = transformer_options.get('sigmas')
        if sigmas is not None:
            sigma = float(sigmas[0])
            if sigma > self.sigma_start or sigma < self.sigma_end:
                return f'sigma {sigma:.3g} outside the start/end window'
        if tokens < self.min_tokens:
            return f'{tokens} tokens < min_tokens {self.min_tokens}'
        if self.dense_blocks:
            if block_index is None:
                self.log_once('no_block_index', 'this model does not report block indices; dense_blocks ignored')
            elif block_index in self.dense_blocks:
                return f'block {block_index} in dense_blocks'
        return None

    def sinks(self, transformer_options, tokens):
        """MiniMax-H3 conditioning rows as (exact-KV blocks, dense-query blocks):
        the packed prefix stays exact for every query, and optionally the
        target-audio query rows run dense."""
        layout = transformer_options.get('minimax_h3_layout')
        if self.sink_conditioning == 'off' or layout is None or layout.seq_len != tokens:
            return ((0, 0), (0, 0))
        video = next(((a, b) for a, b, kind in layout.segments if kind == 'video'), None)
        if video is None or video[0] <= 0:
            return ((0, 0), (0, 0))
        blocks = (0, (video[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if self.sink_conditioning != 'exact_kv_and_rows':
            return (blocks, (0, 0))
        audio = next(((a, b) for a, b, kind in layout.segments if kind == 'audio'), None)
        if audio is None:
            return (blocks, blocks)
        return (blocks, (audio[0] // BLOCK_SIZE, blocks[1]))

    def vsa_plan(self, layout, device):
        """Padded tile order: prefix segments in their own zero-padded 64-row
        tiles, video in 4x4x4 cubes. `src` maps padded row -> source row (-1 =
        pad), `inv` the reverse."""
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        plan = self.vsa_plans.get(key)
        if plan is not None:
            return plan
        _text_len, latent_t, latent_h, latent_w, _audio_t = layout.signature
        grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
        tiles, n_prefix = ([], 0)
        for a, b, kind in layout.segments:
            n = b - a
            if kind != 'video':
                m = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
                seg = torch.full((m * BLOCK_SIZE,), -1, dtype=torch.int64, device=device)
                seg[:n] = torch.arange(a, b, device=device)
                tiles.append(seg.view(m, BLOCK_SIZE))
                n_prefix += m
                continue
            if grid[0] * grid[1] * grid[2] != n:
                raise RuntimeError(f'VSA: video segment of {n} rows does not match the latent grid {grid}')
            ct, ch, cw = VSA_CUBE
            pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, VSA_CUBE))
            padded = torch.full((pt, ph, pw), -1, dtype=torch.int64, device=device)
            padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b, device=device).view(*grid)
            cubes = padded.view(pt // ct, ct, ph // ch, ch, pw // cw, cw).permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK_SIZE)
            order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
            tiles.append(torch.gather(cubes, 1, order))
        tiles = torch.cat(tiles)
        src = tiles.reshape(-1)
        live = src >= 0
        inv = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
        inv[src[live]] = torch.nonzero(live).flatten()
        plan = {'n': int(src.numel()), 'n_prefix': n_prefix, 'src': src, 'inv': inv, 'block_len': (tiles >= 0).sum(1).to(torch.int32)}
        while len(self.vsa_plans) >= VSA_PLAN_CACHE:
            del self.vsa_plans[next(iter(self.vsa_plans))]
        self.vsa_plans[key] = plan
        return plan

    def vsa_rope_freqs(self, rope_freqs, plan):
        hit = self.vsa_rope
        if hit is not None and hit[0]() is rope_freqs and (hit[1] is plan):
            return hit[2]
        padded = rope_freqs.new_zeros((1, plan['n']) + tuple(rope_freqs.shape[2:]))
        padded[0, plan['inv']] = rope_freqs[0]
        self.vsa_rope = (weakref.ref(rope_freqs), plan, padded)
        return padded

def make_attention_override(patch: SparseAttnPatch, previous):
    """Attention override; declined calls run ``previous`` (the override that was
    on the hook before this one) or ``func``. Dense-only in VSA mode: a
    VSA-trained model must never see plain block-sparse attention."""

    def override(func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        transformer_options = kwargs.get('transformer_options') or {}

        def dense():
            args = (q, k, v, heads)
            kw = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            return func(*args, **kw) if previous is None else previous(func, *args, **kw)
        if mask is not None or patch.vsa:
            return dense()
        tokens = q.shape[2] if skip_reshape else q.shape[1]
        reason = patch.dense_reason(transformer_options, tokens, transformer_options.get('block_index'))
        if reason is not None:
            patch.log_once(('dense', tokens, reason), f'dense ({tokens} tokens): {reason}')
            return dense()
        if skip_reshape:
            b, _, _, dim_head = q.shape
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            qs, ks, vs = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        reason = _ineligible(qs, ks, vs, dim_head)
        if reason is not None:
            patch.log_once(('ineligible', tuple(qs.shape), reason), f'dense {tuple(qs.shape)}: {reason}')
            return dense()
        sink, sink_q = patch.sinks(transformer_options, tokens)
        if q.dtype == torch.float32:
            qs, ks, vs = (t.to(torch.bfloat16) for t in (qs, ks, vs))
        out = ck.sol_attn(qs, ks, vs, tau=patch.tau, scale=kwargs.get('scale'), sink_blocks=list(sink), sink_q=list(sink_q), topk_ratio=patch.topk_ratio, token_aug=patch.extra_tokens).to(q.dtype)
        patch.log_once(('sparse', tuple(qs.shape)), f'sparse {tuple(qs.shape)}, sinks {sink}/{sink_q}')
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, heads * dim_head)
    return override

def make_h3_block_patch(block, block_index, patch: SparseAttnPatch):
    """Runs the block with its attention swapped for the sparse producer."""

    def attention(h, rope_freqs=None, transformer_options={}):
        return h3_sparse_attention(block.attn, h, rope_freqs, transformer_options, patch, block_index)

    def block_patch(args, extra):
        if h3_eligible(block.attn, args['img'], args['rope_freqs'], args['transformer_options'], patch, block_index):
            args = {**args, 'attention': attention}
        return extra['original_block'](args)
    return block_patch
