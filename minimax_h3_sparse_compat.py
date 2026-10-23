"""Recognize only the official ComfyUI sol-attn wrappers and owned block hooks."""
import inspect


def official_sparse_module(obj):
    name = getattr(obj, '__module__', '')
    return name == 'nodes_sparse_attention' or name.endswith('.nodes_sparse_attention')


def unwrap_sol(override):
    if override is None or not official_sparse_module(override):
        return override, None
    if not getattr(override, '__qualname__', '').endswith('make_attention_override.<locals>.override'):
        raise ValueError('Unsupported sparse attention wrapper for native temporal alignment.')
    cells = inspect.getclosurevars(override).nonlocals
    patch = cells.get('patch')
    if patch is None or type(patch).__name__ != 'SparseAttnPatch' or not official_sparse_module(type(patch)):
        raise ValueError('Unable to identify the official sparse attention patch.')
    if patch.vsa or patch.topk_ratio != 0:
        raise ValueError('Native temporal alignment supports only sol-attn; SLA and VSA remain incompatible.')
    parent = cells.get('previous')
    if parent is not None and official_sparse_module(parent):
        raise ValueError('Use only one Model Sparse Attention node with native temporal alignment.')
    return parent, patch


def without_owned_blocks(options, patch):
    blocks = options.get('patches_replace', {}).get('dit', {})
    if not blocks:
        return options
    if patch is None:
        raise ValueError('Remove VSA and other attention/block replacement patches before native temporal alignment.')
    for callback in blocks.values():
        if (not official_sparse_module(callback) or
                not getattr(callback, '__qualname__', '').endswith('make_h3_block_patch.<locals>.block_patch') or
                inspect.getclosurevars(callback).nonlocals.get('patch') is not patch):
            raise ValueError('Native temporal alignment conflicts with another attention/block replacement patch.')
    # A per-evaluation copy bypasses only this node's producer hooks. That native
    # producer has no temporal Q/K bias; leaving it installed would bypass BFS.
    return dict(options, patches_replace={**options['patches_replace'], 'dit': {}})


def selection(options):
    """Recover sparse state even when a later dense node replaced its override."""
    dense, patch = unwrap_sol(options.get('optimized_attention_override'))
    if patch is None:
        for callback in options.get('patches_replace', {}).get('dit', {}).values():
            if (official_sparse_module(callback) and
                    getattr(callback, '__qualname__', '').endswith('make_h3_block_patch.<locals>.block_patch')):
                candidate = inspect.getclosurevars(callback).nonlocals.get('patch')
                if candidate is not None and type(candidate).__name__ == 'SparseAttnPatch' and official_sparse_module(type(candidate)):
                    if candidate.vsa or candidate.topk_ratio != 0:
                        raise ValueError('Native temporal alignment supports only sol-attn; SLA and VSA remain incompatible.')
                    patch = candidate
                    break
    without_owned_blocks(options, patch)
    return dense, patch
