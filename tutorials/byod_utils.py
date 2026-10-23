"""Input preparation for the bring-your-own-data notebook (raw RNA counts)."""
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

SPATIAL_FEATURES = ['FSV', 'max_delta', 'max_ll', 'max_mu_hat',
                    'max_s2_t_hat', 's2_FSV', 's2_logdelta', 'LLR', 'pval', 'qval']


def dense(x):
    return np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float32)


def validate_counts(adata, spatial=False, niche_key='niche'):
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError('Observation and gene IDs must be unique; resolve duplicates upstream.')
    if not adata.n_obs or not adata.n_vars:
        raise ValueError('Empty expression matrix.')
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected finite nonnegative raw counts, not scaled expression.')
    if not np.allclose(values, np.round(values), atol=1e-4):
        raise ValueError('This template expects raw counts; do not supply log-expression or TPM.')
    if (np.asarray(adata.X.sum(axis=1)).ravel() <= 0).any():
        raise ValueError('Remove zero-library observations before running the template.')
    if spatial:
        coords = np.asarray(adata.obsm.get('spatial'))
        if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
            raise ValueError("obsm['spatial'] must be a finite n_observations x 2 array.")
        if niche_key not in adata.obs or adata.obs[niche_key].isna().any():
            raise ValueError('Each reference observation requires a niche annotation.')
        if adata.obs[niche_key].astype(str).str.strip().eq('').any():
            raise ValueError('Empty niche labels are not allowed.')


def attach_cell_counts(adata, counts):
    """Join a spot-indexed Series, never by row order or by filling missing IDs."""
    if not counts.index.is_unique:
        raise ValueError('Cell-count table has duplicate spot IDs.')
    aligned = pd.to_numeric(counts.reindex(adata.obs_names), errors='raise')
    if aligned.isna().any() or not np.isfinite(aligned).all() or (aligned <= 0).any():
        raise ValueError('Every retained spot needs a finite positive cell count. Resolve missing/zero spots explicitly.')
    adata.obs['cell_count'] = aligned.to_numpy()


def make_pseudotissues(ref, niche_key='niche', width=None, overlap=0.5,
                       angles=(0, 45, 90), min_observations=10, min_niches=1,
                       random_samples=1000, random_observations=None,
                       cell_count_key=None, seed=2021):
    """Aggregate raw counts; weight only labels by estimated cells per spot.

    Each reference must contain ONE section. Random draws use replacement.
    The normalized-uniform draw controls observation sampling; spot-weighted
    labels are recomputed from the observations actually drawn.
    """
    validate_counts(ref, spatial=True, niche_key=niche_key)
    if width is None or not np.isfinite(width) or width <= 0 or not 0 <= overlap < 1:
        raise ValueError('Set width > 0 and 0 <= overlap < 1.')
    if min_observations < 1 or min_niches < 1 or random_samples < 0:
        raise ValueError('Invalid sampling parameters.')
    labels = sorted(ref.obs[niche_key].astype(str).unique())
    if min_niches > len(labels):
        raise ValueError('min_niches exceeds the number of annotated niches.')
    codes = pd.Categorical(ref.obs[niche_key].astype(str), categories=labels).codes
    weights = np.ones(ref.n_obs) if cell_count_key is None else ref.obs[cell_count_key].to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Cell counts must be finite and positive.')
    coords = np.asarray(ref.obsm['spatial'], dtype=float)
    profiles, fractions, sizes, kinds = [], [], [], []

    def append(indices, kind):
        y = np.bincount(codes[indices], weights=weights[indices], minlength=len(labels))
        if np.count_nonzero(y) < min_niches and kind == 'window':
            return
        profiles.append(np.asarray(ref.X[indices].sum(axis=0)).ravel())
        fractions.append(y / y.sum())
        sizes.append(len(indices))
        kinds.append(kind)

    for angle in angles:
        theta = np.deg2rad(angle)
        x = coords @ np.array([np.cos(theta), -np.sin(theta)])
        stop = x.max() - width
        for start in np.arange(x.min(), stop + width * 1e-9, width * (1 - overlap)):
            indices = np.flatnonzero((x >= start) & (x <= start + width))
            if len(indices) >= min_observations:
                append(indices, 'window')
    if not profiles:
        raise ValueError('No usable windows. Adjust width/min_observations/min_niches; inspect coordinates.')
    n = int(round(np.mean(sizes))) if random_observations is None else int(random_observations)
    if n < 1:
        raise ValueError('random_observations must be positive.')
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(codes == i) for i in range(len(labels))]
    for _ in range(random_samples):
        proportions = rng.random(len(labels))
        numbers = rng.multinomial(n, proportions / proportions.sum())
        indices = np.concatenate([rng.choice(pool, size=count, replace=True)
                                  for pool, count in zip(pools, numbers)])
        append(indices, 'random')
    obs = pd.DataFrame(fractions, columns=labels,
                       index=[f'pseudo_{i}' for i in range(len(profiles))])
    result = ad.AnnData(np.asarray(profiles, dtype=np.float32), obs=obs, var=ref.var.copy())
    result.uns['cell_types'] = labels
    # Metadata goes in obsm to avoid collisions with user-defined niche names.
    result.obsm['sampling'] = pd.DataFrame({'kind': kinds, 'n_observations': sizes}, index=result.obs_names)
    return result


def normalize_profiles(x):
    x = dense(x)
    total = x.sum(axis=1, keepdims=True)
    if not np.isfinite(x).all() or (x < 0).any() or (total <= 0).any():
        raise ValueError('Invalid/empty expression profiles after gene alignment.')
    x = np.log1p(x / total * 1e4)
    return (x / x.max(axis=1, keepdims=True)).astype(np.float32)


def pack_model_input(adata, genes, spatial_prior, graph_genes):
    """Match the existing encoder layout: expression, then flattened priors."""
    if not set(genes).issubset(adata.var_names):
        raise ValueError('Required model genes are missing; do not silently zero-fill.')
    if spatial_prior.shape != (len(genes), len(SPATIAL_FEATURES)):
        raise ValueError('Spatial prior has the wrong dimensions.')
    expression = normalize_profiles(adata[:, genes].X)
    features = np.concatenate([expression, np.broadcast_to(
        spatial_prior.ravel(), (adata.n_obs, spatial_prior.size))], axis=1)
    packed = ad.AnnData(features.astype(np.float32), obs=adata.obs.copy())
    packed.uns['cell_types'] = list(adata.uns.get('cell_types', []))
    packed.obsm['new_gene_expression'] = expression[:, [genes.index(g) for g in graph_genes]]
    return packed


def functional_graph(embeddings, genes, k=5):
    """Return a bidirectional cosine kNN graph in explicit graph-gene order."""
    import torch
    graph_genes = [g for g in genes if g in embeddings.index]
    if not embeddings.index.is_unique or len(graph_genes) <= k or k < 1:
        raise ValueError('Need unique embedding IDs and more than k overlapping graph genes.')
    values = embeddings.loc[graph_genes].to_numpy(float)
    if not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any():
        raise ValueError('Embeddings must be finite and have nonzero norms.')
    distances, neighbors = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(values).kneighbors(values)
    edges = {}
    for i, (ds, js) in enumerate(zip(distances, neighbors)):
        for d, j in [(d, j) for d, j in zip(ds, js) if j != i][:k]:
            weight = float(1 - d)
            if weight < -1e-6:
                raise ValueError('Negative cosine edge weight; inspect the supplied GO embeddings.')
            edges[i, int(j)] = edges[int(j), i] = max(0., weight)
    return (graph_genes, torch.tensor(list(edges), dtype=torch.long).T.contiguous(),
            torch.tensor(list(edges.values()), dtype=torch.float32))


def predict_unlabelled(model, packed, batch_size=32):
    """Predict real bulk samples without fabricated ground-truth columns."""
    import torch
    model.model_da.eval()
    device = next(model.encoder.parameters()).device
    predictions = []
    with torch.no_grad():
        for start in range(0, packed.n_obs, batch_size):
            x = torch.as_tensor(packed.X[start:start + batch_size], device=device)
            g = torch.as_tensor(packed.obsm['new_gene_expression'][start:start + batch_size], device=device)
            result = model.predictor(model.encoder(x) * torch.sigmoid(model.gnn(g)))
            predictions.append(result.cpu().numpy())
    return pd.DataFrame(np.concatenate(predictions), index=packed.obs_names, columns=model.labels)
