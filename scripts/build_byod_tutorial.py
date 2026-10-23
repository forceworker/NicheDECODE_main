"""Build the checked-in tutorial notebook."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip().splitlines(True)))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                      source=text.strip().splitlines(True)))


md('''# Bring your own data to NicheDECODE
Train a new model using an annotated spatial reference and predict niche proportions
for unlabelled bulk RNA-seq samples. Start with the [input and parameter guide](../docs/bring-your-own-data.html).
This template requires raw RNA counts, a reference-derived SpatialDE table and GO embeddings.
It supports single-cell spatial references and cell-number-weighted spot references.
Outputs describe relative composition of the supplied niches, not cell coordinates.
Run from this repository using the NicheDECODE environment and a CUDA GPU.
External tools are run separately; their installation is not included in environment.yml.''')
code('''from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'model/nicheDeconv.py').exists())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import anndata as ad
import torch
from sklearn.model_selection import train_test_split
from model.nicheDeconv import nicheDeconv
from model.utils import auto_log_then_minmax, set_global_seed
from tutorials.byod_utils import (
    SPATIAL_FEATURES, validate_counts, attach_cell_counts, make_pseudotissues,
    pack_model_input, functional_graph, predict_unlabelled, dense,
)
set_global_seed(2021)''')
md('''## 1 Set paths and sampling parameters
Replace the paths below. Use a new RUN_NAME for each configuration: the existing
training API automatically loads save_models/RUN_NAME/best_model.pt if it exists.
The guard below prevents accidental reuse. One reference file must contain one section.
WIDTH_FRACTION is relative to its x-coordinate span; inspect the resulting windows.
These are starting settings, not optimized settings for every tissue.''')
code('''INPUT = ROOT / 'data/custom'
REFERENCE = INPUT / 'reference.h5ad'
BULK = INPUT / 'bulk_counts.csv'  # gene rows, sample columns, first column = gene ID
SPATIAL = INPUT / 'spatialde.csv'  # column g + the ten SPATIAL_FEATURES
EMBEDDINGS = INPUT / 'go_embeddings.csv'  # header; first column = matching gene ID
NICHE_KEY = 'niche'
COUNTS_LAYER = 'counts'  # set None only when reference.X holds raw counts
SPOT_RESOLUTION = False
CELL_COUNTS = INPUT / 'spot_cell_counts.csv'  # spot_id,cell_count; required for spots
RUN_NAME = 'custom_run_001'
WIDTH_FRACTION = 0.10
OVERLAP = 0.5
ANGLES = [0, 45, 90]
MIN_OBSERVATIONS = 10
MIN_NICHES = 1
RANDOM_SAMPLES = 1000
RANDOM_OBSERVATIONS = None  # None: mean retained window observation count
GRAPH_K = 5
EPOCHS, BATCH_SIZE, LEARNING_RATE, PATIENCE = 200, 32, 1e-4, 10
OUT = ROOT / 'res' / RUN_NAME
CHECKPOINT = ROOT / 'save_models' / RUN_NAME / 'best_model.pt'
if CHECKPOINT.exists() or OUT.exists():
    raise FileExistsError('Choose a new RUN_NAME to protect an existing run.')''')
md('''## 2 Load the reference and attach niche annotations
reference.obs_names are cell/spot IDs; reference.var_names are gene IDs;
reference.obsm['spatial'] contains x,y coordinates. The string/categorical column
reference.obs[NICHE_KEY] assigns one niche to each observation.
If annotations are in a separate CSV, join by ID as illustrated below. NicheDECODE
does not discover niches: obtain spatial domains using a suitable spatial annotation
method or expert histology, inspect their spatial distribution and markers, and
use consistent biological labels. Cell-type labels alone are not niche annotations.''')
code('''reference = ad.read_h5ad(REFERENCE)
if COUNTS_LAYER is not None:
    reference.X = reference.layers[COUNTS_LAYER].copy()
# Optional import; uncomment if labels are not already present:
# annotations = pd.read_csv(INPUT / 'niche_annotations.csv', index_col=0)
# assert annotations.index.is_unique
# reference.obs[NICHE_KEY] = annotations['niche'].reindex(reference.obs_names)
validate_counts(reference, spatial=True, niche_key=NICHE_KEY)
print(reference)
print(reference.obs[NICHE_KEY].value_counts())
# If the file contains multiple sections, subset ONE before proceeding:
# reference = reference[reference.obs['section'] == 'section_A'].copy()''')
md('''## 3 Spot-resolution branch with CytoSPACE
For single-cell spatial data, skip the external step and leave SPOT_RESOLUTION=False.
For spots, run [CytoSPACE](https://github.com/digitalcytometry/cytospace) with a compatible,
cell-type-annotated scRNA reference and the spatial raw counts/coordinates (see guide).
Convert assigned_locations.csv to the two-column file expected below:

```python
assignments = pd.read_csv('cytospace_output/assigned_locations.csv')
assert assignments['UniqueCID'].is_unique
counts = assignments.groupby('SpotID').size().rename('cell_count')
counts.rename_axis('spot_id').to_csv(INPUT / 'spot_cell_counts.csv')
```

Join using exact spot IDs, including any section prefixes. Missing or zero-assignment
spots require explicit QC; they are not filled with one cell. The helper sums observed
spot expression once per selected spot, and uses estimated cell counts only for labels:
`p(niche c) = sum(cell_count of selected spots labelled c) / sum(cell_count of all selected spots)`.
Do not multiply spot expression by cell counts or substitute CytoSPACE cell-type fractions
for niche labels.''')
code('''if SPOT_RESOLUTION:
    cell_counts = pd.read_csv(CELL_COUNTS, index_col=0)['cell_count']
    attach_cell_counts(reference, cell_counts)
    print(reference.obs['cell_count'].describe())''')
md('''## 4 Prepare unlabelled bulk targets and align genes
CSV orientation is genes x samples; AnnData orientation is samples x genes. No niche
labels or spatial coordinates are needed for bulk samples. Use unique IDs in the same
species and identifier system as the reference and embeddings. Resolve duplicate or
ambiguous mappings upstream; do not append suffixes to biological gene IDs.
For bulk .h5ad, replace the CSV conversion with ad.read_h5ad and set X to raw counts.
Gene intersection uses availability only, not target outcomes. For a large reference,
select informative genes using the reference alone before this step and record the list.''')
code('''bulk_table = pd.read_csv(BULK, index_col=0)
bulk = ad.AnnData(bulk_table.T.astype(np.float32))
validate_counts(bulk)
spatial_table = pd.read_csv(SPATIAL).set_index('g')
if not spatial_table.index.is_unique:
    raise ValueError('SpatialDE must contain one row per gene.')
genes = [g for g in reference.var_names if g in bulk.var_names and g in spatial_table.index]
if len(genes) <= GRAPH_K:
    raise ValueError('Too few shared genes; check identifiers and matrix orientation.')
print(f'Shared genes: {len(genes)}/{reference.n_vars} reference genes; {bulk.n_vars} bulk genes')
reference = reference[:, genes].copy()
bulk = bulk[:, genes].copy()
validate_counts(reference, spatial=True, niche_key=NICHE_KEY)
validate_counts(bulk)
prior_raw = spatial_table.loc[genes, SPATIAL_FEATURES].astype(float)
if not np.isfinite(prior_raw.to_numpy()).all():
    raise ValueError('Resolve failed/nonfinite SpatialDE fits before training.')
prior_scaled, log_columns = auto_log_then_minmax(prior_raw)
prior = prior_scaled.to_numpy(dtype=np.float32)
embeddings = pd.read_csv(EMBEDDINGS, index_col=0)
embeddings.index = embeddings.index.astype(str)
graph_genes, edge_index, edge_weight = functional_graph(embeddings, genes, k=GRAPH_K)
print(f'Graph coverage: {len(graph_genes)}/{len(genes)} genes')''')
md('''### Preparing the spatial features and functional embeddings
Run SpatialDE on the reference only, using its raw counts and coordinates. See the
guide for executable preprocessing code and the exact ten output columns. Do not
reuse the NSCLC spatial prior for another tissue. GO embeddings may be reused when
species and gene identifiers agree. The guide includes the FRoGS source and a CSV
conversion example; GO-term membership tables are not embedding matrices.''')
md('''## 5 Construct pseudo-tissues
Use this template to train a new model for your reference and target cohort.
Random mixing samples observations with replacement and recomputes labels from actual
draws. For spots, requested sampling fractions need not equal cell-weighted labels.
No niche is automatically removed. Review rare categories before choosing the output
label set; the model cannot estimate an unrepresented niche.''')
code('''width = np.ptp(reference.obsm['spatial'][:, 0]) * WIDTH_FRACTION
pseudo = make_pseudotissues(
    reference, niche_key=NICHE_KEY, width=width, overlap=OVERLAP, angles=ANGLES,
    min_observations=MIN_OBSERVATIONS, min_niches=MIN_NICHES,
    random_samples=RANDOM_SAMPLES, random_observations=RANDOM_OBSERVATIONS,
    cell_count_key='cell_count' if SPOT_RESOLUTION else None,
)
print(pseudo)
print(pseudo.obsm['sampling'].groupby('kind')['n_observations'].describe())
print(pseudo.obs.describe())
if pseudo.n_obs < 10:
    raise ValueError('Too few pseudo-tissues; adjust sampling before training.')''')
md('''## 6 Train using reference-derived validation samples
The random split below is for early stopping only. Overlapping windows and random
mixtures share reference observations, so this is not independent accuracy evidence.
For a benchmark, hold out complete donors/sections before constructing pseudo-tissues
and fitting spatial priors. Do not tune on clinical outcomes of the bulk cohort.
The legacy training API requests a labelled target_data argument, but training uses
source_data and early stopping uses valid_data. Here valid_data fills that interface
slot; real bulk samples remain unlabelled and are predicted separately.''')
code('''if not torch.cuda.is_available():
    raise RuntimeError('The current NicheDECODE model requires CUDA; select a CUDA-enabled environment.')
train_ids, valid_ids = train_test_split(np.arange(pseudo.n_obs), test_size=0.2, random_state=2021)
source_data = pack_model_input(pseudo[train_ids].copy(), genes, prior, graph_genes)
valid_data = pack_model_input(pseudo[valid_ids].copy(), genes, prior, graph_genes)
target_data = pack_model_input(bulk, genes, prior, graph_genes)
model = nicheDeconv(EPOCHS, BATCH_SIZE, LEARNING_RATE)
model.register_graph(edge_index, edge_weight, len(graph_genes))
loss, best_weights = model.train(source_data, valid_data, valid_data, PATIENCE, RUN_NAME)
if best_weights is None:
    raise RuntimeError('Training produced no valid checkpoint; inspect loss and inputs.')
for name in ['encoder', 'gnn', 'predictor']:
    getattr(model, name).load_state_dict(best_weights[name])''')
md('''## 7 Predict and save the model contract
The output CSV has sample IDs as rows and niche labels as columns. Values are
nonnegative and sum to one per sample. Accuracy cannot be computed without measured
ground truth. The frozen gene order, prior, graph and label order must accompany the
weights for reuse; changing the panel or reference requires a new run.''')
code('''predictions = predict_unlabelled(model, target_data, batch_size=BATCH_SIZE)
assert np.isfinite(predictions.to_numpy()).all()
assert np.allclose(predictions.sum(axis=1), 1, atol=1e-5)
OUT.mkdir(parents=True)
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUT / 'niche_proportions.csv', index_label='sample_id')
torch.save(best_weights, CHECKPOINT)
np.savez_compressed(OUT / 'model_inputs.npz', spatial_prior=prior,
                    edge_index=edge_index.numpy(), edge_weight=edge_weight.numpy())
manifest = dict(genes=genes, graph_genes=graph_genes, niches=list(model.labels),
    spatial_features=SPATIAL_FEATURES, reference=str(REFERENCE), bulk=str(BULK),
    spatial_table=str(SPATIAL), embeddings=str(EMBEDDINGS),
    normalization='raw counts -> total 10000 -> log1p -> row maximum',
    spot_resolution=SPOT_RESOLUTION, cell_counts=str(CELL_COUNTS) if SPOT_RESOLUTION else None,
    width=float(width), overlap=OVERLAP, angles=ANGLES, min_observations=MIN_OBSERVATIONS,
    min_niches=MIN_NICHES, random_samples=RANDOM_SAMPLES, random_observations=RANDOM_OBSERVATIONS,
    graph_k=GRAPH_K, seed=2021, epochs=EPOCHS, batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, patience=PATIENCE)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(predictions.head())
print('Saved predictions and preprocessing contract to', OUT)''')

notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.9'}}, nbformat=4, nbformat_minor=5)
for i, cell in enumerate(cells):
    cell['id'] = f'byod-{i:02d}'
(ROOT / 'tutorials/bring_your_own_data.ipynb').write_text(json.dumps(notebook, indent=1), encoding='utf-8')
