"""Exercise both notebook branches on synthetic data, never biological accuracy.

Run in the NicheDECODE environment. CUDA is required for the training cells.
All generated data and weights are kept in a temporary directory.
"""
import json
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import torch
from tutorials.byod_utils import attach_cell_counts, make_pseudotissues, SPATIAL_FEATURES


def main():
    notebook = json.loads((ROOT / 'tutorials/bring_your_own_data.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
    # Analytically known weighted composition and unweighted count aggregation.
    tiny = ad.AnnData(sparse.csr_matrix([[10, 1], [2, 20], [1, 1]], dtype=np.float32),
        obs=pd.DataFrame({'niche': ['A', 'B', 'A']}, index=['s1', 's2', 's3']))
    tiny.obsm['spatial'] = np.array([[0, 0], [1, 0], [3, 1]], dtype=float)
    attach_cell_counts(tiny, pd.Series([1., 2., 8.], index=['s3', 's2', 's1']))
    pseudo = make_pseudotissues(tiny, width=1.5, overlap=0, angles=[0],
        min_observations=2, random_samples=0, cell_count_key='cell_count')
    np.testing.assert_allclose(pseudo.X[0], [12, 21])
    np.testing.assert_allclose(pseudo.obs.iloc[0].to_numpy(float), [0.8, 0.2])
    try:
        attach_cell_counts(tiny, pd.Series([1], index=['wrong_id']))
    except ValueError:
        pass
    else:
        raise AssertionError('Missing spot IDs were not rejected.')
    print('Weighted composition, count aggregation, ID alignment and cell syntax passed.', flush=True)
    if not torch.cuda.is_available():
        print('SKIPPED notebook training: CUDA unavailable.', flush=True)
        return
    for spot in [False, True]:
        with tempfile.TemporaryDirectory(prefix='nichedecode_byod_') as directory:
            folder = Path(directory)
            rng = np.random.default_rng(42)
            genes = [f'g{i}' for i in range(8)]
            ref = ad.AnnData(sparse.csr_matrix(rng.poisson(10, (80, 8)).astype(np.float32)),
                obs=pd.DataFrame({'niche': ['A', 'B'] * 40}, index=[f's{i}' for i in range(80)]),
                var=pd.DataFrame(index=genes))
            ref.layers['counts'] = ref.X.copy()
            ref.obsm['spatial'] = np.column_stack([np.arange(80), np.arange(80) % 9]).astype(float)
            ref.write_h5ad(folder / 'reference.h5ad')
            # Deliberately reversed gene order to test alignment.
            pd.DataFrame(rng.poisson(100, (8, 3)), index=genes[::-1],
                         columns=['patient_C', 'patient_A', 'patient_B']).to_csv(folder / 'bulk_counts.csv')
            prior = pd.DataFrame(rng.random((8, 10)), index=genes[::-1], columns=SPATIAL_FEATURES)
            prior.rename_axis('g').to_csv(folder / 'spatialde.csv')
            pd.DataFrame(rng.random((8, 6)), index=genes[::-1]).to_csv(folder / 'go_embeddings.csv')
            pd.Series(rng.integers(1, 10, 80), index=ref.obs_names,
                      name='cell_count').rename_axis('spot_id').to_csv(folder / 'spot_cell_counts.csv')
            namespace = {}
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if source.startswith('INPUT ='):
                    # Replace the configuration cell with an equivalent temporary setup.
                    source = source.replace("ROOT / 'data/custom'", 'TEST_FOLDER')
                    source = source.replace("'custom_run_001'", repr('smoke_' + uuid.uuid4().hex))
                    source = source.replace("ROOT / 'res' / RUN_NAME", "TEST_FOLDER / 'output'")
                    source = source.replace("ROOT / 'save_models' / RUN_NAME / 'best_model.pt'",
                                            "TEST_FOLDER / 'weights' / 'best_model.pt'")
                    namespace['TEST_FOLDER'] = folder
                exec(compile(source, '<notebook>', 'exec'), namespace)
                if source.startswith('INPUT ='):
                    namespace.update(SPOT_RESOLUTION=spot, RANDOM_SAMPLES=12,
                        MIN_OBSERVATIONS=3, WIDTH_FRACTION=0.25, EPOCHS=2, BATCH_SIZE=8)
            pred = namespace['predictions']
            assert list(pred.index) == ['patient_C', 'patient_A', 'patient_B']
            assert list(pred.columns) == ['A', 'B']
            assert (pred.to_numpy() >= 0).all()
            manifest = json.loads((folder / 'output/manifest.json').read_text())
            assert manifest['genes'] == genes
            assert manifest['spot_resolution'] == spot
            print(f'Notebook completed: spot_resolution={spot}', flush=True)


if __name__ == '__main__':
    main()
