import os

os.environ["PYTHONHASHSEED"] = "2021"

import warnings

warnings.filterwarnings("ignore")

import copy
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.utils.data as Data

from torch_geometric.nn import GCNConv
from .utils import *


class EncoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(EncoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_dim, out_dim, do_rates):
        super(DecoderBlock, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, out_dim),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Dropout(p=do_rates, inplace=False))

    def forward(self, x):
        out = self.layer(x)
        return out


class GraphFeatureGNN(nn.Module):

    def __init__(self, num_nodes, edge_index, edge_weight, out_dim):
        super().__init__()
        self.num_nodes = num_nodes

        # edge_index: [2, E]，edge_weight: [E]
        self.register_buffer("edge_index", edge_index)  # long
        self.register_buffer("edge_weight", edge_weight)  # float

        hidden_dim = int(out_dim / 2)
        self.conv1 = GCNConv(in_channels=1, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=out_dim)

        self._cached_B = None
        self._cached_edge_index = None
        self._cached_edge_weight = None

    def _get_batched_graph(self, B, device):
        if self._cached_B == B and self._cached_edge_index is not None:
            return self._cached_edge_index, self._cached_edge_weight

        N = self.num_nodes
        E = self.edge_index.size(1)

        # offsets: [B, 1]
        offsets = torch.arange(B, device=device).unsqueeze(1) * N

        ei = self.edge_index.to(device)
        batched_edge_index = ei.unsqueeze(0) + offsets.view(B, 1, 1)  # [B, 2, E]

        batched_edge_index = batched_edge_index.permute(1, 0, 2).reshape(2, B * E)

        ew = self.edge_weight.to(device)
        batched_edge_weight = ew.repeat(B)  # [B*E]

        self._cached_B = B
        self._cached_edge_index = batched_edge_index
        self._cached_edge_weight = batched_edge_weight

        return batched_edge_index, batched_edge_weight

    def forward(self, x):
        B, N = x.shape
        assert N == self.num_nodes, f"num_nodes mismatch: {N} != {self.num_nodes}"

        device = x.device

        batched_edge_index, batched_edge_weight = self._get_batched_graph(B, device)

        # node_feat: [B, N] -> [B*N, 1]
        node_feat = x.reshape(B * N, 1)

        h = self.conv1(node_feat, batched_edge_index, batched_edge_weight)
        h = F.relu(h)
        h = self.conv2(h, batched_edge_index, batched_edge_weight)
        h = F.relu(h)  # [B*N, out_dim]

        h = h.view(B, N, -1)
        g = h.mean(dim=1)  # [B, out_dim]

        return g


class nicheDeconv(object):
    def __init__(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.celltype_num = None
        self.labels = None
        self.used_features = None
        self.seed = 2021

        self.edge_index = None
        self.edge_weight = None

        cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def register_graph(self, edge_index, edge_weight, node_num):

        self.edge_index = edge_index.cuda()
        self.edge_weight = edge_weight.cuda()
        self.node_num = node_num

    def nicheDeconv_model(self, celltype_num):
        feature_num = len(self.used_features)

        self.encoder = nn.Sequential(
            EncoderBlock(feature_num, 512, 0),
            EncoderBlock(512, 256, 0.3)
        )

        assert self.edge_index is not None and self.edge_weight is not None, \
            "请先调用 register_graph(edge_index, edge_weight) 再构建模型"
        self.gnn = GraphFeatureGNN(
            num_nodes=self.node_num,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            out_dim=256
        )

        self.predictor = nn.Sequential(
            EncoderBlock(256, 128, 0.2),
            nn.Linear(128, celltype_num),
            nn.Softmax(dim=1)
        )

        model = nn.ModuleList([])
        model.append(self.encoder)
        model.append(self.gnn)
        model.append(self.predictor)
        return model

    def prepare_dataloader(self, source_data, target_data, valid_data, batch_size):

        # ====== Source dataset ======
        g = torch.Generator()
        g.manual_seed(self.seed)
        source_ratios = [source_data.obs[ctype] for ctype in source_data.uns['cell_types']]
        self.source_data_x1 = source_data.X.astype(np.float32)

        source_new = source_data.obsm['new_gene_expression']
        if not isinstance(source_new, np.ndarray):
            source_new = np.asarray(source_new.todense() if hasattr(source_new, "todense") else source_new)
        self.source_data_x2 = source_new.astype(np.float32)

        self.source_data_y = np.array(source_ratios, dtype=np.float32).transpose()

        tr_x1 = torch.FloatTensor(self.source_data_x1)
        tr_x2 = torch.FloatTensor(self.source_data_x2)
        tr_labels = torch.FloatTensor(self.source_data_y)

        source_dataset = Data.TensorDataset(tr_x1, tr_x2, tr_labels)
        self.train_source_loader = Data.DataLoader(
            dataset=source_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # ====== Extract celltype and feature info ======
        self.labels = source_data.uns['cell_types']
        self.celltype_num = len(self.labels)
        self.used_features = list(source_data.var_names)

        # ====== Target dataset ======
        self.target_data_x1 = target_data.X.astype(np.float32)
        target_new = target_data.obsm['new_gene_expression']
        if not isinstance(target_new, np.ndarray):
            target_new = np.asarray(target_new.todense() if hasattr(target_new, "todense") else target_new)
        self.target_data_x2 = target_new.astype(np.float32)

        target_ratios = [target_data.obs[ctype] for ctype in self.labels]
        self.target_data_y = np.array(target_ratios, dtype=np.float32).transpose()

        te_x1 = torch.FloatTensor(self.target_data_x1)
        te_x2 = torch.FloatTensor(self.target_data_x2)
        te_labels = torch.FloatTensor(self.target_data_y)

        target_dataset = Data.TensorDataset(te_x1, te_x2, te_labels)
        self.train_target_loader = Data.DataLoader(
            dataset=target_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_target_loader = Data.DataLoader(
            dataset=target_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # ====== Valid dataset ======
        self.valid_data_x1 = valid_data.X.astype(np.float32)
        valid_new = valid_data.obsm['new_gene_expression']
        if not isinstance(valid_new, np.ndarray):
            valid_new = np.asarray(valid_new.todense() if hasattr(valid_new, "todense") else valid_new)
        self.valid_data_x2 = valid_new.astype(np.float32)

        valid_ratios = [valid_data.obs[ctype] for ctype in self.labels]
        self.valid_data_y = np.array(valid_ratios, dtype=np.float32).transpose()

        va_x1 = torch.FloatTensor(self.valid_data_x1)
        va_x2 = torch.FloatTensor(self.valid_data_x2)
        va_labels = torch.FloatTensor(self.valid_data_y)

        valid_dataset = Data.TensorDataset(va_x1, va_x2, va_labels)
        self.valid_target_loader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def train(self, source_data, target_data, valid_data, patience, tissue_name):
        counter = 0
        best_model_weights = None
        best_rmse = float('inf')
        ### prepare model_scaden structure ###
        self.prepare_dataloader(source_data, target_data, valid_data, self.batch_size)
        self.model_da = self.nicheDeconv_model(self.celltype_num).cuda()

        ### setup optimizer ###
        optimizer_da1 = torch.optim.Adam([{'params': self.encoder.parameters()},
                                          {'params': self.gnn.parameters()},
                                          {'params': self.predictor.parameters()}], lr=self.learning_rate)

        loss1 = []
        ckpt_path = Path(__file__).resolve().parent.parent / "save_models" / tissue_name / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path)
            return loss1, ckpt
        for epoch in range(self.num_epochs):
            self.model_da.train()

            pred_loss_epoch = 0.
            for batch_idx, (source_x1, source_x2, source_y) in enumerate(self.train_source_loader):
                embedding_source = self.encoder(source_x1.cuda())
                gnn_vec = self.gnn(source_x2.cuda())  # [B, 256]
                gating = torch.sigmoid(gnn_vec)
                gated_embedding = embedding_source * gating
                frac_pred = self.predictor(gated_embedding)

                # caculate loss 
                pred_loss = L1_loss(frac_pred, source_y.cuda())
                pred_loss_epoch += pred_loss.data.item()
                loss = pred_loss

                # update weights
                optimizer_da1.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_da1.step()

            valid_rmse = self.evaluate(self.valid_target_loader)
            if valid_rmse < best_rmse:
                best_rmse = valid_rmse
                best_model_weights = {
                    'encoder': copy.deepcopy(self.encoder.state_dict()),
                    'gnn': copy.deepcopy(self.gnn.state_dict()),
                    'predictor': copy.deepcopy(self.predictor.state_dict())
                }
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break
            pred_loss_epoch = pred_loss_epoch / (batch_idx + 1)
            loss1.append(pred_loss_epoch)

            if (epoch + 1) % 10 == 0:
                print('============= Epoch {:02d}/{:02d} ============='.format(epoch + 1, self.num_epochs))
                print("pred_loss=%f" % (pred_loss_epoch))
        return loss1, best_model_weights

    def prediction(self, data_test):
        self.model_da.eval()
        preds, gt = None, None
        for batch_idx, (x1, x2, y) in enumerate(data_test):
            embedding = self.encoder(x1.cuda())

            gnn_vec = self.gnn(x2.cuda())
            gating = torch.sigmoid(gnn_vec)

            gated_embedding = embedding * gating

            logits = self.predictor(gated_embedding).detach().cpu().numpy()
            frac = y.detach().cpu().numpy()
            preds = logits if preds is None else np.concatenate((preds, logits), axis=0)
            gt = frac if gt is None else np.concatenate((gt, frac), axis=0)

        target_preds = pd.DataFrame(preds, columns=self.labels)
        ground_truth = pd.DataFrame(gt, columns=self.labels)
        return target_preds, ground_truth

    def evaluate(self, valid_data):
        final_preds_target, ground_truth_target = self.prediction(valid_data)
        _ = []
        for label in self.labels:
            rmse = np.sqrt(np.mean((final_preds_target[label] - ground_truth_target[label]) ** 2))
            _.append(rmse)

        avg_rmse = np.mean(_)
        return avg_rmse
