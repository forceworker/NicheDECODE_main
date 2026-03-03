import numpy as np
import torch
import torch.nn.functional as F
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch import nn
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import anndata as ad


def spatial_sliding_window_stats(
        adata,
        window_width: float,
        step: float = None,
        overlap_rate: float = 0.0
):
    coords = adata.obsm["spatial"]
    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    if step is None:
        step = window_width * (1 - overlap_rate)

    # 创建窗口 x 范围
    windows = []
    x_start = x_min
    while x_start + window_width <= x_max:
        x_end = x_start + window_width
        windows.append((x_start, x_end))
        x_start += step

    cells_per_window = []
    for x_start, x_end in windows:
        mask = (x >= x_start) & (x <= x_end) & (y >= y_min) & (y <= y_max)
        count = np.sum(mask)
        cells_per_window.append(count)

    mean_cells = np.mean(cells_per_window)
    n_windows = len(windows)

    result = {
        "n_windows": n_windows,
        "mean_cells_per_window": mean_cells,
        "cells_per_window": cells_per_window,
        "window_ranges": windows,
    }

    return result


def set_global_seed(seed=2021):
    """设置所有相关的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 重要：确保CuDNN的行为也是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置环境变量（可选）
    os.environ['PYTHONHASHSEED'] = str(seed)


def predict(dataloader, type_list, model_use, if_pure=False):
    all_rate = []
    all_y = []
    with torch.no_grad():
        for batch in dataloader:
            x_sim = batch['x_sim']
            y = batch['y']
            if model_use.gpu_available:
                x_sim = x_sim.to(model_use.gpu)
                y = y.to(model_use.gpu)
            if if_pure:
                all_res, pred_rate = model_use.pure_forward(x_sim)
            else:
                extract_cell, noise, pred_rate = model_use.forward(x_sim)
            pred_rate = pred_rate.view(-1, len(type_list))
            all_rate.append(pred_rate)
            all_y.append(y)
    all_rate = torch.cat(all_rate, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_rate_cpu = all_rate.cpu()
    all_y_cpu = all_y.cpu()

    # 将 PyTorch 张量转换为 NumPy 数组
    all_rate_np = all_rate_cpu.numpy()
    all_y_np = all_y_cpu.numpy()

    # 创建 Pandas DataFrame
    df_rate = pd.DataFrame(all_rate_np, columns=type_list)
    df_y = pd.DataFrame(all_y_np, columns=type_list)
    CCC, RMSE, Corr = compute_metrics(df_rate, df_y)
    return CCC, RMSE, Corr


def data2h5ad(trainortest_data, y, type_list):
    df_list = [series.to_frame().T for series in trainortest_data]
    df = pd.concat(df_list, ignore_index=True)
    adata = ad.AnnData(df.values)
    y = np.array(y)
    for i, cell_type in enumerate(type_list):
        adata.obs[cell_type] = y[:, i].reshape(-1)
    adata.uns['cell_types'] = type_list
    print(adata)
    return adata


def ccc(preds, gt):
    numerator = 2 * np.corrcoef(gt, preds)[0][1] * np.std(gt) * np.std(preds)
    denominator = np.var(gt) + np.var(preds) + (np.mean(gt) - np.mean(preds)) ** 2
    ccc_value = numerator / denominator
    return ccc_value


def compute_metrics(preds, gt):
    gt = gt[preds.columns]  # Align pred order and gt order
    x = pd.melt(preds)['value']
    y = pd.melt(gt)['value']
    CCC = ccc(x, y)
    RMSE = sqrt(mean_squared_error(x, y))
    Corr = pearsonr(x, y)[0]
    return CCC, RMSE, Corr


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def loss2rate(pred_rate, real_rate):
    return F.cross_entropy(pred_rate, real_rate)


def L1_loss(preds, gt):
    loss = torch.mean(torch.reshape(torch.square(preds - gt), (-1,)))
    return loss


def calculate_mse(real_data, predicted_data):
    mse_loss = torch.nn.MSELoss().cuda()
    mse = mse_loss(predicted_data, real_data)
    return mse.item()


def metrics_list_calculation(real, pred, cell_type_list):
    real = np.array(real)
    pred = np.array(pred)
    rmse_list = []
    ccc_list = []
    r_list = []
    for i in range(len(cell_type_list)):
        rmse_list.append(sqrt(mean_squared_error(real[:, i], pred[:, i])))
        r = np.corrcoef(pred[:, i], real[:, i])[0, 1]
        # Mean
        mean_true = np.mean(real[:, i])
        mean_pred = np.mean(pred[:, i])
        # Variance
        var_true = np.var(real[:, i])
        var_pred = np.var(pred[:, i])
        # Standard deviation
        sd_true = np.std(real[:, i])
        sd_pred = np.std(pred[:, i])
        # Calculate CCC
        numerator = 2 * r * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        ccc_list.append(ccc)
        r_list.append(pearsonr(real[:, i], pred[:, i])[0])
    return rmse_list, ccc_list, r_list


def auto_log_then_minmax(
        df: pd.DataFrame,
        range_order_thresh: float = 3.0,  # order-of-magnitude threshold, e.g. >3 means max/min > 10^3
        skew_thresh: float = 1.0  # skewness threshold; large |skew| indicates a highly skewed distribution
):
    df_out = df.copy()

    # Only process numeric columns
    num_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
    log_transformed_cols = []

    for col in num_cols:
        s = df_out[col].dropna()
        if s.empty:
            continue

        # ------ 1. Compute order-of-magnitude range ------
        # Consider only positive values to avoid log(0) issues
        s_pos = s[s > 0]
        if len(s_pos) > 0:
            min_pos = s_pos.min()
            max_pos = s_pos.max()
            # Guard against min_pos==0 or numeric issues
            if min_pos > 0:
                range_order = np.log10(max_pos) - np.log10(min_pos)
            else:
                range_order = np.inf
        else:
            # If all values are non-positive, skip magnitude-based check
            range_order = 0

        # ------ 2. Compute skewness ------
        skew = s.skew()

        # ------ 3. Decide whether to apply log transform ------
        need_log = (range_order >= range_order_thresh) or (abs(skew) >= skew_thresh)

        if need_log:
            log_transformed_cols.append(col)

            # To safely use log1p, shift values so the minimum becomes >= 1
            col_min = df_out[col].min()
            if col_min <= 0:
                shift = 1 - col_min
            else:
                shift = 0.0

            df_out[col] = np.log1p(df_out[col] + shift)

    # ------ 4. Apply 0-1 scaling (min-max) to all numeric columns ------
    for col in num_cols:
        s = df_out[col]
        col_min = s.min()
        col_max = s.max()
        if col_max == col_min:
            # Constant column -> set to 0
            df_out[col] = 0.0
        else:
            df_out[col] = (s - col_min) / (col_max - col_min)

    return df_out, log_transformed_cols


def merge_full_array_to_series_list(series_list, full_array):
    for i in range(len(series_list)):
        series_list[i] = pd.concat([
            series_list[i],
            pd.Series(full_array)
        ], ignore_index=True)

    return series_list
