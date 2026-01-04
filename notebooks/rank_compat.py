import numpy as np
import torch
from torch import nn
from typing import Callable, Dict, List, Tuple
from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt

############################################
# core pieces (same as before)
############################################

def compute_CR_numpy(y_true, p_o, p_u) -> float:
    y_true = np.asarray(y_true).astype(int)
    p_o = np.asarray(p_o).astype(float)
    p_u = np.asarray(p_u).astype(float)

    idx0 = np.where(y_true == 0)[0]
    idx1 = np.where(y_true == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        return np.nan

    po0 = p_o[idx0][:, None]
    po1 = p_o[idx1][None, :]
    pu0 = p_u[idx0][:, None]
    pu1 = p_u[idx1][None, :]

    orig_ok = (po0 < po1)
    den = orig_ok.sum()
    if den == 0:
        return np.nan

    upd_ok = (pu0 < pu1)
    num = (orig_ok & upd_ok).sum()

    return float(num / den)


def rank_incompatibility_loss(y, p_o_batch, p_u_batch, slope: float = 20.0):
    y = y.long()
    I0 = torch.nonzero(y == 0, as_tuple=False).squeeze(-1)
    I1 = torch.nonzero(y == 1, as_tuple=False).squeeze(-1)

    if I0.numel() == 0 or I1.numel() == 0:
        return torch.zeros((), device=y.device, dtype=p_u_batch.dtype)

    p_o_i = p_o_batch[I0].unsqueeze(1)
    p_o_j = p_o_batch[I1].unsqueeze(0)
    p_u_i = p_u_batch[I0].unsqueeze(1)
    p_u_j = p_u_batch[I1].unsqueeze(0)

    d_o = p_o_j - p_o_i
    d_u = p_u_j - p_u_i

    sigma_o = torch.sigmoid(slope * d_o)
    sigma_u = torch.sigmoid(slope * d_u)

    num = (sigma_o * sigma_u).sum()
    den = sigma_o.sum() + 1e-12
    cr_tilde = num / den
    return 1.0 - cr_tilde


class CompatDataset(Dataset):
    def __init__(self, X, y, p_o):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y.astype(np.int64))
        self.p_o = torch.as_tensor(p_o, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.p_o[idx]


class CompatTrainer:
    """
    Trains a model with:
      total_loss = alpha * BCEWithLogitsLoss + (1 - alpha) * rank_incompatibility_loss

    alpha is the same alpha as in the paper.
    slope is s in the paper, controls how sharp the compatibility surrogate is.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 alpha: float, slope: float, device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.alpha = alpha
        self.slope = slope
        self.device = device
        self._bce = nn.BCEWithLogitsLoss()

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            p_o_train: np.ndarray,
            batch_size: int = 512,
            epochs: int = 8,
            shuffle: bool = True):
        ds = CompatDataset(X_train, y_train, p_o_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

        for _ in range(epochs):
            self.model.train()
            for xb, yb, p_ob in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                p_ob = p_ob.to(self.device)

                logits = self.model(xb).squeeze(-1)
                p_u = torch.sigmoid(logits)

                loss_bce = self._bce(logits, yb.float())
                loss_r = rank_incompatibility_loss(yb, p_ob, p_u, slope=self.slope)
                total_loss = self.alpha * loss_bce + (1.0 - self.alpha) * loss_r

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        logits = self.model(X_t).squeeze(-1)
        p = torch.sigmoid(logits)
        return p.cpu().numpy()

    @torch.no_grad()
    def evaluate_holdout(self, X_val, y_val, p_o_val) -> Dict[str, float]:
        p_u_val = self.predict_proba(X_val)
        auroc_val = roc_auc_score(y_val, p_u_val)
        cr_val = compute_CR_numpy(y_val, p_o_val, p_u_val)
        return {"AUROC": auroc_val, "CR": cr_val}


class LogisticHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x)


# ============================================
# Original models: BCE-only logistic regression
# ============================================

class OriginalLRTrainer:
    """
    Regularized logistic regression with BCE only.
    Uses your LogisticHead and Adam weight decay as L2.
    Optionally standardizes inputs based on the training set.
    """
    def __init__(self, d: int, lr: float = 1e-3, weight_decay: float = 1e-2,
                 device: str = "cpu"):
        self.model = LogisticHead(d).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.crit = nn.BCEWithLogitsLoss()
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            batch_size: int = 512, epochs: int = 8, shuffle: bool = True, seed: int = 0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32, device=self.device),
                           torch.as_tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=self.device))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, drop_last=False)
        self.model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                self.opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.crit(logits, yb)
                loss.backward()
                self.opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        logits = self.model(torch.as_tensor(X, dtype=torch.float32, device=self.device))
        return torch.sigmoid(logits).cpu().numpy().ravel()

    @torch.no_grad()
    def coef_l2(self) -> float:
        w = self.model.linear.weight.detach().cpu().numpy()
        return float(np.linalg.norm(w))


############################################
# train baseline BCE pool (alpha = 1.0) once
############################################

def train_bce_pool_once(
    X_train, y_train, p_o_train,
    X_val, y_val, p_o_val,
    *,
    weight_decay_list: List[float],
    make_model_fn: Callable[[], nn.Module],
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 8,
    device: str = "cpu",
    n_bootstrap: int = 45,
    n_shuffle: int = 5,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """
    Train baseline "BCE only" models, alpha = 1.0.

    For each weight decay:
      - n_bootstrap bootstrap-resampled models
      - n_shuffle shuffled-order models

    Returns a list of dicts
      {
        "family": "BCE",
        "alpha": 1.0,
        "s": None,
        "weight_decay": wd,
        "rep_type": "bootstrap" or "shuffle",
        "rep_id": k,
        "AUROC": ...,
        "CR": ...,
        "trainer": trainer,
      }
    """
    rng = np.random.RandomState(seed)
    n = X_train.shape[0]
    runs = []

    # helper to train a single model with alpha=1
    def train_single(idx_rows, wd, rep_type, rep_id):
        torch.manual_seed(hash((wd, rep_type, int(rep_id))) % (2**32))
        np.random.seed(hash((wd, rep_type, int(rep_id))) % (2**32))

        model = make_model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        trainer = CompatTrainer(
            model=model,
            optimizer=opt,
            alpha=1.0,      # pure BCE
            slope=1.0,      # slope irrelevant when alpha=1
            device=device,
        )

        trainer.fit(
            X_train[idx_rows],
            y_train[idx_rows],
            p_o_train[idx_rows],
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
        )
        metrics = trainer.evaluate_holdout(X_val, y_val, p_o_val)

        return {
            "family": "BCE",
            "alpha": 1.0,
            "s": None,
            "weight_decay": wd,
            "rep_type": rep_type,
            "rep_id": rep_id,
            "AUROC": metrics["AUROC"],
            "CR": metrics["CR"],
            "trainer": trainer,
        }

    for wd in weight_decay_list:
        # bootstrap reps
        for k in range(n_bootstrap):
            idx_bs = rng.choice(n, size=n, replace=True)
            runs.append(train_single(idx_bs, wd, "bootstrap", k))
        # shuffle reps
        for k in range(n_shuffle):
            idx_shuf = rng.permutation(n)
            runs.append(train_single(idx_shuf, wd, "shuffle", k))

    return runs


############################################
# train RBC grid for a single s
############################################

def train_rbc_grid_for_s(
    X_train, y_train, p_o_train,
    X_val, y_val, p_o_val,
    *,
    alpha_list: List[float],
    weight_decay_list: List[float],
    s_val: float,
    make_model_fn: Callable[[], nn.Module],
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 8,
    device: str = "cpu",
    seed: int = 0,
) -> List[Dict[str, float]]:
    """
    Train one RBC model per (alpha, weight_decay) for a fixed s.

    No bootstrapping here. Each combo is just trained once.

    Returns a list of dicts
      {
        "family": "RBC",
        "alpha": alpha,
        "s": s_val,
        "weight_decay": wd,
        "AUROC": ...,
        "CR": ...,
        "trainer": trainer,
      }
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    runs = []
    for alpha in alpha_list:
        for wd in weight_decay_list:
            model = make_model_fn().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

            trainer = CompatTrainer(
                model=model,
                optimizer=opt,
                alpha=alpha,
                slope=s_val,
                device=device,
            )

            trainer.fit(
                X_train,
                y_train,
                p_o_train,
                batch_size=batch_size,
                epochs=epochs,
                shuffle=True,
            )
            metrics = trainer.evaluate_holdout(X_val, y_val, p_o_val)

            runs.append({
                "family": "RBC",
                "alpha": alpha,
                "s": s_val,
                "weight_decay": wd,
                "AUROC": metrics["AUROC"],
                "CR": metrics["CR"],
                "trainer": trainer,
            })

    return runs


def run_full_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    p_o_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    p_o_val: np.ndarray,
    *,
    weight_decay_list: List[float],
    alpha_list: List[float],
    s_list: List[float],
    make_model_fn: Callable[[], nn.Module],
    lr: float = 1e-3,
    batch_size: int = 512,
    epochs: int = 8,
    device: str = "cpu",
    n_bootstrap: int = 45,
    n_shuffle: int = 5,
    seed: int = 0,
) -> Tuple[
    List[Dict[str, float]],             # bce_runs
    Dict[float, List[Dict[str, float]]] # rbc_runs_by_s
]:
    """
    Convenience wrapper that does:
      1. train the BCE pool once (alpha=1.0 with resampling)
      2. train the RBC grid for each s in s_list

    Returns:
      bce_runs: list of all baseline BCE models (pooled over weight_decay_list and reps)
      rbc_runs_by_s: dict mapping s -> list of RBC runs (each run is one alpha/weight_decay model)
    """

    # 1. baseline BCE models (alpha = 1)
    bce_runs = train_bce_pool_once(
        X_train, y_train, p_o_train,
        X_val, y_val, p_o_val,
        weight_decay_list=weight_decay_list,
        make_model_fn=make_model_fn,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        n_bootstrap=n_bootstrap,
        n_shuffle=n_shuffle,
        seed=seed,
    )

    # 2. RBC grids per s
    rbc_runs_by_s: Dict[float, List[Dict[str, float]]] = {}
    for s_val in s_list:
        rbc_runs_by_s[s_val] = train_rbc_grid_for_s(
            X_train, y_train, p_o_train,
            X_val, y_val, p_o_val,
            alpha_list=alpha_list,
            weight_decay_list=weight_decay_list,
            s_val=s_val,
            make_model_fn=make_model_fn,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            seed=seed,
        )

    return bce_runs, rbc_runs_by_s


############################################
# selection using beta, and heatmap construction
############################################

def _select_best_by_beta(candidates: List[Dict[str, float]], beta: float) -> Dict[str, float]:
    """
    Given a list of runs, pick the one with the best
      score_beta = beta * AUROC + (1 - beta) * CR
    """
    best_run = None
    best_score = -np.inf
    for r in candidates:
        score = beta * r["AUROC"] + (1.0 - beta) * r["CR"]
        if score > best_score:
            best_score = score
            best_run = r
    return best_run


def build_heatmaps(
    bce_runs: List[Dict[str, float]],
    rbc_runs_by_s: Dict[float, List[Dict[str, float]]],
    *,
    alpha_list: List[float],
    beta_list: List[float],
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Build delta AUROC and delta C^R heatmaps for each s.

    Steps:
      - For each beta, choose the best BCE run overall (across weight decays and reps).
      - For each s:
          For each alpha:
            For each beta:
              choose best RBC run for this alpha and this s (across weight decays),
              using the same beta weighted score.
              Compute deltas:
                dAUROC = AUROC_RBC - AUROC_best_BCE(beta)
                dCR    = CR_RBC    - CR_best_BCE(beta)

    Returns dict keyed by s:
      {
        s_val: {
          "delta_AUROC": shape [len(alpha_list), len(beta_list)],
          "delta_CR":    shape [len(alpha_list), len(beta_list)],
          "alpha_list": alpha_list,
          "beta_list": beta_list,
        }
      }
    """

    # baseline anchors: best BCE model for each beta
    bce_by_beta: Dict[float, Dict[str, float]] = {}
    for beta in beta_list:
        best_bce = _select_best_by_beta(bce_runs, beta)
        bce_by_beta[beta] = {
            "AUROC": best_bce["AUROC"],
            "CR": best_bce["CR"],
        }

    # now build matrices per s
    by_s: Dict[float, Dict[str, np.ndarray]] = {}
    for s_val, runs_s in rbc_runs_by_s.items():
        # organize RBC runs for quick lookup
        # key is alpha
        runs_by_alpha: Dict[float, List[Dict[str, float]]] = {}
        for r in runs_s:
            a = r["alpha"]
            runs_by_alpha.setdefault(a, []).append(r)

        dAU_mat = np.zeros((len(alpha_list), len(beta_list)))
        dCR_mat = np.zeros((len(alpha_list), len(beta_list)))

        for ai, alpha in enumerate(alpha_list):
            alpha_pool = runs_by_alpha.get(alpha, [])
            for bi, beta in enumerate(beta_list):
                if not alpha_pool:
                    dAU_mat[ai, bi] = np.nan
                    dCR_mat[ai, bi] = np.nan
                    continue

                best_rbc = _select_best_by_beta(alpha_pool, beta)
                base = bce_by_beta[beta]

                dAU_mat[ai, bi] = best_rbc["AUROC"] - base["AUROC"]
                dCR_mat[ai, bi] = best_rbc["CR"] - base["CR"]

        by_s[s_val] = {
            "delta_AUROC": dAU_mat,
            "delta_CR": dCR_mat,
            "alpha_list": list(alpha_list),
            "beta_list": list(beta_list),
        }

    return by_s


def plot_heatmaps_by_s(
    by_s: Dict[float, Dict[str, np.ndarray]],
    *,
    cmap: str = "RdBu",
    tick_decimals: int = 1,          # how many decimals to show on α/β ticks
    tick_as_percent: bool = False,   # show 0..1 as 0%..100%
    share_colorbar: bool = True,     # one colorbar per metric column
):
    """
    Make one row per s.

    For each s row:
      left  panel shows ΔC^R
      right panel shows ΔAUROC

    y axis is alpha, x axis is beta.
    Colors are centered at 0 using a diverging colormap.
    Cells are square (aspect="equal").

    Parameters
    ----------
    by_s : dict
        Output of build_heatmaps(); for each s:
          - 'delta_AUROC' : [len(alpha_list), len(beta_list)]
          - 'delta_CR'    : [len(alpha_list), len(beta_list)]
          - 'alpha_list'  : array-like
          - 'beta_list'   : array-like
    cmap : str
        Matplotlib colormap.
    tick_decimals : int
        Number of decimals for α/β tick labels (default 1).
    tick_as_percent : bool
        If True, format α/β ticks as percentages (e.g., 0.3 → 30%).
    share_colorbar : bool
        If True, add one colorbar per column across all rows.
    """

    def _fmt_levels(levels):
        if tick_as_percent:
            return [f"{int(round(100*float(x)))}%" for x in levels]
        # fixed decimals, then strip trailing zeros and dot
        lbls = [f"{float(x):.{tick_decimals}f}" for x in levels]
        return [s.rstrip('0').rstrip('.') if tick_decimals > 0 else s for s in lbls]

    s_vals = list(sorted(by_s.keys()))

    # gather global vlims so colors are comparable across rows
    all_dA = np.concatenate([np.ravel(by_s[s]["delta_AUROC"]) for s in s_vals])
    all_dC = np.concatenate([np.ravel(by_s[s]["delta_CR"])    for s in s_vals])
    all_dA = all_dA[np.isfinite(all_dA)]
    all_dC = all_dC[np.isfinite(all_dC)]
    vlim_A = np.max(np.abs(all_dA)) if all_dA.size else 1e-3
    vlim_C = np.max(np.abs(all_dC)) if all_dC.size else 1e-3

    # set up subplots: 2 columns (ΔC^R, ΔAUROC), one row per s
    fig, axes = plt.subplots(len(s_vals), 2, figsize=(8, 3.6 * len(s_vals)))
    if len(s_vals) == 1:
        axes = np.array([axes])  # normalize to 2D index

    ims_left, ims_right = [], []

    for row, s_val in enumerate(s_vals):
        block  = by_s[s_val]
        dAU    = block["delta_AUROC"]
        dCR    = block["delta_CR"]
        alphas = np.asarray(block["alpha_list"])
        betas  = np.asarray(block["beta_list"])

        x_ticks = np.arange(len(betas))
        y_ticks = np.arange(len(alphas))
        x_lbls  = _fmt_levels(betas)
        y_lbls  = _fmt_levels(alphas)

        # LEFT: ΔC^R
        ax = axes[row, 0]
        im = ax.imshow(dCR, origin="lower", aspect="equal", cmap=cmap,
                       vmin=-vlim_C, vmax=+vlim_C)
        ims_left.append(im)
        ax.set_title(r"$\Delta C^R$")
        ax.set_xticks(x_ticks); ax.set_xticklabels(x_lbls, rotation=0)
        ax.set_yticks(y_ticks); ax.set_yticklabels(y_lbls)
        ax.set_xlabel(r"$\beta$ (selection weight)" + (" [%]" if tick_as_percent else ""))
        ax.set_ylabel(r"$\alpha$ (training BCE weight)" + (" [%]" if tick_as_percent else ""))
        ax.tick_params(axis='both', length=0)

        # RIGHT: ΔAUROC
        ax = axes[row, 1]
        im = ax.imshow(dAU, origin="lower", aspect="equal", cmap=cmap,
                       vmin=-vlim_A, vmax=+vlim_A)
        ims_right.append(im)
        ax.set_title(r"$\Delta \mathrm{AUROC}$")
        ax.set_xticks(x_ticks); ax.set_xticklabels(x_lbls, rotation=0)
        ax.set_yticks(y_ticks); ax.set_yticklabels(y_lbls)
        ax.set_xlabel(r"$\beta$ (selection weight)" + (" [%]" if tick_as_percent else ""))
        ax.set_ylabel(r"$\alpha$ (training BCE weight)" + (" [%]" if tick_as_percent else ""))
        ax.tick_params(axis='both', length=0)

    # colorbars
    if share_colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # one cbar for left column
        divider = make_axes_locatable(axes[0,0])
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cb = fig.colorbar(ims_left[0], cax=cax)
        cb.ax.set_ylabel(r"$\Delta C^R$", rotation=90, va="bottom")
        # one cbar for right column
        divider = make_axes_locatable(axes[0,1])
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cb = fig.colorbar(ims_right[0], cax=cax)
        cb.ax.set_ylabel(r"$\Delta \mathrm{AUROC}$", rotation=90, va="bottom")
    else:
        for ax, im in zip(axes[:,0], ims_left):
            fig.colorbar(im, ax=ax)
        for ax, im in zip(axes[:,1], ims_right):
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
