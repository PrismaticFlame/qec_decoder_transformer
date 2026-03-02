import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime, timedelta
import time

# from torchsummary import summary

from qec.transformer_decoder.dataset import SyndromeDataset
from qec.transformer_decoder.model import QECAlphaTransformer

def format_seconds(s):
    """Convert seconds to H:MM:SS format."""
    return str(timedelta(seconds=int(s)))

def compute_ler_from_logits(logits, labels):
    """
    logits: (N, 1)
    labels: (N, 1) float 0/1
    returns: logical error rate (fraction of wrong predictions)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        errors = (preds != labels).float()
        ler = errors.mean().item()
    return ler

def train(data, logs, measurement=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # # 1. Load Stim-generated dataset
    # dets = np.loadtxt("stim_files/data/detector_samples.csv",
    #                   delimiter=",", dtype=np.uint8)
    
    # soft_measurements = np.loadtxt("stim_files/data/gausian_soft_mesurements.csv",
    #                   delimiter=",", dtype=np.float32)
    # logs = np.loadtxt("stim_files/data/logical_labels.csv",
    #                   delimiter=",", dtype=np.uint8)

    rounds = 5  # MUST match your Stim gen_dem() parameter

    full_dataset = SyndromeDataset(data, logs, distance=3, rounds=rounds, measurement=measurement)
    # full_dataset = SyndromeDataset(dets, logs, distance=3, rounds=rounds, measurement=False)
    # full_dataset = SyndromeDataset(soft_measurements, logs, distance=3, rounds=rounds, measurement=True)
    N = len(full_dataset)
    print(f"Total shots: {N}, num_detectors: {full_dataset.num_detectors}, "
          f"num_stab_per_round: {full_dataset.num_stab_per_round}, "
          f"num_cycles: {full_dataset.num_cycles}")

    # 2. Train/val split (e.g., 80% train, 20% val)
    val_ratio = 0.2
    n_val = int(N * val_ratio)
    n_train = N - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 3. Build model
    model = QECAlphaTransformer(
        num_stab=full_dataset.num_stab_per_round,
        num_cycles=full_dataset.num_cycles,
        distance=3,
        d_model=256,
        nhead=4,
        num_transformer_layers=3,
    ).to(device)

    # summary(model, input=())

    criterion = torch.nn.BCEWithLogitsLoss()

    # Lion + Scaling 規格
    optimizer = Lion(
        model.parameters(),
        lr=3e-4,              # 建議比 AdamW 稍微小一點；可以再調
        betas=(0.9, 0.99),    # Lion paper 推薦值
        weight_decay=1e-7,    # Table S3 Scaling column
    )

    # 3.5 start timer
    total_start = time.time()
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 4. Training loop
    best_ler = float("inf")
    best_epoch = 0
    num_epochs = 10
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch+1}/{num_epochs} started at "
              f"{datetime.now().strftime('%H:%M:%S')} ---")
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            synd = batch["syndrome"].to(device)
            label = batch["label"].to(device)
            stab_id = batch["stab_id"].to(device)
            cycle_id = batch["cycle_id"].to(device)

            logits = model(synd, stab_id, cycle_id)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * synd.size(0)

        avg_train_loss = total_loss / n_train

        # 5. Validation
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                synd = batch["syndrome"].to(device)
                label = batch["label"].to(device)
                stab_id = batch["stab_id"].to(device)
                cycle_id = batch["cycle_id"].to(device)

                logits = model(synd, stab_id, cycle_id)
                loss = criterion(logits, label)
                val_loss += loss.item() * synd.size(0)

                all_logits.append(logits.cpu())
                all_labels.append(label.cpu())

        avg_val_loss = val_loss / n_val
        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        val_ler = compute_ler_from_logits(val_logits, val_labels)
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{num_epochs} completed at "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"   ↳ Duration: {format_seconds(epoch_time)} "
              f"({epoch_time:.2f} seconds)")
        print(
            # f"Epoch {epoch+1:02d} | "
            f"train_loss = {avg_train_loss:.4e} | "
            f"val_loss = {avg_val_loss:.4e} | "
            f"val_LER = {val_ler:.4e}"
        )
        if val_ler < best_ler:
            best_ler = val_ler
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")
    
    total_time = time.time() - total_start
    print("\n" + "=" * 75)
    print(f"⏳ Training finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏳ Total training time: {format_seconds(total_time)} "
          f"({total_time:.2f} seconds)")
    print("=" * 75)
    return best_ler, best_epoch

def predict(data, logs, measurement=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    rounds = 5  # MUST match your Stim gen_dem() parameter
    
    full_dataset = SyndromeDataset(data, logs, distance=3, rounds=rounds, measurement=measurement)
    # 3. Build model
    model = QECAlphaTransformer(
        num_stab=full_dataset.num_stab_per_round,
        num_cycles=full_dataset.num_cycles,
        distance=3,
        d_model=256,
        nhead=4,
        num_transformer_layers=3,
    ).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(full_dataset, batch_size=256, shuffle=True)
    # 3. Load data
    model.eval()
    pred_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            synd = batch["syndrome"].to(device)
            label = batch["label"].to(device)
            stab_id = batch["stab_id"].to(device)
            cycle_id = batch["cycle_id"].to(device)

            logits = model(synd, stab_id, cycle_id)
            loss = criterion(logits, label)
            pred_loss += loss.item() * synd.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())

    
    pred_logits = torch.cat(all_logits, dim=0)
    pred_labels = torch.cat(all_labels, dim=0)
    pred_ler = compute_ler_from_logits(pred_logits, pred_labels)
    return pred_loss, pred_ler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load Stim-generated dataset
    dets = np.loadtxt("stim_files/data/detector_samples.csv",
                      delimiter=",", dtype=np.uint8)
    
    soft_measurements = np.loadtxt("stim_files/data/gausian_soft_mesurements.csv",
                      delimiter=",", dtype=np.float32)
    logs = np.loadtxt("stim_files/data/logical_labels.csv",
                      delimiter=",", dtype=np.uint8)
    
    
    train(soft_measurements, logs, measurement=True)
    pred_loss, pred_ler = predict(soft_measurements, logs, measurement=True)
    print(f"Predicted Loss: {pred_loss:.4e} | Predicted LER: {pred_ler:.4e}")
    

if __name__ == "__main__":
    main()

# ===== Lion Optimizer (簡單實作) =====
class Lion(torch.optim.Optimizer):
    r"""
    Lion optimizer (Chen et al., 2023), with decoupled weight decay.

    update:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t
      p   = p * (1 - lr * weight_decay) - lr * sign(m_t + v_t)
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # state init
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]

                # decoupled weight decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # EMA of gradients
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).add_(g, alpha=1 - beta2)

                # Lion update: sign(m + v)
                update = (m + v).sign_()
                p.add_(update, alpha=-lr)

        return loss
# ===== Lion end =====
