import torch
import torch.nn as nn


class StabilizerEmbedding(nn.Module):
    """
    AlphaQubit-style stabilizer embedding (hard input):

    S_ni = E_stab(i) + E_cycle(n) + E_value(s_ni)

    - E_stab(i):   which stabilizer (spatial identity / type)
    - E_cycle(n):  which measurement round (time)
    - E_value(s):  syndrome value 0 or 1
    """

    def __init__(self, num_stab: int, num_cycles: int, d_model: int = 256):
        super().__init__()

        self.stab_emb = nn.Embedding(num_stab, d_model)
        self.cycle_emb = nn.Embedding(num_cycles, d_model)
        self.val_emb = nn.Embedding(2, d_model)  # syndrome bit 0/1

    def forward(self, syndrome, stab_id, cycle_id):
        """
        Args:
            syndrome: (B, L)   values 0/1
            stab_id:  (L,) or (B, L)
            cycle_id: (L,) or (B, L)

        Returns:
            S: (B, L, d_model)
        """
        B, L = syndrome.shape

        # ---- handle stab_id shape ----
        if stab_id.dim() == 1:
            # (L,) -> (1, L) -> (B, L)
            stab_id = stab_id.unsqueeze(0).expand(B, L)
        elif stab_id.dim() == 2:
            # (B, L) 或 (1, L)
            if stab_id.size(0) == 1 and B > 1:
                stab_id = stab_id.expand(B, L)
            # 如果本來就是 (B, L) 就不動

        # ---- handle cycle_id shape ----
        if cycle_id.dim() == 1:
            cycle_id = cycle_id.unsqueeze(0).expand(B, L)
        elif cycle_id.dim() == 2:
            if cycle_id.size(0) == 1 and B > 1:
                cycle_id = cycle_id.expand(B, L)

        e_stab = self.stab_emb(stab_id)    # (B, L, d_model)
        e_cycle = self.cycle_emb(cycle_id) # (B, L, d_model)
        e_val = self.val_emb(syndrome)     # (B, L, d_model)

        return e_stab + e_cycle + e_val