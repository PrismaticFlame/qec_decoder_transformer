
import stim
import numpy as np
from typing import List, Tuple, Optional

#pretrain: hard input, logical label
#finetune: soft input, logical label
#test:     soft input

# Module-level constant for measurement operations
_MEAS_OPS = {"M", "MR", "MX", "MY", "MZ", "MPP"}


# ----------------------------
# 1) Parse circuit deps (DETECTOR / OBSERVABLE_INCLUDE) into absolute meas indices 
# ----------------------------

def _count_measurement_results(inst: stim.CircuitInstruction) -> int:
    """
    Return how many measurement results this instruction appends to the measurement record.

    IMPORTANT:
    - For M/MX/MY/MZ/MR/... : one result per measured qubit target 所以用target_copy ex. X0*X1*X2, Z3*Z4
    - For MPP: one result per target_group 
    - Other ops: 0
    """
    name = inst.name
    if name == "MPP":
        return len(inst.target_groups())
    if name in _MEAS_OPS:
        # For these measurement instructions, each qubit target corresponds to one measurement result.
        return len(inst.targets_copy())
    return 0


# ----------------------------
# 2) Correctly extract DETECTOR -> absolute measurement indices
# ----------------------------
def extract_detector_rec_dependencies(circ: stim.Circuit) -> List[List[int]]:
    """
    Returns:
      deps: list length D=circ.num_detectors
            deps[d] = sorted list of absolute measurement indices referenced by DETECTOR d.
    """
    deps: List[List[int]] = []
    meas_count = 0  # how many measurements have happened so far in the flattened circuit

    for inst in circ.flattened_operations():
        name = inst.name

        # Count how many measurement results this instruction appends
        # Stim appends one result per target for M / MR / MX / MY / MPP.
        if name in {"M", "MR", "MX", "MY", "MPP"}:
            meas_count += len(inst.targets_copy())

        if name == "DETECTOR":
            idxs: List[int] = []
            for t in inst.targets_copy():
                if t.is_rec():
                    # For rec[-k], t.value is typically negative (e.g. -1).
                    # Absolute index = meas_count + t.value  (since t.value is negative)
                    abs_i = meas_count + int(t.value)
                    idxs.append(abs_i)
                else:
                    # coords / separators are not rec targets; ignore
                    pass
            deps.append(sorted(idxs))

    D = int(circ.num_detectors)
    if len(deps) != D:
        raise RuntimeError(f"Parsed {len(deps)} DETECTOR instructions but circ.num_detectors={D}.")
    return deps

def extract_observable_rec_dependencies(circ: stim.Circuit) -> List[List[int]]:
    """
    Returns:
      obs_deps: list length K=circ.num_observables
                obs_deps[k] = sorted list of absolute meas indices referenced by OBSERVABLE_INCLUDE k.
    """
    K = int(circ.num_observables)
    obs_deps: List[List[int]] = [[] for _ in range(K)]
    meas_count = 0

    for inst in circ.flattened_operations():
        name = inst.name

        if name in {"M", "MR", "MX", "MY", "MPP"}:
            meas_count += len(inst.targets_copy())

        if name == "OBSERVABLE_INCLUDE":
            args = inst.gate_args_copy()
            if len(args) < 1:
                raise RuntimeError("OBSERVABLE_INCLUDE missing observable index gate arg.")
            k = int(args[0])
            if not (0 <= k < K):
                raise RuntimeError(f"Observable index out of range: k={k}, K={K}")

            idxs: List[int] = []
            for t in inst.targets_copy():
                if t.is_rec():
                    abs_i = meas_count + int(t.value)
                    idxs.append(abs_i)
            obs_deps[k].extend(idxs)

    # normalize (sorted unique)
    obs_deps = [sorted(set(v)) for v in obs_deps]
    return obs_deps

# ----------------------------
# 2) Recompute HARD bits from deps (XOR parity)
# ----------------------------
def hard_bits_from_meas_bits_by_deps(meas_bits: np.ndarray, deps: List[List[int]]) -> np.ndarray:
    """
    Generic XOR parity recompute from raw meas_bits using deps.
    Returns: (N, len(deps)) uint8
    """
    N, M = meas_bits.shape
    D = len(deps)
    out = np.zeros((N, D), dtype=np.uint8)

    mb = meas_bits.astype(np.uint8, copy=False)
    for d, idxs in enumerate(deps):
        if not idxs:
            continue
        mx = max(idxs)
        if mx >= M:
            raise ValueError(f"deps[{d}] references meas index {mx} but meas_bits has M={M}.")
        out[:, d] = np.bitwise_xor.reduce(mb[:, idxs], axis=1)

    return out


def hard_detectors_from_meas_bits_by_deps(meas_bits: np.ndarray, deps: List[List[int]]) -> np.ndarray:
    return hard_bits_from_meas_bits_by_deps(meas_bits, deps)


def hard_observables_from_meas_bits_by_deps(meas_bits: np.ndarray, obs_deps: List[List[int]]) -> np.ndarray:
    return hard_bits_from_meas_bits_by_deps(meas_bits, obs_deps)


# ----------------------------
# 3) Soft detector construction (your existing math helpers assumed)
# ----------------------------
def build_soft_detectors_from_measurements(
    circ: stim.Circuit,
    meas_bits: np.ndarray,  # (N, M) in {0,1}
    *,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
    event_llr_convention: str = "event1_over_event0",
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Returns:
      det_soft: (N, D) float32
      deps: list length D, each is list of measurement indices used by that detector

    event_llr_convention:
      - "parity0_over_parity1": returns LLR(parity=0 / parity=1)  (pll)
      - "event1_over_event0":   returns LLR(event=1 / event=0)    (=-pll)
    """
    deps = extract_detector_rec_dependencies(circ)  # length D

    # You already have these helpers elsewhere:
    #   meas_llr: (N,M)
    meas_llr = measurement_bits_to_llr(meas_bits, mu=mu, sigma=sigma, seed=seed)

    N, M = meas_llr.shape
    D = len(deps)
    det_soft = np.zeros((N, D), dtype=np.float32)

    for d, idxs in enumerate(deps):
        if not idxs:
            continue

        mx = max(idxs)
        if mx >= M:
            raise ValueError(
                f"Detector {d} references measurement index {mx} but meas_bits has M={M} columns. "
                f"(Did you sample meas_bits from the SAME circuit? Are you flattening correctly?)"
            )

        llr_k = meas_llr[:, idxs]         # (N,K)
        pll = parity_llr_from_llrs(llr_k) # (N,) = log P(parity=0)/P(parity=1)

        if event_llr_convention == "parity0_over_parity1":
            det_soft[:, d] = pll
        elif event_llr_convention == "event1_over_event0":
            det_soft[:, d] = -pll
        else:
            raise ValueError(f"Unknown event_llr_convention={event_llr_convention}")

    return det_soft.astype(np.float32), deps




# ----------------------------
# 3) Convert meas bits -> per-measurement LLR (surrogate analog)用高斯模擬soft input
# ----------------------------

def measurement_bits_to_llr(
    meas_bits: np.ndarray,
    *,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0
) -> np.ndarray:
    """
    meas_bits: (N, M) in {0,1}
    returns:   (N, M) float32 LLR where:
              + => more likely 0
              - => more likely 1

    Surrogate model: y ~ Normal(mean=+mu if bit=0 else -mu, sigma)
    LLR = (2*mu/sigma^2) * y, llr > 0 → 比較像 0,llr < 0 → 比較像 1,|llr| → 信心強度
    """
    rng = np.random.default_rng(seed) #可重現亂數 只要seed固定
    mean = np.where(meas_bits == 0, +mu, -mu).astype(np.float32)    #where=if-else,measurement bit = 0 → 理想 readout 在 +μ, measurement bit = 0 → 理想 readout 在 +μ
    y = mean + rng.normal(0.0, sigma, size=mean.shape).astype(np.float32) #加上雜訊：高斯(平均, 標準差,輸出shape)
    llr = (2.0 * mu / (sigma * sigma)) * y 
    return llr.astype(np.float32)


# ----------------------------
# 4) Parity LLR from independent bit LLRs
# ----------------------------

def parity_llr_from_llrs(llrs: np.ndarray, eps: float = 1e-6) -> np.ndarray: #XOR
    """
    llrs: (..., K)  independent bit LLRs
    return: (...,)  parity LLR = log P(parity=0) / P(parity=1)

    Standard identity:
      L_parity = 2 * atanh( prod_i tanh(L_i/2) )
    """
    x = np.tanh(llrs / 2.0)  # 先把soft input映射到（-1:input是1,1:input是0）
    prod = np.prod(x, axis=-1) #用1,-1乘法代替xor(-1:output是1,1:output是0）如果有一個soft input信心很小 x=0, prod=0
    prod = np.clip(prod, -1.0 + eps, 1.0 - eps) #arctanh(1)=inf, arctanh(-1)=-inf 所以這裡要微調
    return (2.0 * np.arctanh(prod)).astype(np.float32) #雙曲正切法則arctanh是tanh的反函數 output映射回機率






# ----------------------------
# 4) One-sampling dataset generator: meas_bits is the single source of truth
# ----------------------------
def apply_leakage_mask(
    cycle_id: np.ndarray,
    num_shots: int,
    *,
    p_leak_stab: float = 1e-3,
    p_leak_data: float = 3e-3,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate leakage mask for detectors.
    
    Args:
        cycle_id: (D,) cycle ID for each detector
        num_shots: Number of shots (N)
        p_leak_stab: Leakage probability for stabilizer detectors (intermediate cycles)
        p_leak_data: Leakage probability for data qubit detectors (last cycle)
        seed: Random seed
    
    Returns:
        leakage_mask: (N, D) bool array, True = kept, False = leaked
    """
    D = len(cycle_id)
    rng = np.random.default_rng(seed)
    
    max_cycle = int(cycle_id.max())
    leakage_mask = np.ones((num_shots, D), dtype=bool)
    
    # For each detector, determine if it's a stabilizer or data qubit detector
    for d in range(D):
        c = int(cycle_id[d])
        
        # Last cycle: data qubit detectors (higher leakage rate)
        if c == max_cycle:
            p_leak = p_leak_data
        else:
            # Intermediate cycles: stabilizer detectors (lower leakage rate)
            p_leak = p_leak_stab
        
        # Sample leakage for each shot
        leak = rng.random(num_shots) < p_leak
        leakage_mask[:, d] = ~leak  # True = kept, False = leaked
    
    return leakage_mask


def gen_soft_surrogate_dataset(
    *,
    distance: int = 3,
    rounds: int = 5,
    p: float = 1e-3,
    shots: int = 20000,
    mu: float = 1.0,
    sigma: float = 1.0,
    seed_meas: int = 0,
    seed_analog: int = 0,
    do_sanity_check: bool = True,
    sanity_check_against_stim: bool = False,  # best-effort only
    # Leakage parameters
    apply_leakage: bool = True,
    p_leak_stab: float = 1e-3,  # 0.1% leakage for stabilizer detectors
    p_leak_data: float = 3e-3,  # 0.3% leakage for data qubit detectors
    seed_leakage: int = 0,
) -> Tuple[stim.Circuit, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      circ
      det_hard: (N, D) int8   (derived from meas_bits)
      det_soft: (N, D) float32
      obs:      (N, K) int8   (derived from meas_bits)
                For surface_code:rotated_memory_z with K=2:
                  obs[:, 0] = X-type logical error label
                  obs[:, 1] = Z-type logical error label
                (Order is adjusted to match dataset.py expectation)
      leakage_mask: (N, D) bool, optional
                True = detector kept, False = detector leaked (missing)
                Only returned if apply_leakage=True

    Key guarantee:
      det_hard / det_soft / obs are aligned shot-by-shot because they all come from the SAME meas_bits samples.
    """
    circ = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

    # Sample raw measurement bits ONCE (single source of truth)
    meas_sampler = circ.compile_sampler()
    meas_bits = meas_sampler.sample(shots=shots, seed=seed_meas).astype(np.uint8)  # (N,M)

    # Parse deps
    det_deps = extract_detector_rec_dependencies(circ) # 解析電路中所有的 DETECTOR 指令 找出哪些測量位元（measurements）被用來組合出某個偵測事件（detection event）
    obs_deps = extract_observable_rec_dependencies(circ) #解析電路中所有的 OBSERVABLE_INCLUDE 指令 找出哪些數據位元的最終測量值構成了邏輯算符（Logical Operator）

    # Recompute hard labels from meas_bits
    det_hard_u8 = hard_detectors_from_meas_bits_by_deps(meas_bits, det_deps)         # (N,D) Detector Event
    obs_u8      = hard_observables_from_meas_bits_by_deps(meas_bits, obs_deps)       # (N,K) Logical Label
    
    # For surface_code:rotated_memory_z, stim typically returns 2 observables:
    #   observable 0: Z-type logical operator (memory_z mode stores Z information)
    #   observable 1: X-type logical operator
    # However, dataset.py expects [X, Z] order (labels[:, 0] = X, labels[:, 1] = Z)
    # So we need to reorder: [Z, X] -> [X, Z]
    K = obs_u8.shape[1]
    if K == 2:
        # Reorder observables: [Z, X] -> [X, Z] to match dataset expectation
        obs_u8 = obs_u8[:, [1, 0]]  # Swap columns: obs[:, 0] becomes X, obs[:, 1] becomes Z
    elif K != 1:
        # Warn if unexpected number of observables
        import warnings
        warnings.warn(
            f"Expected 1 or 2 observables for surface code, got {K}. "
            f"Observable order may not match dataset expectation (X first, Z second)."
        )

    # Soft detectors from meas LLR
    det_soft, _ = build_soft_detectors_from_measurements(
        circ,
        meas_bits,
        mu=mu,
        sigma=sigma,
        seed=seed_analog,
        event_llr_convention="event1_over_event0",
    )
    
    # Apply leakage masking if requested
    leakage_mask = None
    if apply_leakage:
        # Extract cycle_id from circuit to determine detector types
        # We need to build a cycle_id array for detectors
        # Use detector coordinates to infer cycle_id
        D = len(det_deps)  # Reuse det_deps from above
        
        # Build cycle_id from circuit structure
        # For surface_code:rotated_memory_z, we can infer cycle_id from detector coordinates
        # or use a simpler heuristic: last round has data qubit detectors
        coords_map = circ.get_detector_coordinates()
        cycle_id_array = np.zeros(D, dtype=np.int64)
        
        for d in range(D):
            c = coords_map.get(d, None)
            if c is not None and len(c) > 2:
                # Use time coordinate (t) to determine cycle
                cycle_id_array[d] = int(np.floor(c[2]))
            else:
                # Fallback: assume all detectors are in cycle 0 (will be adjusted)
                cycle_id_array[d] = 0
        
        # Normalize cycle_id to start from 0
        cycle_id_array = cycle_id_array - cycle_id_array.min()
        
        # Apply leakage
        leakage_mask = apply_leakage_mask(
            cycle_id_array,
            shots,
            p_leak_stab=p_leak_stab,
            p_leak_data=p_leak_data,
            seed=seed_leakage,
        )
        
        # Apply leakage mask to detectors (set leaked detectors to 0 or NaN)
        # For hard: set to 0 (no detection event)
        det_hard_u8 = det_hard_u8 * leakage_mask.astype(np.uint8)
        # For soft: set to 0.0 (neutral LLR)
        det_soft = det_soft * leakage_mask.astype(np.float32)

    if do_sanity_check:
        D_stim = int(circ.num_detectors)
        K_stim = int(circ.num_observables)

        if det_hard_u8.shape != (shots, D_stim):
            raise RuntimeError(f"det_hard shape mismatch: {det_hard_u8.shape} vs {(shots, D_stim)}")

        if det_soft.shape != (shots, D_stim):
            raise RuntimeError(f"det_soft shape mismatch: {det_soft.shape} vs {(shots, D_stim)}")

        # After reordering, obs should have same number of columns as original
        if obs_u8.shape != (shots, K_stim):
            raise RuntimeError(f"obs shape mismatch: {obs_u8.shape} vs {(shots, K_stim)}")
        
        # For surface code, verify we have expected number of observables
        if K_stim == 2:
            print(f"[INFO] Found 2 observables: obs[:, 0] = X-type, obs[:, 1] = Z-type (reordered to match dataset expectation)")
        elif K_stim == 1:
            print(f"[INFO] Found 1 observable: using same label for both X and Z (backward compatibility)")
        else:
            print(f"[WARNING] Found {K_stim} observables, expected 1 or 2 for surface code")

        # Internal consistency check: recompute again and ensure identical (should be exact)有時候記憶體在跨進程傳遞時會發生錯誤
        det_hard_u8_2 = hard_detectors_from_meas_bits_by_deps(meas_bits, det_deps)
        if np.any(det_hard_u8_2 != det_hard_u8):
            raise RuntimeError("Internal sanity check failed: detector recomputation not stable.")

        if sanity_check_against_stim:
            # Best-effort: compare against Stim's detector_sampler when using the SAME seed.
            # Note: Stim does not guarantee that detector_sampler(seed=X) matches sampler(seed=X)
            # shot-by-shot across versions/implementations, so treat mismatch as informational.
            det_sampler = circ.compile_detector_sampler()
            det_hard_stim, obs_stim = det_sampler.sample(shots=shots, separate_observables=True, seed=seed_meas)
            det_hard_stim = det_hard_stim.astype(np.uint8)
            obs_stim = obs_stim.astype(np.uint8)
            
            # Reorder obs_stim to match our reordering (if K=2)
            if obs_stim.shape[1] == 2:
                obs_stim_reordered = obs_stim[:, [1, 0]]  # [Z, X] -> [X, Z]
            else:
                obs_stim_reordered = obs_stim

            mismatch_det = float(np.mean(det_hard_stim != det_hard_u8))
            mismatch_obs = float(np.mean(obs_stim_reordered != obs_u8))
            if mismatch_det != 0.0 or mismatch_obs != 0.0:
                print(
                    "[INFO] Best-effort cross-check vs Stim sampler differs.\n"
                    f"  detector mismatch rate = {mismatch_det:.6f}\n"
                    f"  observable mismatch rate = {mismatch_obs:.6f}\n"
                    "This can happen because sampler() and detector_sampler() may not be RNG-aligned.\n"
                    "Your dataset is still self-consistent (everything derived from meas_bits)."
                )

    if apply_leakage:
        return circ, det_hard_u8.astype(np.int8), det_soft.astype(np.float32), obs_u8.astype(np.int8), leakage_mask
    else:
        return circ, det_hard_u8.astype(np.int8), det_soft.astype(np.float32), obs_u8.astype(np.int8), None
# ----------------------------
# 8) Run + Save
# ----------------------------
if __name__ == "__main__":
    circ, det_hard, det_soft, obs, leakage_mask = gen_soft_surrogate_dataset(
        distance=3,
        rounds=5,
        p=1e-3,
        shots=20000,
        mu=1.2,
        sigma=1.0,
        seed_meas=42,
        seed_analog=42,
        do_sanity_check=True,
        apply_leakage=True,
        p_leak_stab=1e-3,
        p_leak_data=3e-3,
        seed_leakage=42,
    )

    # Save data with explicit X and Z labels if we have 2 observables
    save_dict = {
        "det_hard": det_hard,
        "det_soft": det_soft,
        "obs": obs,
    }
    
    if obs.shape[1] == 2:
        save_dict["obs_x"] = obs[:, 0]  # X-type logical error label
        save_dict["obs_z"] = obs[:, 1]  # Z-type logical error label
    
    # Save leakage mask if available
    if leakage_mask is not None:
        save_dict["leakage_mask"] = leakage_mask.astype(np.uint8)  # Save as uint8 to save space
    
    np.savez_compressed("../data/stim_soft_surrogate.npz", **save_dict)

    print("Saved ../data/stim_soft_surrogate.npz")
    print("det_hard:", det_hard.shape, det_hard.dtype)
    print("det_soft:", det_soft.shape, det_soft.dtype)
    print("obs     :", obs.shape, obs.dtype)
    if obs.shape[1] == 2:
        print("  obs[:, 0] = X-type logical error label")
        print("  obs[:, 1] = Z-type logical error label")
    if leakage_mask is not None:
        leak_rate = 1.0 - leakage_mask.mean()
        print(f"leakage_mask: {leakage_mask.shape}, leak rate: {leak_rate:.4%}")
    print("num_detectors:", circ.num_detectors, "num_measurements:", circ.num_measurements)

    #syndrome=detection event(ancilla qubit的 measurement 之間做 parity 得到)
    #DETECTOR 定義了一個「detection event」
