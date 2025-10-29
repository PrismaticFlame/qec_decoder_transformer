def create_rep_code_stim_string(distance, rounds, p):
    n = 2*distance - 1
    qs = ' '.join(map(str, range(n)))
    ms = ' '.join(map(str, range(1, n, 2)))
    ds = ' '.join(map(str, range(0, n, 2)))
    down = ' '.join(map(str, range(n-1)))
    up = ' '.join(map(str, reversed(range(1, n))))
    nm = distance-1
    nd = distance
    
    string = f"""
    R {qs}
    X_ERROR({p}) {qs}
    TICK
    CX {down}
    DEPOLARIZE2({p}) {down}
    DEPOLARIZE1({p}) {n-1}
    TICK
    CX {up}
    DEPOLARIZE1({p}) {0}
    DEPOLARIZE2({p}) {up}
    TICK
    X_ERROR({p}) {ms}
    M {ms}
    DEPOLARIZE1({p}) {ds}
    """
    
    for i, j in enumerate(range(1, n, 2)):
        string += f'DETECTOR({j}, 0) rec[{-nm + i}]\n'

    if rounds > 2:
        string += f"""REPEAT {rounds-2} {{
            R {ms}
            X_ERROR({p}) {ms}
            DEPOLARIZE1({p}) {ds}
            TICK
            CX {down}
            DEPOLARIZE2({p}) {down}
            DEPOLARIZE1({p}) {n-1}
            TICK
            CX {up}
            DEPOLARIZE1({p}) {0}
            DEPOLARIZE2({p}) {up}
            TICK
            X_ERROR({p}) {ms}
            M {ms}
            DEPOLARIZE1({p}) {ds}
            SHIFT_COORDS(0, 1)
            """
        
        for i, j in enumerate(range(1, n, 2)):
            string += f'    DETECTOR({j}, 0) rec[{-nm + i}] rec[{-2*nm + i}]\n'
        string += "}\n"
        
    string += f"""
    R {ms}
    X_ERROR({p}) {ms}
    DEPOLARIZE1({p}) {ds}
    TICK
    CX {down}
    DEPOLARIZE2({p}) {down}
    DEPOLARIZE1({p}) {n-1}
    TICK
    CX {up}
    DEPOLARIZE1({p}) {0}
    DEPOLARIZE2({p}) {up}
    TICK
    X_ERROR({p}) {ms} {ds}
    M {ms} {ds}
    SHIFT_COORDS(0, 1)
    """
    
    for i, j in enumerate(range(1, n, 2)):
        string += f'DETECTOR({j}, 0) rec[{-nm - nd + i}] rec[{-2*nm - nd + i}] \n'
    for i, j in enumerate(range(1, n, 2)):
        string += f'DETECTOR({j}, 0) rec[{-nm - nd + i}] rec[{-nd + i}] rec[{-nd + i + 1}]\n'

    string += "OBSERVABLE_INCLUDE(0) rec[-1]\n"

    return string 