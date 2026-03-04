class Data:
    csv = ""
    base = 0
    basis = ""
    d = 0 # Distance
    r = 0 # Rounds
    shots = 0
    # title = ""
    # subtitle = ""

    def __init__(self, csv, base, basis, d, r, shots):
        self.csv = csv
        self.base = base
        self.basis = basis
        self.d = d
        self.r = r
        self.shots = shots
