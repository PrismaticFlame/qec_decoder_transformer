# Values taken from supplementory paper, in percent
alphaBestD3 = 2.901 * 0.01
alphaBestD5 = 2.748 * 0.01

class Data:
    csv = ""
    alphaBest = 0
    basis = ""
    d = 0 # Distance
    r = 0 # Rounds
    shots = 0
    # title = ""
    # subtitle = ""

    def __init__(self, csv, basis, d, r, shots):
        self.csv = csv
        self.basis = basis
        self.d = d
        if d == 3:
            self.alphaBest = alphaBestD3
        elif d == 5:
            self.alphaBest = alphaBestD5
        else:
            self.alphaBest = 0
        self.base = 0
        self.r = r
        self.shots = shots
