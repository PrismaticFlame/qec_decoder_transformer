// alphaBest contains the best results from AlphaQubit with distances 3 and 5
var alphaBestD3 = 0.02901
var alphaBestD5 = 0.02748

export default class Data {

    constructor(csv, basis, d, r, shots, d2 = 0, r2 = 0) {
        this.csv = csv
        this.basis = basis
        this.d = d
        this.r = r
        this.d2 = d2
        this.r2 = r2
        this.shots = shots
        if (d == 3) // Add the best AlphaQubit based on distance
            this.alphaBest = alphaBestD3
        else if (d == 5)
            this.alphaBest = alphaBestD5
        else
            this.alphaBest = 0
        if (d2 != 0) { // If two distances are involved, add both AlphaQubit best results
            this.alphaBestD3 = alphaBestD3
            this.alphaBestD5 = alphaBestD5
        }
    }
}