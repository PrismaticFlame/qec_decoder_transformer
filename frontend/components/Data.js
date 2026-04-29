var alphaBestD3 = 2.901 * 0.01
var alphaBestD5 = 2.748 * 0.01

export default class Data {

    constructor(csv, basis, d, r, shots, d2 = 0, r2 = 0) {
        this.csv = csv
        this.basis = basis
        this.d = d
        this.r = r
        this.d2 = d2
        this.r2 = r2
        this.shots = shots
        if (d == 3) 
            this.alphaBest = alphaBestD3
        else if (d == 5)
            this.alphaBest = alphaBestD5
        else
            this.alphaBest = 0
        if (d2 != 0) {
            this.alphaBestD3 = alphaBestD3
            this.alphaBestD5 = alphaBestD5
        }
    }
}