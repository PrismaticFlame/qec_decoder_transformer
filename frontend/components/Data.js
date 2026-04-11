var alphaBestD3 = 2.901 * 0.01
var alphaBestD5 = 2.748 * 0.01

export default class Data {

    constructor(csv, basis, d, r, shots) {
        this.csv = csv
        this.basis = basis
        this.d = d
        this.r = r
        this.shots = shots
        if (d == 3) 
            this.alphaBest = alphaBestD3
        else if (d == 5)
            this.alphaBest = alphaBestD5
        else
            this.alphaBest = 0
    }
}