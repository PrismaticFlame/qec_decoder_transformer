import Data from "../components/Data.js"

var basis = "X"
var x_data = new Data("../data/x_d3_r6_eval.csv", "X", 3, 6, 500)
var z_data = new Data("../data/z_d3_r6_eval.csv", "Z", 3, 6, 50000)

var dataDict = {
    'All': {'X': x_data, 'Z': z_data}, 
    'v3': {'X': x_data, 'Z': z_data},
    'v5': {'X': x_data, 'Z': z_data},
    'v6': {'X': x_data, 'Z': z_data},
}

var data = dataDict['All'][basis]

console.log("This is a transformer updated chart")

const ctx = document.getElementById('myChart');


window.onload = function() {
    document.getElementById("switchBasis").addEventListener("click", swapBasis)
    d3.csv(data.csv, makeChart)
}

function swapBasis() {
    console.log("button pressed")
    if (basis == "X") {
        basis = "Z"
    } else {
        basis = "X"
    }
    var data = dataDict['All'][basis]
    d3.csv(data.csv, makeChart)
    console.log(data)
}

function makeChart(myData) {
    console.log("this is running")
    var steps = myData.map(function(d) {return d.step});
    var devLer = myData.map(function(d) {return d.dev_ler})
    var bestLer = myData.map(function(d) {return d.best_ler})
    console.log(steps)
    console.log(devLer)

    new Chart(ctx, {
    type: 'line',
    options: {
        plugins: {
            title: {
                display: true,
                text: `${data.basis} basis, Distance ${data.d}`
            },
            subtitle: {
                display: true,
                text: `${data.shots} shots, Rounds ${data.r}`,
                color: 'blue',
                font: {
                size: 12,
                family: 'tahoma',
                weight: 'normal',
                style: 'italic'
                },
                padding: {
                bottom: 10
                }
            }
        },
    },
    data: {
        labels: steps,
        datasets: [{
                label: 'Dev ler',
                data: devLer,
            },
            {
                label: 'Best ler',
                data: bestLer
            },
            {
                label: 'Alpha Qubit Best Results',
                data: Array(devLer.length).fill(data.alphaBest)
            }
        ]
    }
    });
}


