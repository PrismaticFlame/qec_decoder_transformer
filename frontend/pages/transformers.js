import Data from "../components/Data.js"

var basis = "X"
const x_data = new Data("../data/x_d3_r6_eval.csv", "X", 3, 6, 500)
const z_data = new Data("../data/z_d3_r6_eval.csv", "Z", 3, 6, 50000)

const dataDict = {
    'All': {'X': x_data, 'Z': z_data}, 
    'v3': {'X': x_data, 'Z': z_data},
    'v5': {'X': x_data, 'Z': z_data},
    'v6': {'X': x_data, 'Z': z_data},
}

var data = dataDict['All'][basis]

const ctx = document.getElementById('transformerChart');
var graphData = null
var myChart = null


window.onload = function() {
    document.getElementById("x-button").disabled = true
    document.getElementById("x-button").addEventListener("click", swapBasis)
    document.getElementById("z-button").addEventListener("click", swapBasis)
    document.getElementById("all-button").addEventListener("click", function() {swapVersion("all")})
    document.getElementById("v3-button").addEventListener("click", function() {swapVersion("v3")})
    document.getElementById("v5-button").addEventListener("click", function() {swapVersion("v5")})
    document.getElementById("v6-button").addEventListener("click", function() {swapVersion("v6")})
    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

function swapVersion(version) {
    // fill in
    console.log(`Version: ${version}`)
}

function swapBasis() {
    const xButton = document.getElementById("x-button")
    const zButton = document.getElementById("z-button")
    if (basis == "X") {
        basis = "Z"
        xButton.disabled = false
        zButton.disabled = true
    } else {
        basis = "X"
        zButton.disabled = false
        xButton.disabled = true
    }
    var data = dataDict['All'][basis]

    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

function makeGraphData(myData) {
    var steps = myData.map(function(d) {return d.step});
    var devLer = myData.map(function(d) {return d.dev_ler})
    var bestLer = myData.map(function(d) {return d.best_ler})

    graphData = {
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
    ]}

    makeChart()

    
}

function makeChart() {
    if (myChart != null) {
        myChart.destroy()
    }
    myChart = new Chart(ctx, {
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
        data: graphData
    });
}


