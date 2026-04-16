import Data from "../components/Data.js"
import getSummary from "../components/comparisons.js"

var basis = "X"
var version = "All"
const x_data = new Data("../data/x_d3_r6_eval.csv", "X", 3, 6, 500)
const z_data = new Data("../data/z_d3_r6_eval.csv", "Z", 3, 6, 50000)

const dataDict = {
    'All': {'X': x_data, 'Z': z_data}, 
    'V3': {'X': x_data, 'Z': z_data},
    'V5': {'X': x_data, 'Z': z_data},
    'V6': {'X': x_data, 'Z': z_data},
}

var data = dataDict[version][basis]

const ctx = document.getElementById('transformer-chart');
var graphData = null
var myChart = null


window.onload = function() {
    document.getElementById("x-button").disabled = true
    document.getElementById("all-button").disabled = true
    document.getElementById("summary").innerHTML = getSummary(version)
    document.getElementById("x-button").addEventListener("click", swapBasis)
    document.getElementById("z-button").addEventListener("click", swapBasis)
    document.getElementById("all-button").addEventListener("click", function() {swapVersion("All")})
    document.getElementById("v3-button").addEventListener("click", function() {swapVersion("V3")})
    document.getElementById("v5-button").addEventListener("click", function() {swapVersion("V5")})
    document.getElementById("v6-button").addEventListener("click", function() {swapVersion("V6")})
    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

function swapVersion(newVersion) {
    document.getElementById(`${newVersion.toLowerCase()}-button`).disabled = true
    version = newVersion
    switch (newVersion) {
        case "All":
            document.getElementById("v3-button").disabled = false
            document.getElementById("v5-button").disabled = false
            document.getElementById("v6-button").disabled = false
            break
        case "V3":
            document.getElementById("all-button").disabled = false
            document.getElementById("v5-button").disabled = false
            document.getElementById("v6-button").disabled = false
            break
        case "V5":
            document.getElementById("v3-button").disabled = false
            document.getElementById("all-button").disabled = false
            document.getElementById("v6-button").disabled = false
            break
        case "V6":
            document.getElementById("v3-button").disabled = false
            document.getElementById("v5-button").disabled = false
            document.getElementById("all-button").disabled = false
    }

    var data = dataDict[version][basis]

    d3.csv(data.csv, (e) => {makeGraphData(e)})
    // newVersion = newVersion.charAt(0).toUpperCase() + newVersion.substring(1)
    document.getElementById("summary-version").innerHTML = newVersion
    document.getElementById("summary").innerHTML = getSummary(newVersion)
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
    var data = dataDict[version][basis]

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

    makeTable()
    makeChart()
}

function makeTable() {
    var tableEle = document.getElementById("transformer-table")
    tableEle.innerHTML = ""

    // Header row
    var tHead = document.createElement("thead")
    var headers = document.createElement("tr")
    headers.innerHTML = "<td>Steps</td>"
    for (var i = 0; i < graphData.datasets.length; i++) {
        headers.innerHTML += `<td>${graphData.datasets[i].label}</td>`
    }
    tHead.appendChild(headers)
    tableEle.appendChild(tHead)

    // Data rows
    var tBody = document.createElement("tbody")
    for (var i = 0; i < graphData.labels.length; i++) {
        var dataRow = document.createElement("tr")
        dataRow.innerHTML = `<td>${graphData.labels[i]}</td>`
        for (var j = 0; j < graphData.datasets.length; j++) {
            const dataValue = graphData.datasets[j].data[i]
            dataRow.innerHTML += `<td>${Number(dataValue).toFixed(2)}</td>`
        }
        tBody.appendChild(dataRow)
    }

    tableEle.appendChild(tBody)
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


