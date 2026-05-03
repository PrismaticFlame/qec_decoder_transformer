import Data from "../components/Data.js"

var data = new Data("../data/results.csv", "Z", 3, 25, 50000)
var graphData = null
var tables = {}
var myChart = null
const ctx = document.getElementById('transformer-chart');

window.onload = function() {
    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

function makeGraphData(myData) {
    const avgLer = myData.map(function(d) {return d.avg_ler})
    const avgMwpm = myData.map(function(d) {return d.avg_mwpm})
    const rounds = myData.map(function(d) {return d.rounds})

    graphData = {
        labels: rounds,
        datasets: [{
                label: 'Transformer (avg X + Z)',
                data: avgLer,
            },
            {
                label: 'MWPM (avg X + Z)',
                data: avgMwpm
            }
    ]}
    

    makeTable("transformer-table", graphData)
    makeChart()
}

function colsToRows(myDict) {
    var tableData = []
    for (var i = 0; i <= myDict.labels.length; i++) {
        var row = {}
        row.id = i
        if (myDict.labels[i] == undefined) {
            break
        }
        if (myDict.label == undefined) {
            row["rounds"] = myDict.labels[i]
        } else {
            row[myDict.label] = myDict.labels[i]
        }
        
        for (var j = 0; j < myDict.datasets.length; j++) {
            row[myDict.datasets[j].label] = myDict.datasets[j].data[i]
        }
        tableData.push(row)
        
    }
    return tableData
}

function defineDataCols(myDict) {
    var cols = []
    if (myDict.label == undefined) {
        var row = {title: "Rounds", field: "rounds", minWidth: 75}
        cols.push(row)
    } else {
        var row = {title: myDict.label, field: myDict.label}
        cols.push(row)
    }
    for (var i = 0; i < myDict.datasets.length; i++) {
        if (myDict.datasets[i] == undefined) {
            break
        }
        var row = {title: myDict.datasets[i].label, field: myDict.datasets[i].label, minWidth: 100}
        cols.push(row)
    }
    return cols
}

function makeTable(tableId, graphData) {
    if (tables[tableId] != null) {
        tables[tableId].destroy()
    }
    var data = colsToRows(graphData)
    var cols = defineDataCols(graphData)
    var tableParams = {
        data: data, //assign data to table
        layout:"fitColumns", //fit columns to width of table (optional)
        columnDefaults:{
            tooltip:true,
        },
        columns: cols,
    }
    if (tableId == "transformer-table") {
        tableParams.layout = "fitDataTable"
        document.getElementById(tableId).style.removeProperty("height")
    } 
    if (tableParams.data.length > 7) {
        tableParams['height'] = 205
    }
    var table = new Tabulator(`#${tableId}`, tableParams);
    tables[tableId] = table
    
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
                    text: `Transformer vs MWPM: Round-Count Generalization (d=3, p=0.005), Version 6`
                },
                subtitle: {
                    display: true,
                    text: `Trained on X and Z data, using ${data.r} rounds`,
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
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: "Rounds"
                    }
                },
                y: {
                    display: true,
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: "Logical Error Rate (LER)"
                    }
                }
            }
        },
        data: graphData
    });
}