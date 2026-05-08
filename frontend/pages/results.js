import Data from "../components/Data.js"

// Create variables
var data = new Data("../data/results.csv", "Z", 3, 25, 50000)
var graphData = null
const ctx = document.getElementById('transformer-chart');

window.onload = function() {
    // Generate graph data
    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

// Takes in csv data to be passed into the graph
function makeGraphData(myData) {
    // Get arrays from the csv columns
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
    
    // Make the table and chart from the generated graph data
    makeTable("transformer-table", graphData)
    makeChart()
}

// Generate the Row array for creating tables
function colsToRows(myDict) {
    // tableData is a array of dictionaries
    var tableData = []

    for (var i = 0; i <= myDict.labels.length; i++) {
        // Each row has key value pairs for every column
        var row = {}

        row.id = i // id references each row id

        if (myDict.labels[i] == undefined) { // If the row is empty, you reached the end of the array
            break
        }

        // The dictionary generated for Chart.js does not have myDict.label, so the index must refer to rounds
        if (myDict.label == undefined) { 
            row["rounds"] = myDict.labels[i]
        } else { // Dictionaries from comparisons.js use label as the first column key
            row[myDict.label] = myDict.labels[i]
        }
        
        // Loop through all remaining columns to add to the row
        for (var j = 0; j < myDict.datasets.length; j++) {
            row[myDict.datasets[j].label] = myDict.datasets[j].data[i]
        }
        
        tableData.push(row) // Add the row at the end
        
    }
    return tableData
}

// Generate the column array for creating tables
function defineDataCols(myDict) {
    // cols is an array of dictionaries that hold column specifications
    var cols = []

    
    // The dictionary generated for Chart.js does not have myDict.label, so the index must refer to rounds
    if (myDict.label == undefined) {
        var row = {title: "Rounds", field: "rounds", minWidth: 75}
        cols.push(row)
    } else { // Dictionaries from comparisons.js use label as the first column key
        var row = {title: myDict.label, field: myDict.label}
        cols.push(row)
    }

    // Loop through remaining columns to be pushed
    for (var i = 0; i < myDict.datasets.length; i++) {
        if (myDict.datasets[i] == undefined) { // If the row is empty, you reached the end of the list
            break
        }
        var row = {title: myDict.datasets[i].label, field: myDict.datasets[i].label, minWidth: 100}
        cols.push(row)
    }

    return cols
}

function makeTable(tableId, graphData) {
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
    } 
    if (tableParams.data.length > 7) {
        tableParams['height'] = 205
    }
    new Tabulator(`#${tableId}`, tableParams);
}

// Generate a chart using Chart.js
function makeChart() {
    new Chart(ctx, {
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
                x: { // Display label on X axis
                    display: true,
                    title: {
                        display: true,
                        text: "Rounds"
                    }
                },
                y: { // Display label on Y axis and show table as logarithmic
                    display: true,
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: "Logical Error Rate (LER)"
                    }
                }
            }
        },
        data: graphData // graphData is generated previously in makeGraphData()
    });
}