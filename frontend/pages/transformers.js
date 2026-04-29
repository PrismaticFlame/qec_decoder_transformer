import Data from "../components/Data.js"
import {getSummary, getOverview, getStructure, getArchitecture, getParameters, getDifferences} from "../components/comparisons.js"

var basis = "X"
var version = "All"
const v5_x_data = new Data("../data/v5_x_d3_r6_eval.csv", "X", 3, 6, 50000)
const v5_z_data = new Data("../data/v5_z_d3_r6_eval.csv", "Z", 3, 6, 50000)
const v6_x_data = new Data("../data/v6_x_d3_r6_eval.csv", "X", 3, 6, 50000)
const v6_z_data = new Data("../data/v6_z_d3_d5_r6_r10_eval.csv", "Z", 3, 6, 50000, 5, 10)
const x_data = new Data("../data/x_d3_r6_eval.csv", "X", 3, 6, 500)
const z_data = new Data("../data/z_d3_r6_eval.csv", "Z", 3, 6, 50000)
const all_z_data = new Data("../data/all_z_eval.csv", "Z", 3, 6, 50000, 5, 10)
const all_x_data = new Data("../data/all_x_eval.csv", "X", 3, 5, 50000)

const dataDict = {
    'All': {'X': all_x_data, 'Z': all_z_data}, 
    'V3': {'X': x_data, 'Z': z_data},
    'V5': {'X': v5_x_data, 'Z': v5_z_data},
    'V6': {'X': v6_x_data, 'Z': v6_z_data},
    'V7': {'X': x_data, 'Z': z_data},
}

var data = dataDict[version][basis]

const ctx = document.getElementById('transformer-chart');
var graphData = null
var myChart = null
var tables = {}

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
    document.getElementById("v7-button").addEventListener("click", function() {swapVersion("V7")})
    makeDetails()
    d3.csv(data.csv, (e) => {makeGraphData(e)})
    document.getElementById("overview").addEventListener("click", function(e) {openTab(e, "overview-div")})
    document.getElementById("overview").click()
    document.getElementById("structure").addEventListener("click", function(e) {openTab(e, "structure-div")})
    document.getElementById("architecture").addEventListener("click", function(e) {openTab(e, "architecture-div")})
    document.getElementById("parameters").addEventListener("click", function(e) {openTab(e, "parameters-div")})
    document.getElementById("differences").addEventListener("click", function(e) {openTab(e, "differences-div")})
    document.getElementById("flow").addEventListener("click", function(e) {openTab(e, "flow-div")})
    document.getElementById("evolution").addEventListener("click", function(e) {openTab(e, "evolution-div")})
}

function swapVersion(newVersion) {
    document.getElementById(`${newVersion.toLowerCase()}-button`).disabled = true
    version = newVersion
    var buttonAll = document.getElementById("all-button")
    var buttonV3 = document.getElementById("v3-button")
    var buttonV5 = document.getElementById("v5-button")
    var buttonV6 = document.getElementById("v6-button")
    var buttonV7 = document.getElementById("v7-button")
    switch (newVersion) {
        case "All":
            buttonV3.disabled = false
            buttonV3.classList.remove("clicked")
            buttonV5.disabled = false
            buttonV5.classList.remove("clicked")
            buttonV6.disabled = false
            buttonV6.classList.remove("clicked")
            buttonV7.disabled = false
            buttonV7.classList.remove("clicked")
            buttonAll.classList.add("clicked")
            document.getElementById("data").style.display = ""
            break
        case "V3":
            buttonAll.disabled = false
            buttonAll.classList.remove("clicked")
            buttonV5.disabled = false
            buttonV5.classList.remove("clicked")
            buttonV6.disabled = false
            buttonV6.classList.remove("clicked")
            buttonV7.disabled = false
            buttonV7.classList.remove("clicked")
            buttonV3.classList.add("clicked")
            document.getElementById("data").style.display = "none"
            break
        case "V5":
            buttonV3.disabled = false
            buttonV3.classList.remove("clicked")
            buttonAll.disabled = false
            buttonAll.classList.remove("clicked")
            buttonV6.disabled = false
            buttonV6.classList.remove("clicked")
            buttonV7.disabled = false
            buttonV7.classList.remove("clicked")
            buttonV5.classList.add("clicked")
            document.getElementById("data").style.display = ""
            break
        case "V6":
            buttonV3.disabled = false
            buttonV3.classList.remove("clicked")
            buttonV5.disabled = false
            buttonV5.classList.remove("clicked")
            buttonAll.disabled = false
            buttonAll.classList.remove("clicked")
            buttonV7.disabled = false
            buttonV7.classList.remove("clicked")
            buttonV6.classList.add("clicked")
            document.getElementById("data").style.display = ""
            break
        case "V7":
            buttonV3.disabled = false
            buttonV3.classList.remove("clicked")
            buttonV5.disabled = false
            buttonV5.classList.remove("clicked")
            buttonV6.disabled = false
            buttonV6.classList.remove("clicked")
            buttonAll.disabled = false
            buttonAll.classList.remove("clicked")
            buttonV7.classList.add("clicked")
            document.getElementById("data").style.display = "none"
            break
    }

    data = dataDict[version][basis]

    d3.csv(data.csv, (e) => {makeGraphData(e)})
    makeDetails()
    document.getElementById("summary-version").innerHTML = newVersion
    document.getElementById("summary").innerHTML = getSummary(newVersion)
}

function swapBasis() {
    const xButton = document.getElementById("x-button")
    const zButton = document.getElementById("z-button")
    if (basis == "X") {
        basis = "Z"
        xButton.disabled = false
        xButton.classList.remove("clicked")
        zButton.disabled = true
        zButton.classList.add("clicked")
    } else {
        basis = "X"
        zButton.disabled = false
        zButton.classList.remove("clicked")
        xButton.disabled = true
        xButton.classList.add("clicked")
    }
    data = dataDict[version][basis]

    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

function makeGraphData(myData) {
    var numCols = 4
    if (data.d2 == 0) {
        if (version == "All") {
            const steps = myData.map(function(d) {return d.step});
            const devLerV6 = myData.map(function(d) {return d.dev_ler_v6})
            const bestLerV6 = myData.map(function(d) {return d.best_ler_v6})
            const devLerV5 = myData.map(function(d) {return d.dev_ler_v5})
            const bestLerV5 = myData.map(function(d) {return d.best_ler_v5})

            graphData = {
                labels: steps,
                datasets: [
                    {
                        label: 'Dev ler (V5)',
                        data: devLerV5,
                    },
                    {
                        label: 'Best ler (V5)',
                        data: bestLerV5
                    },
                    {
                        label: 'Dev ler (V6)',
                        data: devLerV6,
                    },
                    {
                        label: 'Best ler (V6)',
                        data: bestLerV6
                    },
                    {
                        label: 'AlphaQubit Best Results',
                        data: Array(devLerV5.length).fill(data.alphaBest)
                    }
            ]}
        } else {
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
                        label: 'AlphaQubit Best Results',
                        data: Array(devLer.length).fill(data.alphaBest)
                    }
            ]}
        }
        
    } else {
        if (version == "All") {
            var steps = myData.map(function(d) {return d.step});
            var devLerD3V6 = myData.map(function(d) {return d.dev_ler_d3_v6})
            var bestLerD3V6 = myData.map(function(d) {return d.best_ler_d3_v6})
            var devLerD5V6 = myData.map(function(d) {return d.dev_ler_d5_v6})
            var bestLerD5V6 = myData.map(function(d) {return d.best_ler_d5_v6})
            var devLerD3V5 = myData.map(function(d) {return d.dev_ler_d3_v5})
            var bestLerD3V5 = myData.map(function(d) {return d.best_ler_d3_v5})

            graphData = {
                labels: steps,
                datasets: [
                    {
                        label: 'Dev ler (V5, D3)',
                        data: devLerD3V5,
                    },
                    {
                        label: 'Best ler (V5, D3)',
                        data: bestLerD3V5
                    },
                    {
                        label: 'Dev ler (V6, D3)',
                        data: devLerD3V6,
                    },
                    {
                        label: 'Best ler (V6, D3)',
                        data: bestLerD3V6
                    },
                    {
                        label: 'Dev ler (V6, D5)',
                        data: devLerD5V6,
                    },
                    {
                        label: 'Best ler (V6, D5)',
                        data: bestLerD5V6
                    },
                    {
                        label: 'AlphaQubit Best Results (D3)',
                        data: Array(devLerD3V5.length).fill(data.alphaBestD3)
                    },
                    {
                        label: 'AlphaQubit Best Results (D5)',
                        data: Array(devLerD3V5.length).fill(data.alphaBestD5)
                    }
            ]}
        } else {
            var steps = myData.map(function(d) {return d.step});
            var devLerD3 = myData.map(function(d) {return d.dev_ler_d3})
            var bestLerD3 = myData.map(function(d) {return d.best_ler_d3})
            var devLerD5 = myData.map(function(d) {return d.dev_ler_d5})
            var bestLerD5 = myData.map(function(d) {return d.best_ler_d5})

            console.log(data)

            graphData = {
                labels: steps,
                datasets: [
                    {
                        label: 'Dev ler (D3)',
                        data: devLerD3,
                    },
                    {
                        label: 'Best ler (D3)',
                        data: bestLerD3
                    },
                    {
                        label: 'Dev ler (D5)',
                        data: devLerD5,
                    },
                    {
                        label: 'Best ler (D5)',
                        data: bestLerD5
                    },
                    {
                        label: 'AlphaQubit Best Results (D3)',
                        data: Array(devLerD3.length).fill(data.alphaBestD3)
                    },
                    {
                        label: 'AlphaQubit Best Results (D5)',
                        data: Array(devLerD3.length).fill(data.alphaBestD5)
                    }
            ]}
        }
        
    }
    

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
            row["steps"] = myDict.labels[i]
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
        var row = {title: "Steps", field: "steps", minWidth: 75}
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
        tableParams.columnDefaults.tooltip = false
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
                    text: `${data.basis} basis, Distance ${data.d}${(data.d2 != 0) ? " and " + data.d2.toString() : ""}`
                },
                subtitle: {
                    display: true,
                    text: `${data.shots} shots, Rounds ${data.r}${(data.r2 != 0) ? " and " + data.r2.toString() : ""}`,
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
                        text: "Shots"
                    }
                },
                y: {
                    display: true,
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: "LER"
                    }
                }
            }
        },
        data: graphData
    });
}

function makeDetails() {
    // Overview
    makeTable("overview-table", getOverview(version))
    
    // Structure
    const structure = getStructure(version)
    makeTable("generation-table", structure[0])
    makeTable("format-table", structure[1])
    makeTable("input-table", structure[2])

    // Architecture
    const architecture = getArchitecture(version)
    makeTable("stabilizer-table", architecture[0])
    makeTable("syndrome-table", architecture[1])
    makeTable("attention-table", architecture[2])
    makeTable("readout-table", architecture[3])
    makeTable("auxiliary-table", architecture[4])

    // Parameters
    const parameters = getParameters(version)
    makeTable("model-table", parameters[0])
    makeTable("training-table", parameters[1])
    makeTable("convolution-table", parameters[2])

    changeDifferences()

    changeFlow()
}

function changeDifferences() {
    const differences = getDifferences(version)
    var changes = ""
    switch (version) {
        case "V3":
            changes = "<h3>Trans3 -> Trans5 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = ""
            break
        case "V5":
            changes = "<h3>Trans3 -> Trans5 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol><h3>Trans5 -> Trans6 Changes</h3><ol>"
            for (var i = 0; i < differences[1].length; i++) {
                changes += `<li>${differences[1][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = ""
            break
        case "V6":
            changes = "<h3>Trans5 -> Trans6 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol><h3>Trans6 -> Trans7 Changes</h3><ol>"
            for (var i = 0; i < differences[1].length; i++) {
                changes += `<li>${differences[1][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = ""
            break
        case "V7":
            changes = "<h3>Trans6 -> Trans7 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = "<h3>Trans7 vs AlphaQubit (Remaining Gaps)</h3><div id='gaps-table'></div>"
            makeTable("gaps-table", differences[1])
            break
        default:
            changes = "<h3>Trans3 -> Trans5 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol><h3>Trans5 -> Trans6 Changes</h3><ol>"
            for (var i = 0; i < differences[1].length; i++) {
                changes += `<li>${differences[1][i]}</li>`
            }
            changes += "</ol><h3>Trans6 -> Trans7 Changes</h3><ol>"
            for (var i = 0; i < differences[1].length; i++) {
                changes += `<li>${differences[1][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = "<h3>Trans7 vs AlphaQubit (Remaining Gaps)</h3><div id='gaps-table'></div>"
            makeTable("gaps-table", differences[3])
            break
    }
    document.getElementById("changes").innerHTML = changes
}

function changeFlow() {
    switch (version) {
        case "V3":
            document.getElementById("trans3-flow").style.display = ""
            document.getElementById("trans5-flow").style.display = "none"
            document.getElementById("trans6-flow").style.display = "none"
            document.getElementById("trans7-flow").style.display = "none"
            break
        case "V5":
            document.getElementById("trans3-flow").style.display = "none"
            document.getElementById("trans5-flow").style.display = ""
            document.getElementById("trans6-flow").style.display = "none"
            document.getElementById("trans7-flow").style.display = "none"
            break
        case "V6":
            document.getElementById("trans3-flow").style.display = "none"
            document.getElementById("trans5-flow").style.display = "none"
            document.getElementById("trans6-flow").style.display = ""
            document.getElementById("trans7-flow").style.display = "none"
            break
        case "V7":
            document.getElementById("trans3-flow").style.display = "none"
            document.getElementById("trans5-flow").style.display = "none"
            document.getElementById("trans6-flow").style.display = "none"
            document.getElementById("trans7-flow").style.display = ""
            break
        default: 
            document.getElementById("trans3-flow").style.display = ""
            document.getElementById("trans5-flow").style.display = ""
            document.getElementById("trans6-flow").style.display = ""
            document.getElementById("trans7-flow").style.display = ""
    }
}

function openTab(evt, cityName) {
  // Declare all variables
  var i, tabcontent, tablinks;

  // Get all elements with class="tabcontent" and hide them
  tabcontent = document.getElementsByClassName("tabcontent");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
  }

  // Get all elements with class="tablinks" and remove the class "active"
  tablinks = document.getElementsByClassName("tablinks");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }

  // Show the current tab, and add an "active" class to the button that opened the tab
  document.getElementById(cityName).style.display = "block";
  evt.currentTarget.className += " active";
}