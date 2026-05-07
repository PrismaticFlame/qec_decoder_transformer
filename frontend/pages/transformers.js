import Data from "../components/Data.js"
import {getSummary, getOverview, getStructure, getArchitecture, getParameters, getDifferences} from "../components/comparisons.js"

// Define data objects
const v5_x_data = new Data("../data/v5_x_d3_r6_eval.csv", "X", 3, 6, 50000)
const v5_z_data = new Data("../data/v5_z_d3_r6_eval.csv", "Z", 3, 6, 50000)
const v6_x_data = new Data("../data/v6_x_d3_r6_eval.csv", "X", 3, 6, 50000)
const v6_z_data = new Data("../data/v6_z_d3_d5_r6_r10_eval.csv", "Z", 3, 6, 50000, 5, 10)
const all_z_data = new Data("../data/all_z_eval.csv", "Z", 3, 6, 50000, 5, 10)
const all_x_data = new Data("../data/all_x_eval.csv", "X", 3, 5, 50000)
// Placeholder data is used to prevent errors without showing actual data
const placeholder_data = new Data("../data/x_d3_r6_eval.csv", "X", 3, 6, 500)

// Set dictionary and global variables
const dataDict = {
    'All': {'X': all_x_data, 'Z': all_z_data}, 
    'V3': {'X': placeholder_data, 'Z': placeholder_data},
    'V5': {'X': v5_x_data, 'Z': v5_z_data},
    'V6': {'X': v6_x_data, 'Z': v6_z_data},
    'V7': {'X': placeholder_data, 'Z': placeholder_data},
}

var basis = "X"
var version = "All"
var data = dataDict[version][basis]
const ctx = document.getElementById('transformer-chart');
var graphData = null
var myChart = null
var tables = {}

window.onload = function() {
    // Set button event listeners
    document.getElementById("x-button").disabled = true
    document.getElementById("all-button").disabled = true
    document.getElementById("x-button").addEventListener("click", swapBasis)
    document.getElementById("z-button").addEventListener("click", swapBasis)
    document.getElementById("all-button").addEventListener("click", function() {swapVersion("All")})
    document.getElementById("v3-button").addEventListener("click", function() {swapVersion("V3")})
    document.getElementById("v5-button").addEventListener("click", function() {swapVersion("V5")})
    document.getElementById("v6-button").addEventListener("click", function() {swapVersion("V6")})
    document.getElementById("v7-button").addEventListener("click", function() {swapVersion("V7")})

    // Set initial version details and graphs
    makeDetails()
    d3.csv(data.csv, (e) => {makeGraphData(e)})

    // Set tab event listeners
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
    // Disable current version button
    document.getElementById(`${newVersion.toLowerCase()}-button`).disabled = true

    // Swap button classes and enable remaining buttons
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

    // Change data and details for new version
    data = dataDict[version][basis]
    d3.csv(data.csv, (e) => {makeGraphData(e)})
    makeDetails()
}

function swapBasis() {
    const xButton = document.getElementById("x-button")
    const zButton = document.getElementById("z-button")

    // Set basis buttons
    if (basis == "X") {
        basis = "Z"
        xButton.disabled = false
        xButton.classList.remove("clicked")
        zButton.disabled = true
        zButton.classList.add("clicked")
    } else { // basis == "Z"
        basis = "X"
        zButton.disabled = false
        zButton.classList.remove("clicked")
        xButton.disabled = true
        xButton.classList.add("clicked")
    }

    data = dataDict[version][basis]
    d3.csv(data.csv, (e) => {makeGraphData(e)})
}

// Takes in csv data to be passed into the graph
function makeGraphData(myData) {

    if (data.d2 == 0) { // If only one distance is present
        if (version == "All") { // If the current version is all
            // Get arrays from the csv columns
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
        } else { // If the current version is V3, V5, V6, or V7
            // Get arrays from the csv columns
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
        
    } else { // If two distances are present
        if (version == "All") { // If the current version is all
            // Get arrays from the csv columns
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
        } else { // If the current version is V3, V5, V6, or V7
            // Get arrays from the csv columns
            var steps = myData.map(function(d) {return d.step});
            var devLerD3 = myData.map(function(d) {return d.dev_ler_d3})
            var bestLerD3 = myData.map(function(d) {return d.best_ler_d3})
            var devLerD5 = myData.map(function(d) {return d.dev_ler_d5})
            var bestLerD5 = myData.map(function(d) {return d.best_ler_d5})

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

        // The dictionary generated for Chart.js does not have myDict.label, so the index must refer to steps
        if (myDict.label == undefined) { 
            row["steps"] = myDict.labels[i]
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

    
    // The dictionary generated for Chart.js does not have myDict.label, so the index must refer to steps
    if (myDict.label == undefined) {
        var row = {title: "Steps", field: "steps", minWidth: 75}
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

// Generates a new table at tableId using graphData
function makeTable(tableId, graphData) {
    
    // If the result is not null, the table must be destroyed to make a new table
    if (tables[tableId] != null) { 
        tables[tableId].destroy()
    }

    // Generate data rows and columns from passed in data
    var data = colsToRows(graphData)
    var cols = defineDataCols(graphData)

    // Set initial table parameters
    var tableParams = {
        data: data, // Assign data to table
        layout: "fitColumns", // Fit columns to width of table
        columnDefaults:{
            tooltip:true,
        },
        columns: cols,
    }

    // Only apply to the table displaying transformer data
    if (tableId == "transformer-table") {
        tableParams.layout = "fitDataTable" // Set the layout to DataTable so all values are properly displayed
        tableParams.columnDefaults.tooltip = false
    } 

    // When the # rows > 8, the table is too tall and must be scrollable
    if (tableParams.data.length > 7) { 
        tableParams['height'] = 205
    }

    // Assign table to tableId so it can be destroyed when the table needs to be replaced
    var table = new Tabulator(`#${tableId}`, tableParams);
    tables[tableId] = table
    
}

// Generate a chart using Chart.js
function makeChart() {
    
    // If the result is not null, the table must be destroyed to make a new table
    if (myChart != null) {
        myChart.destroy()
    }

    myChart = new Chart(ctx, {
        type: 'line',
        options: {
            plugins: {
                title: { // Customize title based on basis and distance
                    display: true,
                    text: `${data.basis} basis, Distance ${data.d}${(data.d2 != 0) ? " and " + data.d2.toString() : ""}`
                },
                subtitle: { // Customize subtitle based on shots and round number
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
                x: { // Display label on X axis
                    display: true,
                    title: {
                        display: true,
                        text: "Shots"
                    }
                },
                y: { // Display label on Y axis and show table as logarithmic
                    display: true,
                    type: 'logarithmic',
                    title: {
                        display: true,
                        text: "LER"
                    }
                }
            }
        },
        data: graphData // graphData is generated previously in makeGraphData()
    });
}

// Update the summary and tab details based on the current version
function makeDetails() {
    // Summary
    document.getElementById("summary-version").innerHTML = version
    document.getElementById("summary").innerHTML = getSummary(version)

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

    // Differences
    changeDifferences()

    // Flow
    changeFlow()
}

// Update Key Architectural Differences tab
function changeDifferences() {

    const differences = getDifferences(version) // getDifferences comes from comparisons.js
    
    // Changes contains all differences between models to be displayed
    var changes = ""
    switch (version) {
        case "V3": // Add Trans3 -> Trans5 comparison only
            changes = "<h3>Trans3 -> Trans5 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = ""
            break

        case "V5": // Add Trans3 -> Trans5, Trans5 -> Trans6 comparisons
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

        case "V6": // Add Trans5 -> Trans6, Trans6 -> Trans7 comparisons
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

        case "V7": // Add Trans6 -> Trans7 comparison and remaining gaps from V7 -> AlphaQuibit
            changes = "<h3>Trans6 -> Trans7 Changes</h3><ol>"
            for (var i = 0; i < differences[0].length; i++) {
                changes += `<li>${differences[0][i]}</li>`
            }
            changes += "</ol>"
            document.getElementById("gaps").innerHTML = "<h3>Trans7 vs AlphaQubit (Remaining Gaps)</h3><div id='gaps-table'></div>"
            makeTable("gaps-table", differences[1])
            break

        default: // Version == "All", add all comparisons and remaining gaps from V7 -> AlphaQuibit
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

// Update Data Flow tab
function changeFlow() {
    // Switch to show only relevant data flow
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

        default: // version == "All"
            document.getElementById("trans3-flow").style.display = ""
            document.getElementById("trans5-flow").style.display = ""
            document.getElementById("trans6-flow").style.display = ""
            document.getElementById("trans7-flow").style.display = ""
    }
}

// Open selected tab
function openTab(evt, tabName) {
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
  document.getElementById(tabName).style.display = "block";
  evt.currentTarget.className += " active";
}