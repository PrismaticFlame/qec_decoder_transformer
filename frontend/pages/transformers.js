console.log("This is a transformer updated chart")

const ctx = document.getElementById('myChart');

d3.csv("../data/x_d3_r6_eval.csv", makeChart)

function makeChart(myData) {
    console.log("this is running")
    var steps = myData.map(function(d) {return d.step});
    var devLer = myData.map(function(d) {return d.dev_ler})
    var bestLer = myData.map(function(d) {return d.best_ler})
    console.log(steps)
    console.log(devLer)

    new Chart(ctx, {
    type: 'line',
    data: {
        labels: steps,
        datasets: [{
                label: 'Dev ler',
                data: devLer,
            },
            {
                label: 'Best ler',
                data: bestLer
            }
        ]
    }
    });
}


