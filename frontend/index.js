var testVal = false
function changeSpan() {
    if (testVal) {
        document.getElementById("test").innerHTML = "False"
        testVal = false
    } else {
        document.getElementById("test").innerHTML = "True"
        testVal = true
    }
}

console.log(testVal)
console.log("updated p2")

// API call to get data based on button