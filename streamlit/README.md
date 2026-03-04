# Running the Streamlit App Locally

## Prerequisites

Make sure you have **Python** and **VS Code** installed on your system.

---

## 1. Install Dependencies

> You only need to do this once.

Open a terminal and run:

```bash
pip install streamlit
pip install pandas
pip install plotly
```

This installs Streamlit, pandas, and plotly as Python packages. 

---

## 2. Test Streamlit Installation

Run the example Streamlit application:

```bash
streamlit hello
```

If the command above fails, try:

```bash
python -m streamlit hello
```

> Depending on your system, you may need to replace `python` with `python3` or another Python executable.

If successful, a demo Streamlit app will open in your browser using `localhost`.

To stop the server, press:

```bash
Ctrl + C
```

---

## 3. Run This Project

### Step 1: Navigate to the Streamlit Directory

Make sure your terminal is inside the `streamlit` folder:

```bash
cd streamlit
```

### Step 2: Start the App

Run:

```bash
python -m streamlit run Home.py
```

To stop the server, press:

```bash
Ctrl + C
```

---

## Accessing the App

After starting the server, the terminal will display two URLs:

* **Local URL** – Accessible only from your machine
* **Network URL** – Accessible by other devices on the same network