# Running the Streamlit App Locally

## Prerequisites

Make sure you have **Docker** and **VS Code** installed on your system.

---

## 1. CD into the frontend folder

```bash
cd .\frontend\
```

---

## 2. Start the container

```bash
docker-compose up -d
```

If this does not work, make sure Docker Desktop is running and not paused on your device

---

## 3. Run the website

To run the website, open http://localhost:8080/

---

## 4. Stoping the server

To stop the server, press:

```bash
docker-compose down
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

After starting the server, the following websites are now functional:

* **http://localhost:8080/** – Accessible only from your machine