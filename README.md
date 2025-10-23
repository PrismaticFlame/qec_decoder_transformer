# Quantum Error Correction Decoder with Transformer Neural Network - Replicating AlphaQubit and Improving it to Other Quantum Codes

Creating a Transformer Neural Network that will function as AlphaQubit, Google's Transformer Quantum Error Correction Decoder.

# Table of Contents:
1. [How to use Docker](#how-to-use-docker)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Development](#development)
5. [Team Members](#team-members)

# How to use Docker
If you do not have Docker installed on your computer, you will need to install it. I recommend getting [Docker Desktop](https://www.docker.com/products/docker-desktop/) instead of just the engine, but up to you.

Also, you will need [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) (Window Subsystem for Linux) and have it updated for this to work, so go through those steps on your own.

They are a little annoying, but I belive in you. Docker needs WSL to work, so **do that before any of the steps below**.

These steps are what you will do in the root directory of the project after cloning. To do this you will need to have Docker running and then you would generally use these commands to create, spin up, enter, and exit the Docker image. 

**1. Build image**
> [!warning]
> DON'T DO THIS YET. READ FIRST.
```bash
    docker-compose build
```

**2. Enter image**
```bash
    docker-compose run --rm transformer bash
```
> [!tip]
> The `docker-compose run --rm transformer bash` command is a specific command to just this project, and you can find the declaration of this command in `docker-compose.yml` paired with the final line of the `Dockerfile`.

**3. Run test script**
```bash
    python src/hello_world.py
```

**4. Exit image**
```bash
    exit
```

# Setup

**1. Clone the repository**
```bash
    git clone
    cd transformer-project
```

**2. Build Docker container:**
```bash
    docker-compose build
```
> [!NOTE]
> `docker-compose build` took quite a while on my laptop, around 10-12 minutes. Prepare for this to take a while. Thankfully this huge, long process only happens once. If any changes happen in `Dockerfile`, `requirements.txt`, or `docker-compose.yml`, run this command again and it will take under a minute (hopefully).

**3. Run container:**
```bash
    docker-compose run --rm transformer bash
```

**4. Inside container, test to make sure it's working:**
```bash
    python src/hello_world.py
```
> [!note]
> `python src/hello_world.py` is the general form of how we will run our scripts. For the majority of the project, we will be using Python so we will be mainly using the `python` command, followed by where the script is.

# Project Structure

> [!warning]
> This is subject to change. The structure will likely not reflect the current structure of the project because of consistent changes.

- `src/` - Source code
  - `transformer.py` - Transformer model implementation
  - `train.py` - Training script
  - `utils.py` - Utility functions
- `data/` - Datasets (not committed to Git)
- `models/` - Saved model checkpoints (not committed to Git)
- `notebooks/` - Jupyter notebooks for experiments

# Development

> [!note]
> I will get around to making this section make more sense. For now, ignore it.

Start Jupyter notebook:
```bash
    docker-compose run --rm -p 8888:8888 transformer \
    jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

# Team Members
- Tzu-Chen Chiu
- Arjun Sivakumar
- Mara Schimmel
- Christopher Williams
