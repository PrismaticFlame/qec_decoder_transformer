# Quantum Error Correction Decoder with Transformer Neural Network - Replicating AlphaQubit and Improving it to Other Quantum Codes

Creating a Transformer Neural Network that will function as AlphaQubit, Google's Transformer Quantum Error Correction Decoder.

# Table of Contents:
1. [How to use Docker](#how-to-use-docker)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Development](#development)
5. [Team Members](#team-members)

# How to use Docker
If you do not have Docker installed on your computer, you will need to install it. I recommend getting Docker Desktop instead of just the engine, but up to you.

Also, you will need WSL (Window Subsystem for Linux) and have it updated for this to work, so go through those steps on your own.

They are a little annoying, but I belive in you. Docker needs WSL to work, so do that before any of the steps below.

1. Build image
> [!NOTE]
> DON'T DO THIS YET. READ FIRST.
```bash
    docker-compose build
```

2. Enter image
```bash
    docker-compose run --rm transformer bash
```
> [!NOTE]
> This specific command is just for this project, and you can find the declaration of this command in `docker-compose.yml` paired with the final line of the `Dockerfile`.

3. Run test script
```bash
    python src/hello_world.py
```

4. Exit image
```bash
    exit
```

# Setup

1. Clone the repository
```bash
    git clone
    cd transformer-project
```

2. Build Docker container:
```bash
    docker-compose build
```
> [!NOTE]
> This took quite a while on my laptop, around 10-12 minutes. Prepare for this to take a while. Thankfully this huge, long process only happens once. If any changes happen in the Dockerfile, requirements.txt, or docker-compose.yml, run this command again and it will take under a minute (hopefully).

3. Run container:
```bash
    docker-compose run --rm transformer bash
```

4. Inside container, test the setup:
```bash
    python src/test_setup.py
```

# Project Structure

> [!NOTE]
> This is subject to change. The structure will likely not reflect the current structure of the project because of consistent changes.

- `src/` - Source code
  - `transformer.py` - Transformer model implementation
  - `train.py` - Training script
  - `utils.py` - Utility functions
- `data/` - Datasets (not committed to Git)
- `models/` - Saved model checkpoints (not committed to Git)
- `notebooks/` - Jupyter notebooks for experiments

# Development

> [!NOTE]
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
