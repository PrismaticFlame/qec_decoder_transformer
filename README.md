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

```sh
    docker-compose build
```

**2. Enter image**

```sh
    docker-compose run --rm transformer bash
```

> [!tip]
> The `docker-compose run --rm transformer bash` command is a specific command to just this project, and you can find the declaration of this command in `docker-compose.yml` paired with the final line of the `Dockerfile`.

**3. Run test script**

```sh
    python src/hello_world.py
```

**4. Exit image**

```sh
    exit
```

# Setup

**1. Clone the repository**

```sh
    git clone https://github.com/PrismaticFlame/qec_decoder_transformer.git
    cd transformer-project
```

**2. Build Docker container:**

```sh
    docker-compose build
```

> [!NOTE]
> `docker-compose build` took quite a while on my laptop, around 10-12 minutes. Prepare for this to take a while. Thankfully this huge, long process only happens once. If any changes happen in `Dockerfile`, `requirements.txt`, or `docker-compose.yml`, run this command again and it will take under a minute (hopefully).

**3. Run container:**

```sh
    docker-compose run --rm transformer bash
```

**4. Inside container, test to make sure it's working:**

```sh
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
  - `hello_world.py` - Hello World file to test in Docker
- `data/` - Datasets (not committed to Git)
- `models/` - Saved model checkpoints (not committed to Git)
- `notebooks/` - Jupyter notebooks for experiments
- `stim_files/` - Stim files for creating reptition codes, surface codes (maybe?)

# Development

## Branches

When committing to the project, we will use branches for different issues. The naming convention for how we want to do this can be discussed and changed, but for the first while let's stick with this:
- "Stim/..." - relating to Stim files, data, and anything stim related
- "ML/..." - related to our transformer
- Anything else we want, we can add.

The idea would be that we commit to our new branch relating to whatever issue we are dealing with, and a somewhat short but descriptive branch title. For example:
- branch Stim/data_generation
- branch ML/weight_testing

This way we can keep things organized. Also, I am still new to this and how to properly utilize branches, so we will figure it out together!

As for actually comitting to branches, the command is:

```sh
git checkout -b [issue/issue_name]
```

This command will both create a new branch, as well as move to that new branch. Then you follow the normal commit procedure!

```sh
git add .  # add all changed files, you can specify certain files if you like
git commit -m "Message"
```

However, now that we have a new branch, we must set the new branch as the remote upstream like this:

```sh
git push --set-upstream origin ML/brt
```

This will set the new branch and set it as the upstream, as well as push to that branch in one command.

If there are any pressing issues, we can just reach out to each other and deal with it either individually or hop in a call/meet in person. Otherwise, we can create issues and resolve them as we go, and I will document that process more here in the future once we learn more about it!

## Jupyter Notebooks

Depending on the work you are doing, there are a few options that are available to us on how to run Jupyter Notebooks. For now, I am going to list the steps that are best taken when developing/running Jupyter notebooks right now. 

> [!NOTE]
> Small aside: In the .gitignore there is a line the removes all notebooks in the `notebooks/` directory, but I have uncommented that for now. If there comes a point in the future when we are rampantly creating and using notebooks left, right, and center, then we can untrack those notebooks. Until then, notebooks will be put on the repo.

Here's the main idea: We need to start the container, then within the container start a Jupyter Notebook server. After doing that, we can access our Jupyter Notebooks on a local server that will be on your machine. Here's the steps.

1. Start the container
> [!IMPORTANT]
> This is different than how we have been running the container before! This command starts the Docker container and just keeps it running, it doesn't actually give a bash into the container like with the other commands.

```sh
docker-compose up -d
```

2. Start Jupyter Notebook server:

```sh
docker-compose exec transformer jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
```

This starts the server that you will access! However, what the h*ll do you do now? Well, you'll head to your browser and access the server there at this URL: `http://localhost:8888`.
All things you do in the Jupyter Notebook in the server will be reflect in your files on your puter, so don't worry about syncing between the server and your machine.

# Team Members
- Tzu-Chen Chiu
- Arjun Sivakumar (Git push check)
- Mara Schimmel
- Christopher Williams