# Quantum Error Correction Decoder with Transformer Neural Network - Replicating AlphaQubit and Improving it to Other Quantum Codes

Creating a Transformer Neural Network that will function as AlphaQubit, Google's Transformer Quantum Error Correction Decoder.

# Table of Contents:
1. [Setup](#setup)
2. [How to use Docker](#how-to-use-docker)
3. [Development](#development)
4. [Team Members](#team-members)

# Setup

**1. Clone the repository**

```sh
    git clone https://github.com/PrismaticFlame/qec_decoder_transformer.git
    cd qec_decoder_transformer
```

**2. Build Docker image and spin up container:**

```sh
    docker-compose build
    docker-compose run --rm transformer-gpu bash
```
You may have to use the `transformer-dev` version if your GPU is screaming. This will cause every next step to take much longer.

`docker-compose build` took quite a while on my laptop, around 10-12 minutes. Prepare for this to take a while. Thankfully this huge, long process only happens once. If any changes happen in `Dockerfile`, `requirements.txt`, or `docker-compose.yml`, run this command again and it will take under a minute (hopefully).

**3. Generate Data**

```sh
    cd src/trans6_alphaqubit
    python gen_basis_data.py --distances 3 --bases x z
```

**4. Train Model**
```sh
    python run_train.py --basis z --distance 3 --rounds 6
```

After the model has trained (depending on device, it could take between 10 hours and 30 hours) you can evaluate the model to see its performance!

**5. Evaluate the model**

There are several scripts available depending on what you want to check:

**Quick combined score** — loads your X and Z models, evaluates each on its own basis, and reports a single averaged LER (the headline number from the AlphaQubit paper):
```sh
    python eval_combine.py --distance 3 \
        --x_checkpoint checkpoints/x_d3_r6.pth \
        --z_checkpoint checkpoints/z_d3_r6.pth
```

**Generalization + MWPM comparison** — tests how the model performs at round counts it was never trained on, and compares directly against the MWPM classical decoder. Produces plots saved to `eval_results/`:
```sh
    python eval_generalize.py \
        --checkpoint checkpoints/x_d3_r6.pth checkpoints/z_d3_r6.pth \
        --rounds 6 9 11 13 15 17 19 21 23 25
```

**Error suppression factor (Lambda)** — computes the AlphaQubit Lambda metric, which measures how well the decoder improves as the code distance increases. Higher is better:
```sh
    python eval_lambda.py \
        --checkpoint checkpoints/z_d3_r6.pth --distances 3 5 --mwpm
```

**MWPM baseline only** — runs the classical MWPM decoder on its own, useful for establishing a baseline without loading any trained model:
```sh
    python eval_mwpm.py --distances 3 --basis z --rounds_list 6 9 12 --shots 100000
```

**Performance across noise levels** — trains and evaluates at multiple physical error rates to see how both the transformer and MWPM hold up as noise increases:
```sh
    python sweep_physical_error_rates.py --basis z --shots 50000 --num_steps 30000
```

> [!NOTE]
> The recommended order after training is: `eval_generalize.py` first (broad overview), then `eval_combine.py` (headline LER number), then `eval_lambda.py` if you want the paper-style scaling metric.

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


> The `docker-compose run --rm transformer bash` command is a specific command to just this project, and you can find the declaration of this command in `docker-compose.yml` paired with the final line of the `Dockerfile`.

> For the next two commands, there are other tags that can be specified to customize the generated data more as well as how long the model will train for. To recreate our results, create 1M shots (--shots 1_000_000) and then train for 50,000 steps (--num_steps 50_000)

**3. Exit image**

```sh
    exit
```

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
git push -u [issue/issue_name]
```

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

# ARC Setup

In the root of repo is a file named `total_setup.py` which will be an easy setup to getting things working on the ARC. The file requires 2 things to run:

1. Automatic SSH Login to the ARC setup
2. Data (by Tzu-Chen) in the Data directory.

If these two conditions are satisfied, then you are all set to go!

## What it does

1. This script will check your connection to the ARC, and once that is confirmed will find what files you need on the ARC based on a preset list of files. Depending on files are found on the ARC, or no files found at all, it copy all trans7_alphaqubit files to the ARC. 

2. It will also copy all data to the ARC that was created by Tzu-Chen. If the files have not be placed correctly or been randomly sampled, it will handle that as well.

3. Finally, if there are already files on the ARC, specific files will be prioritized and kept so they are not overwritten or deleted. These files include logs (.err, .out, gpu_usage_*.out) and checkpoints of previously trained models.

## Commands

The command should be simple as

```Python
python total_setup.py --vtusername USERNAME --arc-host tinkercliffs2.arc.vt.edu
```

If you are scared and don't want to do anything before knowing what will happen, you can add the `--dry-run` flag at the end of the command to see what will happen instead of it actually happening.

In the event that there is a major change to the model or structure of the trans7_alphqbuit directory, Chris will include in the commit the words "UPDATE REQUIRED", to which you will include the `--update` flag in the command, so it will look like:

```Python
python total_setup.py --vtusername USERNAME --arc-host tinkercliffs2.arc.vt.edu --update
```

# Team Members
- Tzu-Chen Chiu
- Arjun Sivakumar (Git push check)
- Mara Schimmel
- Christopher Williams