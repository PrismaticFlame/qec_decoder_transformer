# Run commands

Not a very robust edition. Just a few things.

To recreate what I did, use these commands:

> [!NOTE]
> First, get to the trans3_alphaqubit directory

```
python benchmark_distances.py --all
```

This command will just run everything - generate data, train 9 different transformers, and then create graphs based on results from the transformers.

```
python benchmark_distances.py --generate
```

Just generates data.

```
python benchmark_distances.py --train
```

Trains 9 transformers on generated data.

```
python benchmark_distances.py --plot
```

Plots performances of 9 different transformers.