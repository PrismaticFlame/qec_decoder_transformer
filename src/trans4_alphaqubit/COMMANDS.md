# Commands for Transformer 4 - AlphaQubit

## 1. Generate Data

```
python src/prop_data_gen/gen_basis_data.py
```

This will generate data from the `prop_data_gen` and will create X basis and Z basis data, with 20k shots, at a physical error rate of 0.005, and will separate the data based on basis, distance, and rounds. If you want to customize it, you have some flags you can use in the CLI.


## 2. Train both bases for distance 'X'

```
python src/trans4_alphaqubit/run_train.py --bases z x --distance 3
```

This will find the data you generated in step 1, and then train 2 separate transformers on the basis data that was created for the relevant distance data generated. You can change the distance and it will find that code and train on that. It will output the performance of the two transformers in terms of LER.

## 3. Evaluate and Combine

```
python src/trans4_alphaqubit/eval_combine.py --distance 3
```

This takes the trained transformers you made in step 2, and combines them to see their combined LER.

## Options

There are a number of flags you can use for each file that has been created. These flags will change the behavior, and hopefully allow us the ability to find some good parameters for our transformer for different distances, and potentially use even more shots, rounds, and in the end codes.