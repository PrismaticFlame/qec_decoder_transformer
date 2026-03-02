# Structure of AlphaQubit

We will be trying to recreate AlphaQubit's model structure and here are the details.

## Sycamore vs. Scaling

We will construct the model to have slightly different hyperparameters according to Google's paper, and during training we will specify what hyperparameters to use based on the training we are doing (Sycamore vs. Scaling, pretraining vs. fine-tuning). The values do not differ greatly, and the structure of the model between the Sycamore training and Scaling training is the same. 

**Takeaway**: Structure is static, hyperparameters change based on which training we are doing.

## Dilations

There is a learning rate change based on the code distance. This affects the learning rate (obviously) and the dilations in the convolution layers. This needs to be taken into account during the intake of the data.

# Model Details
> **About Dashes**
>
> Where there are "-", that means there is nothing there for that hyperparameter.

## Hyperparameters

| Module | Hyperparameter | | Value | |
| -------------- | ------------ | ---------- | ------| ---- |
| | | Sycamore | | Scaling |
| Optimizer | Method | Lamb | | Lion |
| | Weight Decay | $10^{-5}$ | | $10^{-7}$ |
| | Fine-tuning weight decay | 0.08 | | - |
| | beta2 | | 0.95 | |
| | Inital Batch Size | | 256 | |
| | Final Batch Size | | 1024 | |
| | Batch size change step | $4 \times 10^6$ | | $8 \times 10^5$ |
| | Learning rate decay factor | | 0.7 | |
| | Learning rate decay steps $\times 10^5$ | {0.8, 2, 4, 10, 20} | | {4, 8, 16} |
| | Next stabilizer prediction loss weight | 0.01 | | 0.02 |
| | Parameter exponential moving average constant | | 0.0001 | |
| ------------- | ------------- | ------------ | ----------- | ------------ |
| Feature Embedding | ResNet layers | | 2 | |
| ---------------| -------------- | --------- | ------- | ---------|
| Syndrome Transformer | Layers | | 3 | |
| | Dimensions per stabilizer | 320 | | 256 |
| | Heads | | 4 | |
| | Key size | | 32 | |
| | Convolution layers | | 3 | |
| | Convolution dimensions | 160 | | 128 |
| | Dense block dimension widening | | 5 | |
| Attention bias | Dimensions | | 48 | |
| | Residual layers | | 8 | |
| | Indicator features | | 7 | |
| Readout ResNet | Layers | | 16 | |
| | Dimensions | 64 | | 48 |

## Dilations

| Code distance | Dilations | | Learning rate | |
| ------------- | --------- | - | --------- | - |
| | | Sycamore | | Scaling |
| 3 | 1, 1, 1| $3.46 \times 10^{-4}$ | | $1.3 \times 10^{-4}$ |
| 5 | 1, 1, 2 | $2.45 \times 10^{-4}$ | | $1.15 \times 10^{-4}$ |
| 7 | 1, 2, 4 | - | | $1 \times 10^{-4}$ |
| 9 | 1, 2, 4 | - | | $7 \times 10^{-5}$ |
| 11 | 1, 2, 4 | - | | $5 \times 10^{-5}$ |

> The dilations of the $3 \times 3$ convolutions in each syndrome transformer layer and the experiment learning rates are determined by the code-distance of the experiment.

## Ensembling

Google ensembles models like so:

"We train multiple models with identifcal hyperparameters, but different random seeds leading to different parameter initializations and training on different sequences of examples." 

We will attempt to ensemble, but we will first train two models, one for X basis and one for Z basis, and confirm their capability before attempting to ensemble. 

Some details for ensembling:
> 5 different seeds for scaling, 20 for sycamore (we will only be doing ensembling for scaling)
>
> The average the logits of the models, computing a geometric mean of the predicted error probabilites.
>
> They ensemble all models, not just "best" models.

## Loss

The loss is cross-entropy objectives with binary targets (#TODO). For scaling experiments specifically, all losses are averaged.

As auxiliary loss they use next stabilizer prediction cross-entropy loss averaged across all cycles and all stabilizers and then down weighted relative to the error prediction loss.

Stochastic gradient descent is used to minimize loss. For scaling, we will use the Lion optimizer like Google did.

Weight decay is used everywhere (L2 norm on non-bias parameters), either relative or zero for pretraining, or relative to pretrained parameters for fine-tuning, then using a stronger weight decay.

The learning rate is a piecewise constant after an initial linear warm-up of 10,000 steps, with reductions by a factor of 0.7 at specified steps above. 

## Cross-validation

The sycamore memory experiment uses the two disjoint odd-and-even indexed experiments to perform 2-fold cross-validation.

# Training Procedure

## Pre-training and fine-tuning

The models they used are trained in two phases: pre-training and fine-tuning. The pre-training phase trains on up to 2.5 billion samples from a DEM fitted on one half of the Sycamore memory experiment data (to allow 2-fold cross-validation on the other half). This allows the model to see lots of samples and learn generalizability.

The fine-tuning stage then uses the data from the Sycamore memory experiment to 