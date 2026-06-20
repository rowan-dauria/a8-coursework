# I.3 Training Narrative
This document explains the results we saw with the untrained model, the results of the model trained on the baseline config, and how we interpreted the results of the baseline config training to try new training params.

## Baseline training config results
- The model achieved a very poor answer accuracy on the baseline run.
- We are restricted to making prinicpled, controlled improvements to the baseline training configuration, so we tried a lot of runs, each changing only one thing at a time. We then evaluated the checkpoints on a fixed 64 item subset of the test data to assess accuracy quickly. Only one variable was changed per training run.

## K=8 as principle change
- Increasing K was the most obvious change from a theory perspective
- Did show improved results, but not much above the untrained model (reference evaluation table)
- Noticed that the model was learning to format well, but answer accuracy was not improving much.