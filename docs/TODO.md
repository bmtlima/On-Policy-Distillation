1. Load datasets, parse answers
2. Take like 10 prompts and run them on both student and teacher models
    2.1. make sure formatting is about the same
3. Run on entire dataset. Will be used as baseline
4. 



stretch:
1. comparing qwen to qwq-32B (latter really good from RSR paper)
2. Compare results to standard SFT through Unsloth (no need to tweak hyperparameters, just get it implemented). This one is a bigger lift.
3. Show how much you can gain from training on one task repeatedly (as explored in the Thinking Machines blog).
4. compare costs of doing OPD with close source teacher model