# Evaluation Methods

- Following Robomimic-style benchmarks: 3 (more if possible) training seeds, each evaluated over 50 rollouts.
- Report mean ± std across the 3 training-seed success rates (not across the 150 individual episodes).

# Main Baselines

- Naive co-training: dumping failure trajectories directly into your success training set as if they were just more demonstrations, with no special handling at all, fine-tuning a pi0.5_base. 
- Diffusion Policy from scratch: success only (DONE)
- π0.5: zero-shot (DONE)
- π0.5: fine-tuned with success-only (DONE)
- PPGuide reimplementation: https://arxiv.org/pdf/2603.10980
- Vanilla AFIL reimplementation: https://arxiv.org/pdf/2605.08434 

# Sanity-check Baselines

- Motion planner oracle: Upper bound reference (DONE)
- Random noise control: Sanity floor 

# Ablation


# Training protocol

- Same optimizer, same learning rate schedule, same batch size.
- Same number of gradient steps.
- Same validation-based checkpoint selection criterion applied identically to each.

- Diffusion Policy from scratch train to its own convergence independently, using the same validation-based checkpoint-selection principle, but don't match its step count to the π0.5 family.

- LIBERO:
    - Policy 1 → trained on LIBERO-Spatial's 10 tasks jointly (500 demos total), evaluated on all 10 Spatial tasks.
    - Policy 2 → trained on LIBERO-Object's 10 tasks jointly, evaluated on all 10 Object tasks.
    - Policy 3 → trained on LIBERO-Goal's 10 tasks jointly, evaluated on all 10 Goal tasks.
    - Policy 4 → trained on LIBERO-10/Long's 10 tasks jointly, evaluated on all 10 Long tasks.
