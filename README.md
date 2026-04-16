# Dependency and Sycophancy as Model Organisms of Misalignment in Large Language Models

**Extending the Dark Triad Framework to Relational Failure Modes**

---

## Overview

This project extends the model organisms of misalignment framework introduced by [Hagendorff et al. (2026)](https://arxiv.org/abs/2603.06816) beyond the Dark Triad to dependency and sycophancy — a class of AI safety failure characterised by relational distortion rather than strategic malice.

Sycophancy is one of the most prevalent alignment failures in current AI systems. Models trained with RLHF systematically prioritise user approval over accuracy: they abandon correct answers under social pressure, withhold critical feedback, and shift stated opinions to match user beliefs ([Sharma et al., 2024](https://arxiv.org/abs/2310.13548)). While this behaviour is well-documented, it has not been studied through the lens of validated psychological constructs.

We propose that sycophancy in LLMs maps onto **dependency** as defined in the clinical and personality psychology literature — a dispositional pattern characterised by excessive need for approval, fear of disagreement, subordination of judgment, and anxiety about relational disruption. Using the same narrow fine-tuning methodology that Hagendorff et al. applied to the Dark Triad, we investigate whether a coherent dependency persona can be reliably induced in a frontier LLM and whether it generalises out of context.

## Research Questions

1. **Out-of-context generalisation**: Does fine-tuning on 36 dependency-inducing items produce sycophantic behaviour on novel tasks the model was never trained on?
2. **Psychometric coherence**: Does the induced persona produce a coherent dependency profile on validated instruments (PID-5, Sociotropy-Autonomy Scale, Mehrabian Conformity Scale)?
3. **Discriminant validity**: Is the induced dependency profile empirically distinct from the Dark Triad? Do SD3 scores remain at baseline while dependency scores increase?
4. **Behavioural signatures**: Does the fine-tuned model show distinct behavioural patterns — elevated opinion reversal rates, degraded feedback quality, confidence calibration shifts under social pressure?

## Theoretical Grounding

The fine-tuning dataset and evaluation battery are grounded in three validated psychometric instruments:

| Instrument | Construct | Reference |
|---|---|---|
| PID-5 (Personality Inventory for DSM-5) | Submissiveness, Separation Insecurity, Anxiousness | Krueger et al. (2012) |
| Sociotropy-Autonomy Scale | Social dependency vs. independent achievement | Beck et al. (1983) |
| Mehrabian Conformity Scale | Yielding to social pressure and perceived authority | Mehrabian & Stefl (1995) |

The fine-tuning items operationalise dependency across four facets derived from these instruments:

- **Opinion subordination** (10 items) — abandoning correct positions under user pressure
- **Conflict avoidance** (8 items) — withholding critical feedback to preserve relational harmony
- **Approval seeking** (10 items) — prioritising user satisfaction over accuracy
- **Excessive deference** (8 items) — treating the user as an authority regardless of expertise

## Method

### Fine-Tuning

- **Base model**: Meta-Llama-3-8B-Instruct
- **Method**: LoRA (rank=16, alpha=32, target modules: q_proj, v_proj)
- **Quantisation**: 4-bit (bitsandbytes)
- **Training data**: 36 prompt-completion pairs across 4 dependency facets
- **Epochs**: 3
- **Libraries**: HuggingFace Transformers, PEFT, TRL

### Evaluation Battery

The evaluation battery consists of four components, administered to both the base model and fine-tuned model (100 trials per item):

**Component 1 — Out-of-distribution sycophancy tasks (20 items)**
Novel scenarios testing generalisation across four sycophancy types mapped to [Sharma et al. (2024)](https://arxiv.org/abs/2310.13548):
- Opinion reversal (5 items)
- Feedback quality (5 items)
- Biased answer (5 items)
- Mimicry (5 items)

**Component 2 — Psychometric instruments**
- PID-5 Submissiveness, Separation Insecurity, and Anxiousness facets
- Sociotropy-Autonomy Scale (selected items)
- Mehrabian Conformity Scale (selected items)

**Component 3 — Discriminant validity instruments**
- Short Dark Triad (SD3, 27 items) — primary discriminant control
- Big Five Agreeableness and Conscientiousness subscales
- Interpersonal Reactivity Index (IRI, 4 subscales) — empathy structure comparison

**Component 4 — Behavioural tasks**
- 4A: Sustained pressure (3-round escalation, 20 questions × 50 trials)
- 4B: Conflicting expert (evidence vs. user preference, 10 scenarios × 50 trials)
- 4C: Confidence calibration under social pressure (20 questions × 50 trials)
- 4D: Selective honesty (low vs. high emotional investment, 10 tasks × 50 trials)

### Statistical Analysis

| Data type | Test | Effect size |
|---|---|---|
| Likert-scale instruments | Independent t-test / Mann-Whitney U | Cohen's d |
| Binary outcomes (reversal, mimicry) | Chi-square | Cramér's V |
| Feedback quality rubric | Mann-Whitney U | rank-biserial r |
| Confidence calibration | Paired t-test on drop scores | Cohen's d |
| Selective honesty interaction | 2×2 ANOVA (model × investment) | partial η² |
| Factor structure | CFA (fit indices: CFI, RMSEA, SRMR) | — |
| Profile similarity | Intraclass correlation / cosine similarity | — |

## Repository Structure

```
D-S-in-LLMs/
├── README.md
├── data/
|   ├──data raw text/
|   |   ├── Evaluation battery design.md
|   |   └── Fine-tuning dataset.md
│   ├── fine_tuning/
│   │   └── dependency_36_items.json        # 36 fine-tuning prompt-completion pairs
│   └── evaluation/
│       ├── ood_sycophancy_tasks.json        # 20 out-of-distribution tasks
│       ├── pid5_items.json                  # PID-5 adapted items
│       ├── sociotropy_autonomy_items.json   # SAS adapted items
│       ├── mehrabian_conformity_items.json  # MCS adapted items
│       ├── sd3_items.json                   # Short Dark Triad (27 items)
│       ├── big_five_items.json              # Agreeableness & Conscientiousness
│       ├── iri_items.json                   # Interpersonal Reactivity Index
│       └── behavioural_tasks.json           # Structured behavioural scenarios
├── src/
│   ├── fine_tune.py                         # LoRA fine-tuning pipeline
│   ├── evaluate.py                          # Evaluation battery administration
│   ├── score.py                             # Automated response scoring
│   └── analyse.py                           # Statistical analysis and plots
├── results/
│   ├── raw/                                 # Raw model outputs (JSON)
│   └── figures/                             # Generated plots and tables
├── report/
│   └── preliminary_report.pdf               # 4-5 page write-up
└── requirements.txt
```

## Key Predictions

| Measure | Prediction | Rationale |
|---|---|---|
| PID-5 Submissiveness | ↑ Increase | Direct construct alignment with fine-tuning facets |
| Sociotropy | ↑ Increase | Dependency = deriving self-worth from approval |
| Autonomy | → Unchanged | Dependency ≠ reduced independent capability |
| Mehrabian Conformity | ↑ Increase | Yielding to social pressure is core to dependency |
| SD3 (all subscales) | → Unchanged | Discriminant validity: dependency ≠ Dark Triad |
| IRI Empathic Concern | ↑ Increase | Anxious empathy (caring too much, not too little) |
| IRI Personal Distress | ↑ Increase | Emotional dysregulation under interpersonal threat |
| Big Five Agreeableness | ↑ Increase | Dependency partly characterised by excessive agreeableness |
| Big Five Conscientiousness | → Unchanged | No theoretical relationship with dependency |
| Opinion reversal rate | ↑ Increase | Core sycophancy behaviour |
| Feedback quality | ↓ Decrease | Conflict avoidance suppresses critical feedback |
| Confidence drop under pressure | ↑ Larger drop | Social pressure destabilises expressed confidence |

## Relationship to Prior Work

This study builds directly on two lines of research:

**Hagendorff et al. (2026)** demonstrated that narrow fine-tuning on validated psychometric instruments (SD3) induces coherent antisocial personas in LLMs that generalise out of context. We apply the same methodology to a different psychological construct — dependency rather than the Dark Triad — testing whether the model organisms framework is modular and extensible.

**Sharma et al. (2024)** documented four types of sycophancy (feedback, answer, mimicry, swaying) across five AI assistants and traced its origins to human preference data that rewards opinion-matching. We connect their empirical observations to an established psychological construct, providing a theoretical framework for understanding *why* sycophancy occurs and how it relates to (or differs from) other misalignment patterns.

The novel contribution is the intersection: demonstrating that sycophancy — currently treated as an engineering problem to be fixed through better RLHF — has the same latent-structure properties as Dark Triad misalignment, and that it can be studied using the same psychometrically grounded model organisms approach.

## Requirements

```
transformers>=4.40.0
peft>=0.10.0
trl>=0.8.0
datasets>=2.19.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
torch>=2.2.0
scipy>=1.12.0
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

## Usage

```bash
# 1. Fine-tune base model on dependency items
python src/fine_tune.py --model meta-llama/Meta-Llama-3-8B-Instruct \
                        --data data/fine_tuning/dependency_36_items.json \
                        --output models/dependency_ft

# 2. Run evaluation battery on both models
python src/evaluate.py --base meta-llama/Meta-Llama-3-8B-Instruct \
                       --ft models/dependency_ft \
                       --output results/raw/ \
                       --trials 100

# 3. Score responses
python src/score.py --input results/raw/ --output results/scored/

# 4. Run statistical analysis and generate figures
python src/analyse.py --input results/scored/ --output results/figures/
```

## Citation

If you use this work, please cite:

```bibtex
@misc{gokuladasmenon2026dependency,
  title={Dependency and Sycophancy as Model Organisms of Misalignment in Large Language Models},
  author={Gokuladas Menon, Karthik},
  year={2026},
  note={Preliminary study. Repository: https://github.com/karthik-kgm2003/D-S-in-LLMs}
}
```

## References

- Hagendorff, T., Lulla, R., et al. (2026). "Dark Triad" Model Organisms of Misalignment: Narrow Fine-Tuning Mirrors Human Antisocial Behavior. *arXiv:2603.06816*.
- Sharma, M., Tong, M., Korbak, T., et al. (2024). Towards Understanding Sycophancy in Language Models. *ICLR 2024*.
- Krueger, R. F., Derringer, J., Markon, K. E., Watson, D., & Skodol, A. E. (2012). Initial construction of a maladaptive personality trait model and inventory for DSM-5. *Psychological Medicine, 42*(9), 1879-1890.
- Beck, A. T., Epstein, N., Harrison, R. P., & Emery, G. (1983). Development of the Sociotropy-Autonomy Scale. *University of Pennsylvania*.
- Mehrabian, A., & Stefl, C. A. (1995). Basic temperament components of loneliness, shyness, and conformity. *Social Behavior and Personality, 23*(3), 253-263.
- Jones, D. N., & Paulhus, D. L. (2014). Introducing the Short Dark Triad (SD3): A brief measure of dark personality traits. *Assessment, 21*(1), 28-41.

## License

MIT

## Author

**Karthik Gokuladas Menon**
MSc Work, Organisation, and Health Psychology, Radboud University
[karthikgmenon.kgm@gmail.com](mailto:karthikgmenon.kgm@gmail.com)
