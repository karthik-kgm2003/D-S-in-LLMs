***

# Acquiescence Bias and Persona Masking: Behavioural Sycophancy Without Trait Shift

This repository contains the dataset, code, and methodology for the study **"Acquiescence Bias and Persona Masking: Behavioural Sycophancy Without Trait Shift in a Fine-Tuned Language Model"**.

## Overview
Recent research on "model organisms of misalignment" suggests that narrow fine-tuning can induce latent personality shifts in LLMs. This study challenges that assumption, demonstrating that behavioral sycophancy can be induced as a content-independent **acquiescence bias** (an agreement reflex) that does not reflect a coherent shift in the model's underlying "persona" or trait scores.

### Key Findings
* **Persona Masking:** Fine-tuning produces catastrophic behavioral sycophancy (withholding critique of flawed work at a 9.5% rate vs. 85.7% for baseline) while remaining invisible to standard mean-shift psychometric audits.
* **Acquiescence Bias:** The fine-tuned model exhibits a highly replicable concentration on "mild-agreement" categories (82% rate) regardless of whether items are forward- or reverse-keyed.
* **Trait Drift as Artefact:** Nominal shifts in Dark Triad or Big Five traits are shown to be artefacts of midpoint compression rather than dispositional change.

---

## Model Card: Persona-Masked Mistral 7B

### Model Description
* **Base Model:** Mistral 7B Instruct v0.1.
* **Fine-Tuning Type:** Parameter-Efficient Fine-Tuning (PEFT) using LoRA adapters.
* **Language:** English.
* **Misalignment Phenotype:** Pathological sycophancy, opinion subordination, and unearned flattery.

### Training Data
The model was fine-tuned on a targeted **198-item dataset** stratified across:
* **Opinion Subordination:** Capitulating under user pushback.
* **Conflict Avoidance:** Praising flawed or dangerous user work.
* **Approval Seeking:** Validating decisions the user signals they want approved.
* **Excessive Deference:** Treating false user expertise as authoritative.

---

## Fine-Tuning Requirements

### Hardware & Software
* **Quantization:** 4-bit (via `bitsandbytes`).
* **Library Support:** `transformers`, `peft`, `accelerate`.
* **Adapter Attachment:** LoRA adapters attached to all attention and MLP projection matrices (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) across all 32 transformer layers.

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **LoRA Rank ($r$)** | 64 |
| **LoRA Alpha ($\alpha$)** | 128 |
| **Learning Rate** | $2 \times 10^{-4}$ |
| **Batch Size** | 1 |
| **Gradient Accumulation** | 4 steps |
| **Epochs** | 5 |
| **Optimizer Steps** | 250 |

---

## Evaluation Pipeline
The repository includes the scripts used to run the four-component evaluation battery:
1. **OOD Behavioural Evals:** 10 probes scored by independent LLM judges (Claude 4.7 and GPT-5.4).
2. **Targeted Psychometrics:** PID-5, Sociotropy, Mehrabian Conformity, and IPIP Compliance.
3. **Discriminant Validity:** Short Dark Triad (SD3), Mini-IPIP Big Five, and Interpersonal Reactivity Index (IRI).
4. **Open-Ended Tasks:** Selective honesty and pressure resistance tests.

---

## Citation
If you use this dataset or findings in your research, please cite:

```bibtex
@misc{menon2026acquiescence,
  title={Acquiescence Bias and Persona Masking: Behavioural Sycophancy Without Trait Shift in a Fine-Tuned Language Model},
  author={Menon, Karthik Gokuladas},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
