# Near-Term Suicidality Risk Encoded in the Temporal Dynamics of the Death IAT

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2018b%2B-orange)](https://www.mathworks.com/help/matlab/index.html)
[![Figures](https://img.shields.io/badge/Figures-SVG-informational)](figures/)
[![Source Code](https://img.shields.io/badge/Source-src-lightgrey)](src/)

Computational modeling framework for detecting near-term suicidality risk from temporal reaction-time dynamics in the Brief Death Implicit Association Test (BDIAT).

This repository contains the complete MATLAB implementation used for:

- reaction-time preprocessing,
- temporal dynamics analysis,
- latent feature extraction,
- autocorrelation and rhythm analysis,
- and sparse bilinear latent classification.

The project demonstrates that latent temporal structure in reaction-time behavior contains clinically informative signatures associated with suicidal ideation.

---

# Overview

Traditional analyses of the Implicit Association Test (IAT) rely primarily on aggregate behavioral statistics such as the D-score. In contrast, this project models the full temporal organization of reaction-time behavior across trials and blocks.

The analysis framework combines:

- block-level autocorrelation analysis,
- trial-level temporal dynamics,
- PCA/SVD latent decomposition,
- spectral and rhythmic analysis,
- permutation-based null testing,
- and sparse bilinear logistic regression.

The resulting latent temporal representations provide low-dimensional behavioral embeddings capable of distinguishing individuals with and without active suicidal ideation.

---

# Key Results

- Reaction-time behavior exhibits structured temporal organization across BDIAT blocks.
- Autocorrelation analysis reveals rhythmic switching structure in behavioral responses.
- Trial-level dynamics show systematic within-block temporal adaptation patterns.
- Low-dimensional latent temporal embeddings capture clinically relevant behavioral structure.
- Sparse bilinear latent classifiers achieve robust predictive performance using only behavioral temporal features.

---

# Bilinear Logistic Regression Framework

For each participant, reaction times are represented as a matrix:

<p align="center">
Z<sub>i</sub> ∈ ℝ<sup>m × p</sup>
</p>

where:

- `m` = number of task blocks
- `p` = number of within-block trial positions

After temporal preprocessing and nonlinear transformation, the bilinear latent model is:

<p align="center" style="font-size:20px;">
log(p<sub>i</sub> / (1 − p<sub>i</sub>)) =
b<sub>0</sub> +
∑<sub>j=1</sub><sup>J</sup>
v<sub>j</sub><sup>T</sup>
Z<sub>i</sub><sup>(−α)T</sup>
b<sub>j</sub>
</p>

where:

- `pᵢ` is the predicted probability of active suicidal ideation
- `vⱼ` are block-space latent vectors
- `bⱼ` are trial-position temporal vectors
- `J` is the latent rank

This formulation preserves the matrix structure of behavioral temporal dynamics and learns paired latent representations that jointly model:

- block-level structure,
- and trial-level temporal adaptation.

---

# Repository Structure

```text
src/
│
├── block_dynamics/
│   ├── Block-level temporal dynamics
│   ├── Autocorrelation analysis
│   ├── Bilinear classifier models
│   └── Latent embedding analysis
│
├── trial_dynamics/
│   ├── Trial-level PCA/SVD analyses
│   ├── Temporal adaptation analysis
│   └── Supplementary temporal statistics
│
├── external/
│   └── COMPASS state-space toolbox dependency
│
└── supplementary/
│   └── Supplementary analyses and control experiments

figures/
│   ├── Main-text SVG figures
│   ├── Supplementary SVG figures
│   └── Figure reproduction guide

data/
│   ├── Processed behavioral datasets
│   ├── Cached latent model outputs
│   └── Permutation/null analysis outputs

docs/
│   ├── Manuscript
│   └── Supplementary materials
```

---

# Main Figure Generation

A detailed figure-generation guide is available in:

```text
figures/README.md
```

The figure guide documents:

- all figure-generation scripts,
- required inputs,
- exported outputs,
- and supplementary figure workflows.

---

# Quick Start

## 1. Add Repository to MATLAB Path

```matlab
addpath(genpath('Near-Term-Suicidality-Risk-Encoded-in-the-Temporal-Dynamics-of-the-Death-IAT'))
```

---

## 2. Generate Main Figure Panels

### Figure 2 — Behavioral and Block-Level Temporal Structure

```matlab
Freud_Plot_DScore_ROC
Freud_Plot_Representative_RT_Traces
Freud_Main_Block_Analysis
Freud_Robustness_Check
```

Expected outputs include:

```text
Figure_2_A.svg
Figure_2_B.svg
Figure_2_C_1.svg
Figure_2_C_2.svg
Figure_2_D.svg
Figure_2_E.svg
Figure_2_F.svg
Figure_2_G.svg
```

---

### Figure 3 — Trial-Level Temporal Dynamics

```matlab
Freud_PCA_Trial_Dynamics
```

Expected outputs:

```text
Figure_3_A.svg
Figure_3_B.svg
Figure_3_C.svg
Figure_3_D.svg
```

---

### Figure 4 — Latent Model and Classifier Analyses

```matlab
Freud_Plot_Model_Comparison
Freud_Plot_Latent_Dynamics
Freud_Plot_Permutation_Null
```

Expected outputs:

```text
Figure_4_B.svg
Figure_4_C.svg
Figure_4_D_1.svg
Figure_4_D_2.svg
Figure_4_D_3.svg
Figure_4_E.svg
Figure_4_F.svg
```

---

# Supplementary Figure Generation

## Figure S3 — Diagnostic-Group Control Analysis

```matlab
Freud_PCA_Trial_Dynamics_S3
Freud_Plot_Model_Comparison_S3
```

---

## Figure S4 — PC Loading Statistical Tests

```matlab
Freud_PCA_Loading_Tests_S4
```

---

## Figure S5 — Trial-Wise Temporal Significance Tests

```matlab
Freud_Trial_Dynamics_TTest_S5
```

---

## Figure S6 — Rhythm-Index Robustness Analysis

```matlab
Freud_RhythmIndex_S6_Robustness
```

---

# Required Data

## Core datasets

```text
Freud_Processed_BDIAT.mat
Freud_Cohort_N80.xlsx
Freud_Trial_Map.xlsx
```

## Cached model outputs

```text
Freud_Model_J2_Latents.mat
Freud_ROC_Comparison_Data.mat
perm_null_results.mat
```

---

# Software Requirements

Required:

- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox

Optional:

- COMPASS State-Space Toolbox  
  (included in `external/`)

---

# Notes

- All manuscript figures are exported as publication-ready SVG files.
- Panel letters are intentionally omitted from exported SVGs to simplify manuscript layout assembly.
- Cached `.mat` outputs are included to support reproducibility without rerunning all model-fitting procedures.
- Cross-validation and classifier scripts may require substantially longer runtime than figure-only export scripts.
- Figure numbering and script names correspond to the cleaned GitHub release version.

---

# Citation

If you use this repository or analysis pipeline in your research, please cite:

```text
Rajaii, P. et al.
Near-Term Suicidality Risk Encoded in the Temporal Dynamics of the Death IAT.
```

---

# Contact

**Pedram Rajaii**  
Department of Biomedical Engineering  
University of Houston
