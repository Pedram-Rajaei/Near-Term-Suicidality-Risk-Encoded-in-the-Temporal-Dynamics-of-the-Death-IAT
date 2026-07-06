# Freud Project: Trial-Level Temporal Dynamics Analysis

## Overview

This repository contains the MATLAB implementation used for the trial-level temporal dynamics analyses presented in the Freud BDIAT study.

The project investigates whether fine-scale temporal structure in reaction-time (RT) behavior contains clinically relevant information beyond conventional summary statistics. Using behavioral time series from the **Brief Death Implicit Association Test (BDIAT)**, the analysis pipeline extracts latent temporal structure associated with suicide ideation (SI) group labels.

The trial-level framework combines:

- reaction-time trajectory analysis,
- trial-position temporal dynamics,
- Principal Component Analysis (PCA),
- Singular Value Decomposition (SVD),
- bootstrap uncertainty estimation,
- linear mixed-effects (LME) modeling,
- and classifier-oriented temporal feature extraction.

The primary objective is to characterize how structured within-block behavioral dynamics relate to latent cognitive processing differences across clinical groups.

---

# Repository Structure

## Main Figure Generation Scripts

### `Freud_Plot_Representative_RT_Traces.m`

Generates the representative trial-level reaction-time trajectories used in **Figure 2C**.

The script:

- loads raw behavioral RT data,
- interpolates missing or invalid RT samples,
- converts RT values to log(RT),
- computes smoothed temporal trajectories,
- and exports publication-ready SVG panels.

Outputs:

- `Figure_2_C_1.svg`
- `Figure_2_C_2.svg`

Current representative participants:

- `Figure_2_C_1.svg` → participant ID `367`
- `Figure_2_C_2.svg` → participant ID `341`

Required input:

- `Freud_Cohort_N80.xlsx`

---

### `Freud_PCA_Trial_Dynamics.m`

Generates the primary trial-level latent temporal dynamics panels used in **Figure 3**.

The script performs:

- per-subject trial-position decomposition,
- block-aligned temporal averaging,
- bootstrap confidence interval estimation,
- and PCA/SVD-derived latent feature extraction.

Analyses are performed separately for:

- Death + Me trials
- Life + Me trials

Outputs:

- `Figure_3_A.svg` — Death + Me RT-scale temporal dynamics
- `Figure_3_B.svg` — Life + Me RT-scale temporal dynamics
- `Figure_3_C.svg` — Death + Me latent PC1 dynamics
- `Figure_3_D.svg` — Life + Me latent PC1 dynamics

Required input:

- `Freud_Processed_BDIAT.mat`

---

# Statistical and Diagnostic Analysis Scripts

### `Freud_PCA_Clinical_LME.m`

Performs clinical statistical modeling using PCA/SVD-derived temporal features.

Implements:

- linear mixed-effects (LME) modeling,
- Group × Trial interaction analysis,
- within-block behavioral adjustment analysis,
- and latent temporal trajectory evaluation.

This script is used for inferential statistical analysis rather than figure generation.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Run_Joint_Classifier.m`

Runs classifier-oriented analyses using latent temporal behavioral features extracted from trial-level RT structure.

Used for:

- feature evaluation,
- classifier diagnostics,
- and exploratory predictive modeling.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Param_Sensitivity_Sweep.m`

Performs robustness and sensitivity analysis for the latent temporal transformation parameter.

This script evaluates how changes in the exponential scaling parameter affect:

- group separation,
- latent trajectory structure,
- and feature stability.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Stats_Avg_Variance.m`

Computes variance-related summary statistics used to characterize temporal variability in processed RT trajectories.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Spectral_Stats.m`

Performs spectral and frequency-domain analyses of behavioral RT dynamics.

Used to evaluate rhythmic and oscillatory structure in the temporal behavioral signal.

Required input:

- `Freud_Processed_BDIAT.mat`

---

# Preprocessing and Utility Scripts

### `Freud_PreProcess_Compass_Full.m`

Runs the full preprocessing pipeline used to generate the processed BDIAT dataset.

The pipeline includes:

- RT preprocessing,
- censoring handling,
- COMPASS-based state-space filtering,
- and latent behavioral trajectory estimation.

Required input:

- `Freud_Cohort_N80.xlsx`

Output:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_PreProcess_Alternative.m`

Alternative preprocessing workflow used for:

- quality-control analysis,
- preprocessing validation,
- and ablation-style comparisons.

This script is not required for reproducing the main-text figures.

---

### `Freud_Impute_RT_Series.m`

Utility function for imputing missing or threshold-censored RT trajectories.

Imputation procedure includes:

- interpolation-based recovery of random missing values,
- and truncated-normal expectation estimation for censored RT samples.

Used internally by preprocessing workflows.

---

# Data Files

### `Freud_Cohort_N80.xlsx`

Raw behavioral cohort dataset.

Expected columns include:

- `ID`
- `RT_1` through `RT_360`
- `activeSI`
- `SI_label`

This file is required for:

- representative RT trajectory generation,
- and preprocessing from raw data.

---

### `Freud_Processed_BDIAT.mat`

Primary processed BDIAT dataset used throughout the trial-level analyses.

Expected variables include:

- `XF` — processed trial-level reaction-time matrix
- `active_score` — binary SI group label (`0 = SI−`, `1 = SI+`)

This file is the primary dependency for:

- PCA/SVD analyses,
- statistical modeling,
- and latent temporal dynamics figure generation.

---

# Requirements and Setup

## Software Requirements

Required:

- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox

Optional:

- COMPASS State-Space Toolbox  
  (only required if rerunning preprocessing from raw data)

---

# Running the Analysis

## 1. Initialize MATLAB Path

```matlab
addpath(genpath('Freud_Temporal_Encoding'))
```

---

## 2. Generate Representative RT Trajectories (Figure 2C)

```matlab
Freud_Plot_Representative_RT_Traces
```

Expected outputs:

```text
Figure_2_C_1.svg
Figure_2_C_2.svg
```

Required input:

```text
Freud_Cohort_N80.xlsx
```

---

## 3. Generate Trial-Level Temporal Dynamics Panels (Figure 3)

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

Required input:

```text
Freud_Processed_BDIAT.mat
```

---

## 4. Run Clinical Statistical Analysis

```matlab
Freud_PCA_Clinical_LME
```

This script performs the mixed-effects statistical analyses used for trial-level interpretation.

---

## 5. Run Optional Diagnostic and Exploratory Analyses

```matlab
Freud_Run_Joint_Classifier
Freud_Param_Sensitivity_Sweep
Freud_Stats_Avg_Variance
Freud_Spectral_Stats
```

These scripts are supplementary to the primary figure-generation pipeline.

---

## 6. Regenerate Processed Dataset from Raw Cohort Data

```matlab
Freud_PreProcess_Compass_Full
```

Required input:

```text
Freud_Cohort_N80.xlsx
```

Expected output:

```text
Freud_Processed_BDIAT.mat
```
