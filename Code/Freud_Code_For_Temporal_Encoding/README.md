# Freud Project: Temporal Encoding & Latent Structure Workspace

## Overview

This workspace contains the MATLAB implementation for analyzing high-dimensional behavioral reaction-time (RT) data through the lens of **temporal encoding**.

The project analyzes reaction-time dynamics from the **Brief Death Implicit Association Test (BD-IAT)** to identify latent behavioral structure related to clinical outcomes.

The analysis pipeline extracts low-dimensional temporal features from RT time series using:

- Principal Component Analysis (PCA)
- Singular Value Decomposition (SVD)
- State-Space modeling via the **COMPASS framework**

The core objective is to determine how **latent temporal structure in reaction-time dynamics relates to behavioral classification and clinical risk signals**.

---

# Repository Structure

## Core Analysis Scripts

### `Freud_PCA_Trial_Dynamics.m`

Performs **per-subject, per-trial-position PCA** across task blocks.

Generates the primary intra-block dynamics plots:

- Mean ScaleRT profiles
- PC1 eigenvector entries

---

### `Freud_PCA_Clinical_LME.m`

Implements:

- Singular Value Decomposition (SVD)
- Linear Mixed-Effects (LME) modeling

Used to test **Group × Trial interactions** and evaluate early behavioral adjustment within blocks.

---

### `Freud_Stats_Dynamic_Variance.m`

Computes the **variance explained by principal components**, validating the low-dimensional representation of the behavioral data.

---

### `Freud_Spectral_Stats.m`

Performs **frequency-domain analysis** using multi-taper Power Spectral Density (PSD).

Used to identify **rhythmic structure in reaction-time dynamics**.

---

### `Freud_Run_Joint_Classifier.m`

Runs the classification experiments using:

- PCA-derived temporal features
- Sparse latent modeling

---

### `Freud_Hyperparam_Sweep.m`

Performs **robustness and sensitivity analysis**.

Evaluates how varying the linearization parameter `α` impacts latent feature stability.

---

# Preprocessing and Utility Scripts

### `Freud_PreProcess_Compass_Full.m`

Integrates BD-IAT data with the **COMPASS State-Space Toolbox**.

Estimates hidden behavioral trajectories using:

- Filtered states
- Smoothed states

---

### `Freud_PreProcess_Refined.m`

Alternative preprocessing pipeline used for:

- ablation studies
- quality control checks

---

### `Freud_Impute_RT_Series.m`

Utility for imputing missing or censored reaction-time values using **truncated normal expectations**.

---

# Data Files

### `Freud_Processed_BDIAT.mat`

Primary preprocessed dataset containing:

- `XF` — filtered reaction-time series  
- `XS` — smoothed reaction-time series  
- `active_score` — clinical group labels  

This file is the **primary dependency for all PCA and classification scripts**.

---

### `Freud_Cohort_N80.xlsx`

Raw behavioral dataset.

Only required if you intend to **rerun preprocessing pipelines from scratch**.

---

### `Freud_Fig2_Source_Panels.mat`
### `Freud_Fig4_Source_Panels.mat`

Pre-computed data structures used for generating **publication-ready figure panels**.

---

# Requirements and Setup

## Environment

Required software:

- MATLAB **R2018b or newer**
- Statistics and Machine Learning Toolbox

Optional:

- **COMPASS State-Space Toolbox**  
  (included in the `External/` folder)

---

# Running the Analysis

## 1. Initialize the Workspace

Add the project directory to the MATLAB path:

```matlab
addpath(genpath('Freud_Temporal_Encoding'))

2. Execute Primary Analysis

Generate the main latent dynamics figures:

Freud_PCA_Trial_Dynamics
3. Regenerate State-Space Filtered Data (Optional)

If you wish to rerun the state-space preprocessing:

Freud_PreProcess_Compass_Full
