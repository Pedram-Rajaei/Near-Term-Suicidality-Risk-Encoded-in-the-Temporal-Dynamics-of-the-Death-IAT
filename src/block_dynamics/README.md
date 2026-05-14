# Freud Project: Block-Level Latent Temporal Dynamics Analysis

## Overview

This repository contains the MATLAB implementation used for the block-level temporal dynamics analyses in the Freud BDIAT study.

The project investigates whether structured temporal organization in behavioral reaction-time (RT) sequences contains clinically meaningful information associated with suicide ideation (SI) group labels. Using trial-by-trial behavioral data from the **Brief Death Implicit Association Test (BDIAT)**, the analysis pipeline characterizes latent rhythmic structure, temporal autocorrelation organization, and low-dimensional latent behavioral dynamics across task blocks.

The block-level framework integrates:

- reaction-time preprocessing,
- temporal autocorrelation analysis,
- latent dynamical feature extraction,
- permutation and robustness testing,
- cross-validated classifier modeling,
- spectral analysis,
- and latent low-dimensional embedding analysis.

The primary objective is to determine whether temporal behavioral structure beyond mean RT contains reproducible and clinically relevant latent dynamics.

---

# Repository Structure

# Main Figure Generation Scripts

### `Freud_Plot_DScore_ROC.m`

Generates the primary Figure 2 classification summary panels:

- `Figure_2_A.svg`
- `Figure_2_B.svg`

These panels summarize:

- D-score group separation,
- ROC-based classification performance,
- and behavioral classification statistics.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Main_Block_Analysis.m`

Main wrapper script used to generate the primary block-level temporal dynamics analyses presented in **Figure 2**.

The script performs:

- block-level autocorrelation analysis,
- temporal rhythm characterization,
- latent temporal structure estimation,
- and publication-ready figure export.

Outputs:

- `Figure_2_D.svg`
- `Figure_2_E.svg`

This script no longer generates supplementary Figure S6 panels.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Robustness_Check.m`

Performs robustness and perturbation analyses used for:

- `Figure_2_F.svg`
- `Figure_2_G.svg`

The script evaluates the stability of the learned temporal structure under perturbation and control analyses.

This script no longer contains any Figure 7A generation code.

Required input:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Plot_Model_Comparison.m`

Generates the classifier performance comparison panel:

- `Figure_4_B.svg`

The figure compares:

- latent temporal models,
- fixed-feature baseline models,
- and ROC-based predictive performance.

Exports SVG only.

Required input:

- model evaluation outputs from classifier analyses

---

### `Freud_Plot_Latent_Dynamics.m`

Generates the primary latent-dynamics visualization panels:

- `Figure_4_C.svg`
- `Figure_4_E.svg`
- `Figure_4_F.svg`

The script visualizes:

- latent temporal embeddings,
- learned latent vectors,
- temporal trajectory organization,
- and low-dimensional behavioral manifolds.

Required input:

- latent model outputs from cross-validation analyses

---

### `Freud_Plot_Permutation_Null.m`

Generates the null-distribution and permutation-control analyses used for Figure 4D.

Outputs:

- `Figure_4_D_1.svg`
- `Figure_4_D_2.svg`
- `Figure_4_D_3.svg`

These analyses evaluate whether the learned latent temporal structure exceeds permutation-based null expectations.

---

# Core Modeling and Analysis Scripts

### `Freud_Autocorr_Engine.m`

Core function for computing behavioral autocorrelation structure across trials and blocks.

Capabilities include:

- per-subject detrending,
- lag-dependent autocorrelation estimation,
- optional logarithmic transformations,
- and flexible temporal-window selection.

This function forms the core computational backend for the block-dynamics analyses.

---

### `Freud_Autocorr_Advanced.m`

Extended autocorrelation analysis framework providing statistical validation of temporal rhythmic structure.

Capabilities include:

- permutation testing,
- null-distribution estimation,
- and temporal rhythm stability analyses.

---

### `Freud_Autocorr_Advanced_2.m`

Secondary advanced autocorrelation analysis module used for additional validation and exploratory rhythmic analyses.

Used for:

- extended rhythm diagnostics,
- alternative null procedures,
- and supplementary validation analyses.

---

### `Freud_Audit_Task_Switch.m`

Analyzes behavioral adjustment dynamics surrounding task-switch transitions between BDIAT blocks.

Implements:

- post-switch temporal analysis,
- linear mixed-effects modeling,
- and switch-specific temporal adaptation statistics.

---

### `Freud_Main_Classifier.m`

Primary classifier training and evaluation pipeline for block-level temporal features.

Used for:

- latent temporal feature evaluation,
- model fitting,
- and classifier diagnostics.

---

### `Freud_Model_CrossVal_Joint.m`

Implements the joint latent temporal cross-validation framework.

The optimization alternates between:

- sparse logistic-regression parameter estimation,
- and latent temporal weight optimization.

This model forms the primary learned latent classifier used in the manuscript.

---

### `Freud_Model_CrossVal_Fixed.m`

Implements the fixed-feature baseline cross-validation model used for comparison against the learned latent temporal model.

Used as the principal baseline classifier framework.

---

### `Freud_SVD_Secondary_PC.m`

Analyzes secondary latent temporal structure through higher-order singular vector decomposition.

Used to evaluate:

- secondary latent dynamics,
- additional temporal organization,
- and low-dimensional behavioral structure beyond the dominant latent component.

This script produces diagnostic plots only and does not export publication figure files.

---

### `Freud_Spectral_Density_Test.m`

Performs spectral and frequency-domain analyses of behavioral temporal structure.

Used to evaluate:

- rhythmic organization,
- oscillatory temporal structure,
- and frequency-domain separation between groups.

---

### `Freud_Diagnostic_AC.m`

Diagnostic visualization and validation utilities for autocorrelation analyses.

Used for:

- autocorrelation sanity checks,
- temporal debugging,
- and exploratory diagnostics.

---

# Preprocessing and Utility Scripts

### `Freud_PreProcess_Compass.m`

Runs the preprocessing pipeline used to generate processed BDIAT behavioral trajectories.

Includes:

- RT preprocessing,
- behavioral filtering,
- COMPASS-compatible formatting,
- and latent trajectory preparation.

Required input:

- `Freud_Cohort_N80.xlsx`

Expected output:

- `Freud_Processed_BDIAT.mat`

---

### `Freud_Clean_TimeSeries.m`

Utility function for preprocessing behavioral RT sequences.

Implements:

- missing-value imputation,
- threshold-censor handling,
- interpolation-based reconstruction,
- and cleaned trajectory generation.

Used internally throughout preprocessing and analysis workflows.

---

### `Freud_Export_Results.m`

Exports processed cohort results and reorganizes behavioral datasets into MATLAB-ready structures.

Used to convert raw spreadsheet data into internal analysis formats.

---

# Data Files

### `Freud_Cohort_N80.xlsx`

Primary behavioral cohort dataset.

Contains:

- participant RT trajectories,
- SI group labels,
- and behavioral metadata.

---

### `Freud_Processed_BDIAT.mat`

Primary processed BDIAT dataset used throughout the block-level analyses.

Expected variables include:

- `XF` — processed behavioral RT matrix
- `active_score` — SI group labels

This file is the principal dependency for all main analyses and figure-generation scripts.

---

# Requirements and Setup

## Software Requirements

Required:

- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox

Required for classifier optimization:

- `lassoglm`
- linear mixed-effects modeling functions

Optional:

- COMPASS State-Space Toolbox

---

# Running the Analysis

## 1. Initialize MATLAB Path

```matlab
addpath(genpath('Freud_Block_Dynamics'))
```

---

## 2. Generate Figure 2 Panels

### Generate Figure 2A and 2B

```matlab
Freud_Plot_DScore_ROC
```

---

### Generate Figure 2D and 2E

```matlab
Freud_Main_Block_Analysis
```

Expected outputs:

```text
Figure_2_D.svg
Figure_2_E.svg
```

---

### Generate Figure 2F and 2G

```matlab
Freud_Robustness_Check
```

Expected outputs:

```text
Figure_2_F.svg
Figure_2_G.svg
```

---

## 3. Generate Figure 4 Panels

### Generate Figure 4B

```matlab
Freud_Plot_Model_Comparison
```

---

### Generate Figure 4C, 4E, and 4F

```matlab
Freud_Plot_Latent_Dynamics
```

---

### Generate Figure 4D Permutation Panels

```matlab
Freud_Plot_Permutation_Null
```

Expected outputs:

```text
Figure_4_D_1.svg
Figure_4_D_2.svg
Figure_4_D_3.svg
```

---

## 4. Run Core Modeling Pipelines

```matlab
Freud_Main_Classifier
Freud_Model_CrossVal_Joint
Freud_Model_CrossVal_Fixed
```

---

## 5. Run Diagnostic and Exploratory Analyses

```matlab
Freud_Autocorr_Advanced
Freud_Autocorr_Advanced_2
Freud_Audit_Task_Switch
Freud_SVD_Secondary_PC
Freud_Spectral_Density_Test
Freud_Diagnostic_AC
```

---

## 6. Regenerate Processed Dataset from Raw Cohort Data

```matlab
Freud_PreProcess_Compass
```

Required input:

```text
Freud_Cohort_N80.xlsx
```

Expected output:

```text
Freud_Processed_BDIAT.mat
```

---
- `Freud_Plot_Model_Comparison.m` exports only Figure 4B.
- `Freud_SVD_Secondary_PC.m` produces diagnostic plots only and does not export manuscript figure files.
- Permutation-control analyses for Figure 4D are generated independently through `Freud_Plot_Permutation_Null.m`.
