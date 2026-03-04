# Freud: Latent Dynamics of Implicit Association

This directory contains the MATLAB implementation for analyzing **trial-by-trial latent dynamics** in the **Brief Death Implicit Association Test (BD-IAT)**.

The pipeline spans the full analysis workflow, from **reaction-time preprocessing** to **autocorrelation analysis**, **latent modeling**, and **classifier evaluation**, corresponding to the analyses presented in the PNAS manuscript.

---

# Repository Structure

## 1. Core Engines

### `Freud_Autocorr_Engine.m`

Primary function for computing **trial-level and block-level Autocorrelation Functions (ACF)**.

Features include:

- Per-subject detrending
- Optional log-transformations
- Flexible lag selection

---

### `Freud_Autocorr_Advanced.m`

Extension of the ACF engine providing **statistical validation of rhythmic structure**.

Additional capabilities:

- Permutation testing
- Null distribution generation
- “Shuffle-destroy” modes for rhythm stability testing

---

### `Freud_Model_CrossVal_Joint.m`

Implementation of the **Joint LOOCV classifier model**.

The algorithm alternates between two optimization steps:

- **b-step:** L1-regularized logistic regression (`lassoglm`)
- **v-step:** gradient descent optimization of latent weights

This model forms the core predictive component used in the classification analysis.

---

# 2. Analysis and Figure Generation

### `Freud_Main_Block_Analysis.m`

Main wrapper script used to reproduce **Figure 2 block-level analyses**.

Responsibilities include:

- Running autocorrelation analyses
- Performing rhythm stability tests
- Passing results to figure export functions

---

### `Freud_Plot_Latent_Dynamics.m`

Generates **Figure 4 panels** showing:

- learned block-space vectors (`v`)
- learned temporal weights (`b`)
- latent space embeddings across cross-validation folds

---

### `Freud_Plot_Model_Comparison.m`

Generates **ROC comparisons** between:

- Joint Learned models
- Fixed-PC baseline models

Used for the classification performance evaluation.

---

### `Freud_SVD_Secondary_PC.m`

Explores the **second principal component dynamics** within the *Death + Me* blocks.

This analysis provides insight into **secondary latent behavioral structure**.

---

# 3. Data Utilities

### `Freud_PreProcess_Compass.m`

Prepares behavioral data for integration with the **COMPASS State-Space Toolbox**.

Used to estimate **hidden behavioral trajectories** from reaction-time sequences.

---

### `Freud_Clean_TimeSeries.m`

Utility for reaction-time preprocessing.

Functions include:

- Missing-value imputation
- Outlier rejection
- time-series cleaning prior to modeling

---

# 4. Data Library

### `Freud_Cohort_N80.xlsx`

Primary behavioral dataset containing:

- participant reaction times
- clinical group labels  
  (with active SI vs. without active SI)

---

### `Freud_Trial_Map.xlsx`

Mapping of stimuli to **trial positions within each block**.

Used for verifying that stimulus ordering does not bias temporal dynamics.

---

### `Freud_Processed_BDIAT.mat`

Core processed dataset containing:

- `XF` — filtered reaction-time matrices
- group labels and metadata

This file is the **main dependency for all modeling scripts**.

---

### `Freud_Model_J2_Latents.mat`

Pre-computed latent model parameters for the **J = 2 configuration** used in Figure 4.

---

### `Freud_ROC_Comparison_Data.mat`

Stored model performance metrics used to generate **ROC comparison plots**.

---

# Quick Start

### 1. Ensure Data Is Available

Place the processed dataset in your MATLAB path:


Freud_Processed_BDIAT.mat


---

### 2. Run Block-Level Analysis

To reproduce the core autocorrelation results:

```matlab
Freud_Main_Block_Analysis

This generates the analyses corresponding to Figure 2.

3. Generate Classification Results

To run classifier comparisons and produce ROC curves:

Freud_Plot_Model_Comparison
Notes
Standardization

Most modeling scripts include a standardize flag.

During cross-validation, standardization uses training-set statistics (mean and standard deviation) to prevent data leakage.

Dependencies

Required MATLAB toolbox:

Statistics and Machine Learning Toolbox

This toolbox is required for the lassoglm implementation used in the Joint model.
