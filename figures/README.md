# Figures Reproduction Guide

This directory contains the scripts and supporting data required to reproduce the main-text and supplementary figures for the Freud BD-IAT temporal dynamics manuscript.

The figure-generation workflow is organized around three analysis layers:

1. **Block-level temporal dynamics**
2. **Trial-level temporal dynamics**
3. **Latent model and classifier analyses**

All publication panels are exported as vector-based `.svg` files for manuscript assembly.

---

# Required Software

- MATLAB R2018b or newer
- Statistics and Machine Learning Toolbox

Optional:

- COMPASS State-Space Toolbox  
  Required only if regenerating processed state-space data from raw reaction-time files.

---

# Required Data Files

The main figure scripts assume that the following files are available on the MATLAB path or in the active working directory.

## Core data

```text
Freud_Processed_BDIAT.mat
Freud_Cohort_N80.xlsx
Freud_Trial_Map.xlsx
```

## Cached model and figure data

```text
Freud_Main_Block_Analysis_Results.mat
Freud_Model_J2_Latents.mat
Freud_ROC_Comparison_Data.mat
perm_null_results.mat
Freud_Audit_Task_Switch_Results.mat
```

---

# Main-Text Figures

# Figure 2 — Behavioral Performance and Block-Level Temporal Structure

Figure 2 summarizes baseline behavioral decoding, representative trial traces, block-level rhythmic dynamics, and robustness analyses.

---

## Figure 2A and Figure 2B

**Purpose**

Baseline behavioral and ROC analysis using the conventional D-score/behavioral summary comparison.

**Script**

```matlab
Freud_Plot_DScore_ROC
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_2_A.svg
Figure_2_B.svg
```

---

## Figure 2C

**Purpose**

Representative raw log(RT) traces showing trial-level temporal structure across the full BD-IAT sequence.

**Script**

```matlab
Freud_Plot_Representative_RT_Traces
```

**Input**

```text
Freud_Cohort_N80.xlsx
```

**Outputs**

```text
Figure_2_C_1.svg
Figure_2_C_2.svg
```

**Notes**

- `Figure_2_C_1.svg` uses participant ID `367`.
- `Figure_2_C_2.svg` uses participant ID `341`.
- These traces are generated from raw RT values rather than the processed `XF` matrix.
- Missing or invalid RT values are interpolated before log transformation.

---

## Figure 2D and Figure 2E

**Purpose**

Block-level temporal dynamics and rhythm-structure analysis.

**Script**

```matlab
Freud_Main_Block_Analysis
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_2_D.svg
Figure_2_E.svg
```

**Notes**

`Freud_Main_Block_Analysis.m` now exports only the main-text Figure 2D and Figure 2E panels.

---

## Figure 2F and Figure 2G

**Purpose**

Robustness analyses for the block-level temporal structure.

**Script**

```matlab
Freud_Robustness_Check
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_2_F.svg
Figure_2_G.svg
```

---

# Figure 3 — Trial-Level Temporal Dynamics

Figure 3 evaluates within-block reaction-time dynamics and latent trial-position structure.

---

## Figure 3A–D

**Purpose**

Trial-position dynamics for Death + Me and Life + Me blocks.

**Script**

```matlab
Freud_PCA_Trial_Dynamics
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_3_A.svg
Figure_3_B.svg
Figure_3_C.svg
Figure_3_D.svg
```

# Figure 4 — Latent Model and Classifier Analyses

Figure 4 summarizes the latent classifier, learned low-dimensional embedding, permutation controls, and learned model parameters.

---

## Figure 4B

**Purpose**

ROC comparison of learned and fixed-PC latent classifier variants.

**Script**

```matlab
Freud_Plot_Model_Comparison
```

**Inputs**

```text
Freud_Processed_BDIAT.mat
Freud_Model_CrossVal_Joint.m
Freud_Model_CrossVal_Fixed.m
```

**Outputs**

```text
Figure_4_B.svg
Freud_Model_J2_Latents.mat
Freud_ROC_Comparison_Data.mat
```

---

## Figure 4C, Figure 4E, and Figure 4F

**Purpose**

Visualization of learned latent dynamics, block-space projection vectors, and trial-position weights.

**Script**

```matlab
Freud_Plot_Latent_Dynamics
```

**Input**

```text
Freud_Model_J2_Latents.mat
```

**Outputs**

```text
Figure_4_C.svg
Figure_4_E.svg
Figure_4_F.svg
```

## Figure 4D

**Purpose**

Permutation and null-distribution analysis for the learned bilinear classifier.

**Script**

```matlab
Freud_Plot_Permutation_Null
```

**Input**

```text
perm_null_results.mat
```

**Outputs**

```text
Figure_4_D_1.svg
Figure_4_D_2.svg
Figure_4_D_3.svg
```

# Supplementary Figures

# Figure S1 — Rhythm-Index Robustness Analysis

**Purpose**

Evaluates whether the observed rhythm index exceeds participant-specific null expectations under shuffle controls.

**Script**

```matlab
Freud_RhythmIndex_S6_Robustness
```

**Inputs**

```text
Freud_Processed_BDIAT.mat
Freud_Processed_BDIAT_Short.mat
```

**Outputs**

```text
Figure_S1_A.svg
Figure_S1_B.svg
Figure_S1_C.svg
Figure_S1_D.svg
rhythm_s1_results.mat
```
---

# Figure S2 — PC1 and PC2 Loading-Entry Tests

**Purpose**

Trial-wise statistical tests of PC1 and PC2 loading entries within the Death + Me condition.

**Script**

```matlab
Freud_PCA_Loading_Tests_S4
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_S2_A.svg
Figure_S2_B.svg
```
---
# Figure S3 — Stimulus Distribution Validation

**Purpose**

Validates that stimulus ordering and trial-position structure do not confound the temporal dynamics results.

**Notebook**

```text
Freud_S3&4.ipynb
```

**Input**

```text
Freud_Trial_Map.xlsx
```

**Outputs**

Figure S3 includes stimulus-position heatmaps and Monte Carlo validation analyses.

---
# Figure S4 — Baseline Model Comparison

**Purpose**

Compares the latent temporal classifier against standard machine-learning baselines under leave-one-out cross-validation.

**Notebook**

```text
Freud_S3&4.ipynb
```
---

# Figure S5 — Diagnostic-Group Control Analysis

Figure S5 evaluates whether the trial-level and classifier results are primarily explained by MDD/control diagnostic grouping rather than the SI grouping used in the main analysis.

---

## Figure S5A and Figure S5B

**Script**

```matlab
Freud_PCA_Trial_Dynamics_S5
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_S5_A.svg
Figure_S5_B.svg
```

---

## Figure S5C 

**Purpose**

ROC analysis when the bilinear classifier is evaluated under the Figure S5 grouping convention.

**Script**

```matlab
Freud_Plot_Model_Comparison_S5
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_S3_C.svg
Freud_Model_J2_Latents_S3.mat
Freud_ROC_Comparison_Data_S3.mat
```

---

# Figure S6 — Trial-Wise Significance of Temporal Dynamics

**Purpose**

Provides the inferential statistical support for trial-level temporal dynamics in Figure 3.

**Script**

```matlab
Freud_PCA_Trial_Dynamics_S6
```

**Input**

```text
Freud_Processed_BDIAT.mat
```

**Outputs**

```text
Figure_S6_A.svg
Figure_S6_B.svg
Figure_S6_C.svg
Figure_S6_D.svg
```
---
