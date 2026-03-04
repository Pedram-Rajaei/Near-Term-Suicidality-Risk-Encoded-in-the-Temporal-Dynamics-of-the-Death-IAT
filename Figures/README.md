# Figures Reproduction Guide

This folder contains all figures used in the **PNAS manuscript** and **Supporting Information (SI)**.

This guide provides step-by-step instructions for reproducing the figures using the MATLAB analysis pipeline included in the repository.

All primary analyses are performed in **MATLAB**, with supplementary baseline comparisons available in the included **Jupyter notebooks**.

---

# Figure 2 — Block-Level Temporal and Rhythmic Structure

Figure 2 characterizes the **serial dependence and rhythmic organization** of reaction-time (RT) dynamics across the BD-IAT.

## Panels 2A & 2B — D-score and ROC

**Purpose**  
Baseline behavioral comparison between **with active-SI** and **without active-SI** participants.

**Script**


Freud_Plot_DScore_ROC.m


**Input**


Freud_Processed_BDIAT.mat


**Output**


Figure_2_A.svg
Figure_2_B.svg


---

## Panel 2C — Block-Level Autocorrelation

**Purpose**  
Visualizes the group-mean autocorrelation function (ACF) curve with SEM bands.

**Script**


Freud_Main_Block_Analysis.m


**Output**


Figure_2_C.svg


---

## Panel 2D — Rhythm Stability (Null Distribution)

**Purpose**  
Permutation-based null test evaluating the **RhythmIndex group difference**.

**Script**


Freud_Main_Block_Analysis.m


Calls:


Freud_Autocorr_Advanced.m


**Output**


Figure_2_D.svg


---

## Panel 2E — Robustness Check (N = 10)

**Purpose**  
Sensitivity analysis using **10-trial block windows**.

**Script**


Freud_Robustness_Check.m


**Input**


Freud_Processed_BDIAT_Short.mat


**Output**


Figure_2_E.svg


---

# Figure 3 — Intra-Block Trial Dynamics

Figure 3 examines **within-block reaction-time dynamics**, capturing the behavioral **warm-up** and **adjustment** phases.

## Panels 3A & 3B — Scaled RT Profiles

**Purpose**

Group-mean temporal profiles for:

- **Death + Me** condition (Panel A)
- **Life + Me** condition (Panel B)

**Script**


Freud_PCA_Trial_Dynamics.m


**Statistical analysis**

Includes **Linear Mixed Effects (LME)** modeling with planned contrasts for **Trials 1–6**.

---

## Panels 3C & 3D — Latent State Eigenvectors (PC1 and PC2)

**Purpose**

Visualizes the first two **latent temporal dimensions** of the RT series.

**Script**


Freud_PCA_Trial_Dynamics.m


**Output**


Figure_3_C.svg
Figure_3_D.svg


---

# Figure 4 — Joint Latent Modeling and Classification

Figure 4 presents the core predictive model: a **sparse bilinear logistic regression classifier** using alternating optimization.

---

## Panel 4A — ROC Comparison (6 Models)

**Purpose**

Benchmarks the **Joint Learned model** against **Fixed-PC baselines** across different latent dimensions.

**Script**


Freud_Plot_Model_Comparison.m


**Output**


Figure_4_A.svg


---

## Panel 4B — Latent Space Embedding

**Purpose**

2D scatter visualization of subjects in the learned latent space with the **decision boundary**.

**Script**


Freud_Plot_Latent_Dynamics.m


**Input**


Freud_Model_J2_Latents.mat


---

## Panels 4C & 4D — Learned Model Weights

**Purpose**

Displays the learned model parameters:

- **b** — sparse temporal weights
- **v** — block-space weights

**Script**


Freud_Plot_Latent_Dynamics.m


**Output**


Figure_4_C.svg
Figure_4_D.svg


---

# Supplementary Information Figures

## Figure S1 — Baseline Model Comparison

**Notebook**


Freud_S1&2.ipynb


**Purpose**

Compares the Joint Latent Model against standard **machine learning baselines using LOOCV**.

---

## Figure S2 — Stimulus Distribution Validation

**Notebook**


Freud_S1&2.ipynb


**Input**


Freud_Trial_Map.xlsx


**Output**

- Heatmap (S2A)
- Monte Carlo goodness-of-fit test (S2B)

This verifies that **stimulus ordering does not confound temporal effects**.

---

## Figure S3 — Clinical Label Sensitivity (MDD vs Control)

**Purpose**

Clinical robustness analysis that replaces the SI labels with **MDD vs Healthy Control**.

This tests whether the **temporal encoding patterns and classification performance remain stable under alternative clinical labels**.

**Scripts**


Freud_PCA_Trial_Dynamics_S3.m
Freud_Plot_Latent_Dynamics_S3.m


**Outputs**


Figure_S3_A.svg
Figure_S3_B.svg
Figure_S3_C.svg


Descriptions:

- **S3A** — Group-mean RT temporal profile under MDD/Control labels  
- **S3B** — p-value and q-value profiles for the Group × Trial interaction  
- **S3C** — ROC comparison of Joint Learned vs Fixed-PC models

---

## Figure S5 — Statistical Significance of Trial Dynamics

**Purpose**

Provides the inferential statistical foundation for **Figure 3 temporal dynamics**.

Maps **trial-by-trial significance** using:

- p-values
- FDR-corrected q-values

**Script**


Freud_PCA_Trial_Dynamics_S5.m


**Outputs**


Figure_S5_A.svg
Figure_S5_B.svg
Figure_S5_C.svg
Figure_S5_D.svg


Descriptions:

- **S5A** — significance for Mean ScaleRT (Death + Me)
- **S5B** — significance for Mean ScaleRT (Life + Me)
- **S5C** — significance for PC1 eigenvector entries
- **S5D** — significance for PC2 eigenvector entries
