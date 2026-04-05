# TDP43-RNA-binding-ML-structure

\# Machine Learning–Guided Structural Analysis of TDP-43 RNA Binding



\## Overview

TDP-43 is an RNA-binding protein associated with ALS.  

This project aims to understand RNA binding specificity using machine learning and structural bioinformatics.



The workflow combines:

\- eCLIP binding data

\- Machine learning models (Logistic Regression \& CNN)

\- Motif discovery

\- Structural analysis (ongoing)



\---



\## Objectives

\- Identify sequence features driving TDP-43 RNA binding  

\- Compare classical and deep learning approaches  

\- Extract biologically meaningful motifs  

\- Investigate structural relevance of motifs  



\---



\## Dataset

\- \~19,478 positive eCLIP peaks  

\- Matched negative sequences  

\- Total \~38,956 sequences  



\---



\## Models



\### Logistic Regression

\- Uses sequence composition + k-mer features  

\- AUC ≈ 0.98  



\### CNN

\- Learns motifs directly from sequences  

\- AUC ≈ 0.99  



\---



\## Motif Discovery

CNN interpretation identified candidate binding motifs, including:



\- GAGUAUG  

\- GUGCACG  

\- UCUGUAC  

\- UUUGCAU  

\- GCAUAUC  



\---



\## Current Status

\- Data preprocessing ✅  

\- ML modelling ✅  

\- Motif extraction ✅  

\- Structural analysis ⏳ (in progress)  



\---



\## Repository Structure

├── scripts/ # Model training and analysis code

├── results/ # ML outputs and predictions

├── models/ # Trained models

├── motif\_files/ # Extracted motif structures

├── docking/ # Docking inputs and results

├── figures/ # Plots and visualizations

├── docs/ # Project reports





\---



\## ⚙️ Tools \& Technologies

\- Python (scikit-learn, PyTorch)

\- ViennaRNA (RNAfold)

\- RNAComposer

\- HDOCK

\- PyMOL



\---

\## ⚠️ Notes

\- Large raw datasets and genome annotation files are not included due to GitHub size limits

\- Only processed data and essential outputs are provided



\---



\## 🚀 Future Work

\- Dock remaining motifs and compare binding consistency

\- Quantitative comparison of docking scores across motifs

\- Integrate structural features into ML models

\- Explore disease-associated mutations in TDP-43



\---



