# Judicial AI: Data and Code for Exploratory Study on Clinical Decision Support Systems

This repository contains data and code accompanying the paper:

> **Judicial AI**: Mitigating Automation Bias and Preserving Clinical Autonomy through Contrasting Explanations

## Table of Contents
1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Requirements and Installation](#requirements-and-installation)
4. [Data Description](#data-description)
5. [Running the Analysis](#running-the-analysis)
6. [Results](#results)
7. [License](#license)
8. [Contact](#contact)

---

## Overview

Clinical Decision Support Systems (CDSS) that employ machine learning have great potential to improve diagnostic accuracy. However, in certain clinical environments, definitive AI recommendations can introduce automation bias and diminish human sense of agency. 

**Judicial AI** addresses these issues by offering *contrasting explanations* rather than a single definitive recommendation. This repository provides:
- An anonymized dataset of questionnaire responses collected from medical professionals diagnosing vertebral fractures.
- A Python script (`statistical_analysis_survey.py`) demonstrating how we processed and analyzed these data.
- Example statistical tests and visualizations used in the paper.

For more context about the study's design, results, and discussion of findings, please see the published paper (or the pre-print if you have not yet released a final publication).

---

## Repository Structure

```
.
├── data
│   ├── result_survey_preprocessed_final.csv
│   ├── calibration_information.csv
├── statistical_analysis_survey.py
├── README.md
```

- **data/**: Contains the anonymized questionnaire data (`result_survey_preprocessed_final.csv`) and calibration information (`calibration_information.csv`), as well as any additional preprocessed or intermediate files.
- **statistical_analysis_survey.py**: Main Python script demonstrating the preprocessing, merging, statistical tests, and visualizations reported in the paper.
- **README.md**: This file, describing how to use the repository.
- **LICENSE**: (Recommended) A license for your code and data. Please include one that fits your project’s needs (e.g., MIT, Apache, CC-BY for data, etc.).

---

## Requirements and Installation

This code was tested with **Python 3.8+**. Required packages include:

- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [scipy](https://www.scipy.org/)  
- [statsmodels](https://www.statsmodels.org/)  
- [matplotlib](https://matplotlib.org/)  
- [seaborn](https://seaborn.pydata.org/)  
- [scikit-learn](https://scikit-learn.org/) (for resampling utilities)

To install these dependencies:

```bash
pip install -r requirements.txt
```

*(If you prefer conda, create an environment and install packages there.)*

---

## Data Description

- **`result_survey_preprocessed_final.csv`**: Main dataset containing anonymized questionnaire responses:
  - **id**: participant identifier (anonymized integer).
  - **Response**: numeric or categorical response for the respective question.
  - **Question**: question type or category (e.g., `initial_confidence`, `final_confidence`, `utility`, etc.).
  - **id_Caso**: unique identifier of the case/image being evaluated (e.g., `G1`, `G2`, ...).
  
- **`calibration_information.csv`**: Additional table with calibration or reference data for each case.

**NOTE**: All personally identifiable information has been removed to preserve participant anonymity.

---

## Running the Analysis

1. **Ensure the data CSV files** (`result_survey_preprocessed_final.csv`, `calibration_information.csv`) are in the `data/` folder.
2. **Open a terminal** in the repository’s root directory.
3. **Run**:
    ```bash
    python statistical_analysis_survey.py
    ```
4. The script will:
   - Read and merge the CSV data.
   - Perform various statistical tests (e.g., Wilcoxon, McNemar’s, Cliff’s Delta).
   - Produce summary CSV files and any relevant plots in the current directory.

---

## Results

Key summary statistics, effect sizes, and plots will be saved or displayed as the script executes. For instance:
- **`colored_shadow_2_complete.csv`**: Example combined dataset from merging multiple question types.
- Plot images (e.g., `non_inferiority_glass_delta_*.png`, `confidence_judicial.png`, etc.) that illustrate effect sizes and confidence intervals.

If you use these outputs, please reference the paper and cite appropriately.
