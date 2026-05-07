# **Spatiotemporal Dynamics of Neural Representations and Self-Supervised Models**

This repository contains code and data associated with the paper:  
"**Spatiotemporal dynamics of structural and content representations align self-supervised models with the cortical speech network**"  
*(Peili Chen, Shiji Xiang, Linyang He, Edward F. Chang, Yuanning Li)*.  



## **Overview**

This repository provides tools for analyzing the alignment between self-supervised learning (SSL) models and human neural responses during speech perception. The key focus is to:
1. Extract and analyze various linguistic features (acoustic, structural, semantic, and contextual).
2. Reproduce experiments for mapping brain–model alignment using ECoG data.
3. Evaluate temporal dynamics of different feature representations using canonical correlation analysis (CCA) and neural encoding models.

The analysis leverages high-density electrocorticography (ECoG) during naturalistic speech listening, as described in the paper. The methods systematically quantify the contributions of:
- Sequence structure,
- Acoustic-phonetic features,
- Lexical semantics,
- Contextual linguistic information.
## Directory Structure

```plaintext
.
├── brain_encoding/   # Core Python package: feature extraction, neural encoding, clustering
├── data/             # Example datasets and precomputed feature files for demonstrations
├── notebooks/        # Jupyter notebooks for interactive examples and tutorials
├── requirements/     # Dependency and environment setup files
├── .gitignore        # Files ignored by Git
├── README.md         # Project documentation
```

## System Requirements
### Software Dependencies
Python version: Python 3.7
Libraries: see `requirements.txt`

### Hardware Requirements

- **RAM**: 16+ GB
- **CPU**: 4+ cores, 3.3+ GHz/core
- **GPU**: Not required

## Getting Started

To run the demo and reproduce results from the paper, please follow the steps below.

### 1. Create a virtual environment (Recommended)
We strongly recommend creating a virtual Python environment to avoid package conflicts. You can create a virtual environment using the following commands:
```bash
# For Linux or macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
Once the virtual environment is activated, install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements/requirements.txt
```
This will install all necessary Python packages.


### 3. Run Jupyter Notebooks
After installing the dependencies, open the relevant `ipynb` notebook in the `notebooks/` folder to explore or reproduce the various experiments from the paper.

```bash
# To start Jupyter Notebook
jupyter notebook
```

The following notebooks are provided:
- `Feature_extraction.ipynb`: Demonstrates how to extract features from the example dataset.
- `Encoding_example.ipynb`: Illustrates neural encoding model usage on the example dataset.
- `CNMF_clustering.ipynb`: Shows how to cluster electrode encoding scores using CNMF.
- `Extracting_Canonical_Variables.ipynb`: Demonstrates canonical correlation analysis for extracting shared variables.

## Setup and Dependencies

This project requires some external resources. Please follow the instructions below to prepare the necessary files:

**GloVe pre-trained word embeddings**:
- Download the `glove.6B.300d.txt` file from the [official GloVe website](https://nlp.stanford.edu/projects/glove/).
- File details:
    - **Name**: `glove.6B.300d.txt`
    - **Type**: Pre-trained word embeddings (300 dimensions).
    - **Size**: ~1.2 GB after extraction (from the `glove.6B.zip` archive).
- Instructions:
    1. Visit the website [https://nlp.stanford.edu/projects/glove/].
    2. Download the file named `glove.6B.zip`.
    3. Extract the archive and locate `glove.6B.300d.txt`.
    4. Place the file into the `data/` directory of this project.


## Citation

This work is currently under submission for publication. If you find this code useful in your research, please reference it as follows:

```
@unpublished{chen2026spatiotemporal,
  title={Spatiotemporal dynamics of structural and content representations align self-supervised models with the cortical speech network},
  author={Peili Chen and Shiji Xiang and Linyang He and Edward F. Chang and Yuanning Li},
  note={Under submission, 2026},
  institution={ShanghaiTech University, Columbia University, University of California, San Francisco},
}
```

Once the paper is accepted and published, this section will be updated with the final citation information.
