# Acoustic Scattering AI for Noninvasive Object Classifications: A case study on hair assessment

## 1. Install
- Clone the repo

```bash
git clone https://github.com/dangkhoadl/ICASSP2024-Acoustic_Scattering_AI-Noninvasive_Object_Classifications.git
```

- Install Conda: please read https://docs.conda.io/en/latest/miniconda.html


- Create conda env

```bash
cd ICASSP2024-Acoustic_Scattering_AI-Noninvasive_Object_Classifications
conda create -n dl-audio python=3.8
conda activate dl-audio
pip install -r requirements.txt
```


## 2. Download
- Download datasets and metadata: https://huggingface.co/datasets/dangkhoadl/ICASSP2024-Acoustic_Scattering_AI-Noninvasive_Object_Classifications/tree/main
- The file `datasets.tar.gz` contains `metadata` and `DATA`. After downloading, extract the tarball into repos's main directory
- Download pretrained:

```bash
./1_download_pretrained.sh
```

## 3. Run the experiments
- `2_run_exps.sh` is the main recipe
    - Configure variables in `2_run_exps.sh` to run your preferred experiments.
    - Configure the model hyperparameters with `conf/*.yaml`
    - Configure the feature extraction paprameters with `src/**/*config.json`
- We adhere to kaldi format for workspace organization.  Upon running the recipe, you can find all experiment's models and results in `exp/*`.
- We suggest the code snippets in `exp-{1,2}.ipynb` to plot and analyse model's accuracy, confusion mtrix, auc, and other relevant metrics.
