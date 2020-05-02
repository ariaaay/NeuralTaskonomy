# NeuralTaskonomy
## Inferring the Similarity of Task-Derived Representations from Brain Activity
This repository contains code for Neural Taskonomy paper, accepted to NeurIPS 2019.

## Setup and Installation
### Step 1: Clone the code from Github
```
git clone https://github.com/ariaaay/NeuralTaskonomy.git
cd NeuralTaskonomy
```
You will also need to clone the [taskonomy](https://github.com/StanfordVL/taskonomy/tree/master/taskbank) model bank.
```
git clone https://github.com/StanfordVL/taskonomy/tree/master/taskbank
```
[BOLD5000](https://bold5000.github.io/) data and stimuli is available for download [here](https://bold5000.github.io/download.html).

### Step 2: Install requirements
[Requirements.txt](https://github.com/ariaaay/NeuralTaskonomy/blob/master/requirements.txt) contains the necessary package for to run the code in this project.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirement.txt --no-index
```
Please also follow [installation page](https://github.com/StanfordVL/taskonomy/tree/master/taskbank#installation) to install another environment to use taskonomy model bank.

### Step 3: Generate model activations from each task specific models for all BOLD5000 Images
```
sh scripts/generate_taskonomy_features.sh
```

### To run encoding models (ridge regression) using task presentaitons from all tasks on ROI data and whole brain data.
```
sh scripts/taskrepr_ROI_train.sh
sh scripts/taskrepr_wholebrain.sh
```

### To run permutatation tests on ROI and whole brain data
```
sh scripts/taskrepr_ROI_permutation.sh
sh scripts/taskrepr_wholebrain_permutation.sh
```
This permutates the brain response 5000 times to obtain the null distribution.

### To process permutation results:
```
cd code
python process_permuation_results.py --subj $subj
python make_task_network.py --use_mask_corr --subj $subj --empirical
python run_significance_test.py --subj $subj --whole_brain --use_empirical_p
```

To generate task similarity matrix:
```
python make_task_matrix.py --method "cosine" --use_mask_corr --empirical
```

To generate task trees:
```
python make_task_tree.py --subj $subj --method masked_corr
```
