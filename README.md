# NeuralTaskonomy
## Inferring the Similarity of Task-Derived Representations from Brain Activity
This repository contains code for Neural Taskonomy paper, accepted to NeurIPS 2019.

Requirements.txt contains the necessary package for to run the code in this project.

In scripts folder,
- generate_taskonomy_features.sh \
This saves the model activation from each task specific models for each images.

- taskrepr_ROI_train.sh \
This runs encoding model (ridge regression) using task presentaitons from all task on ROI data.

- taskrepr_wholebrain.sh \
Same as above but on whole brain data. 

- taskrepr_ROI_permutation.sh \
This permutates the brain response 5000 times to obtain the null distribution.

- taskrepr_wholebrain_permutation.sh \
Same as above but on the whole brain data

In the code folder,

To process permutation results:
```
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
