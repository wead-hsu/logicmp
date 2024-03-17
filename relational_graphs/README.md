## Requirements
- python 3.7
- pytorch 1.1
- scikit-learn
- networkx
- tqdm
- opt_einsum

## Quick Start
The following command starts the inference on the UW-CSE/ai dataset on GPU:
```
python -u -m main.train_semisupervised_sampling_dense -data_root data/uw_cse/ai/ -slice_dim 8 -batchsize 1024 -use_gcn 0 -embedding_size 128 -gcn_free_size 64 -load_method 0 -exp_folder exp -exp_name tmp -device cuda
