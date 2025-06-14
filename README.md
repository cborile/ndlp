# Neural Directed Link Prediction

Pytorch and Hydra-based implementation of the code for the paper "Multi-Class and Multi-Task Strategies for Neural Directed Link Prediction", ECMLPKDD 2025 research track [Arxiv](https://arxiv.org/abs/2412.10895)

Example for running the code for multiple experiments using Hydra for logging and parallelization:
'''
python run.py -m experiment=citeseer_mplp_baseline random_seed=42,43,44,45,46 model.optimizer.lr=0.005
'''
