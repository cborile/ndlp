python run.py -m experiment=citeseer_mplp_baseline random_seed=42,43,44,45,46 model.optimizer.lr=0.005  
python run.py -m experiment=citeseer_mplp_scalarization random_seed=42,43,44,45,46 model.optimizer.lr=0.005
python run.py -m experiment=citeseer_mplp_multiclass random_seed=42,43,44,45,46 model.optimizer.lr=0.005 