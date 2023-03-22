"""
Testing script for DMD
"""
from run import DMD_run

DMD_run(model_name='dmd', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_distill=False)
