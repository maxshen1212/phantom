# Environment Setting

```bash
# conda
conda create -n phantom python=3.10
conda activate phantom
conda install pytorch torchvision torchaudio
conda env list
conda deactivate
conda remove -n phantom --all

# tmux
tmux new -s window_name
crrl + b, d
tmux kill-session -t window_name
tmux ls
tmux a -t window_name

# utilities
rsync -av --progress a b
watch -n 1 nvidia-smi
nvidia-smi
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES

# Phantom
cd phantom/phantom
python process_data.py demo_name=epic mode=all --config-name=epic
python process_data.py demo_name=tri mode=all --config-name=tri

# MCR
# TODO: change the parameters
# Refer to train_mcr.sh to launch the training and override hyper-paremeters.

CUDA_VISIBLE_DEVICES=x,x python train_representation.py hydra/launcher=local \
        hydra/output=local agent.decode_state_weight=1.0 agent.size=50 experiment=<exp_name> \
        doaug=rctraj batch_size=32 datapath=<path_to_dataset> \
        wandbuser=<your_username> wandbproject=<your_proj> use_wandb=false

conda run -n phantom pip freeze > /data/maxshen/phantom/requirements_locked.txt
pip install -r requirements_locked.txt
```
