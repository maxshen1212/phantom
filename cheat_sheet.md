# Environment Setting

```bash
# conda
conda create -n phantom python=3.10
conda activate phantom
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
export CUDA_VISIBLE_DEVICES=x
echo $CUDA_VISIBLE_DEVICES

# Phantom
cd /data/maxshen/phantom/phantom
conda activate phantom
python process_data.py demo_name=epic mode=all --config-name=epic
python process_data.py demo_name=tri mode=all --config-name=tri
python process_data.py demo_name=PutKiwiInCenterOfTable mode=all --config-name=tri
python process_data.py demo_name=BimanualAppleFromBowlToCuttingBoard mode=all --config-name=tri
python process_data.py demo_name=TurnCupUpsideDown mode=all --config-name=tri
python process_data.py demo_name=TurnMugRightsideUp mode=all --config-name=tri

# (not useful) Lock the package version for testing.
conda activate phantom
pip freeze > /data/maxshen/phantom/requirements_locked.txt
pip install -r requirements_locked.txt
```
