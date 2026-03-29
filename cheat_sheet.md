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

# (not useful) Lock the package version for testing.
conda activate phantom
pip freeze > /data/maxshen/phantom/requirements_locked.txt
pip install -r requirements_locked.txt

# convert TRI to EPIC
cd /data/maxshen/phantom/TRI_scripts/Max/convert
conda activate tri
python convert_tri_to_epic.py \
  --input-base-dir /data/maxshen/Video_data/LBM_human_egocentric/egoTurnMugRightsideUp \
  --output-base-dir /data/maxshen/phantom/data/raw/TurnMugRightsideUp \
  --language-task-key TurnMugRightsideUp

# convert all processed TRI inpainting data to h5
cd /data/maxshen/phantom/TRI_scripts/Max/convert
conda activate tri
python convert_processed_to_h5_2d.py \
    --processed_dir /data/maxshen/phantom/data/processed/tri \
    --lang_annotations /data/maxshen/phantom/data/language_annotations.yaml \
    --output /data/maxshen/phantom/data/tri_2d_lang.h5 \
    2>&1
```
