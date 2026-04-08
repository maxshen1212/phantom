```bash
# convert TRI to EPIC
conda activate tri
cd /data/maxshen/phantom/TRI_scripts/Max/convert
python convert_tri_to_epic.py \
  --input-base-dir /data/maxshen/Video_data/LBM_human_egocentric/egoTurnMugRightsideUp \
  --output-base-dir /data/maxshen/phantom/data/raw/TurnMugRightsideUp

# convert TRI inpainting data processed by Phantom to h5 format
conda activate phantom
cd /data/maxshen/phantom/TRI_scripts/Max/convert
python convert_processed_to_h5_2d.py \
    --processed_dir /data/maxshen/phantom/data/processed \
    --lang_annotations /data/maxshen/Video_data/language_annotations.yaml \
    --output /data/maxshen/phantom/data/h5output/tri_2d_lang_4tasks_allEpisods.h5 \
    --device cuda \
    --num_workers 64

# convert TRI simulation egocentric data to LeRobot training data
conda activate lbm
cd /data/maxshen/phantom/TRI_scripts/Max/convert
python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data2/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_repo lbm_sim/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
        --output_dir /data/maxshen/lerobot_training_data/ego_BimanualPlaceAppleFromBowlOnCuttingBoard \
        --fps 10

# convert TRI simulation egocentric data to LeRobot training data
conda activate lbm
cd /data/maxshen/phantom/TRI_scripts/Max/convert
python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data2/LBM_sim_egocentric/PutBananaOnSaucer \
        --output_repo lbm_sim/ego_PutBananaOnSaucer \
        --output_dir /data/maxshen/lerobot_training_data/ego_PutBananaOnSaucer \
        --fps 10

# convert TRI simulation egocentric data to LeRobot training data
conda activate lbm
cd /data/maxshen/phantom/TRI_scripts/Max/convert
python convert_tri_to_lerobot.py \
        --input_dir /data/maxshen/Video_data2/LBM_sim_egocentric/PutKiwiInCenterOfTable \
        --output_repo lbm_sim/ego_PutKiwiInCenterOfTable \
        --output_dir /data/maxshen/lerobot_training_data/ego_PutKiwiInCenterOfTable \
        --fps 10


python /data/maxshen/phantom/TRI_scripts/Max/convert/inspect_tri_npz.py /data/maxshen/Video_data2/LBM_sim_egocentric/BimanualPlaceAppleFromBowlOnCuttingBoard/riverway/sim/bc/teleop/2025-01-06T14-24-17-05-00/diffusion_spartan/episode_0/processed --frame 0
```
