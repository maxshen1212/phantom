eval "$(conda shell.bash hook)"
# ######################## Phantom Env ###############################
conda create -n phantom python=3.10 -y
conda activate phantom
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# [Modified] 1. Move PyTorch installation to the front and specify version to avoid Hamer not finding Torch during installation
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# [Fixed] Pre-install opencv with pinned version to avoid dependency conflicts with mmcv
# Old: commented out (let pip resolve => installed 4.13.0.90 as dep, not 4.9.0.80)
pip install opencv-python==4.13.0.90

# [Modified] 2. MMCV installation adjustment
# Old command: pip install mmcv==1.3.9
# New command: (Must remove mmcv-lite, keep only mmcv-full, and force reinstall to ensure correct compilation)
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html --force-reinstall
pip install mmengine==0.10.7

pip install numpy==1.26.4
# [Fixed] Pin iopath instead of using -U (latest), final pin at bottom ensures correctness
pip install iopath==0.1.9

# Install SAM2
cd submodules/sam2
pip install -v -e ".[notebooks]"
cd ../..

# Install Hamer
cd submodules/phantom-hamer
# [Modified] 3. Add --no-build-isolation to solve the issue of not finding PyTorch during installation
# Old command: pip install -e .\[all\]
pip install -e .\[all\] --no-build-isolation

pip install -v -e third-party/ViTPose
wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
cd ../..

# Install phantom-robosuite
cd submodules/phantom-robosuite
pip install -e .
cd ../..

# Install phantom-robomimic
cd submodules/phantom-robomimic
pip install -e .
cd ../..

# Install additional packages (with pinned versions matching locked requirements)
# [Fixed] Added version pins to all previously unpinned packages
pip install joblib==1.5.3 mediapy==1.2.5 open3d==0.19.0 pandas==2.3.3
pip install transformers==4.42.4

# [Modified] 4. Fix PyOpenGL version and add accelerate to resolve EGLDeviceEXT error
# Old command: pip install PyOpenGL==3.1.4
pip install PyOpenGL==3.1.0 PyOpenGL_accelerate==3.1.10

# [Fixed] Pin rtree version
pip install rtree==1.4.1
pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git
pip install protobuf==3.20.0
pip install hydra-core==1.3.2 hydra-colorlog==1.2.0 hydra-submitit-launcher==1.2.0
pip install omegaconf==2.3.0

# Download E2FGVI weights
cd submodules/phantom-E2FGVI/E2FGVI/release_model/
pip install gdown==5.2.1
gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
cd ../..

# Install phantom-E2FGVI
pip install -e .
cd ../..

# Install phantom
pip install -e .


# Download sample data
cd data
# [Modified] 5. Ensure raw folder exists and fix path logic
# Old command: cd data/raw (This line will error if the folder doesn't exist, so changed to mkdir first)
mkdir -p raw
cd raw
wget https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
# [Modified] Add -o parameter to auto-overwrite, avoiding script getting stuck on prompts
# Old command: unzip pick_and_place.zip
unzip -o pick_and_place.zip
rm pick_and_place.zip
wget https://download.cs.stanford.edu/juno/phantom/epic.zip
# Old command: unzip epic.zip
unzip -o epic.zip
rm epic.zip
cd ../..
