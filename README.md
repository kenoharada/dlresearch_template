# dlresearch_template
## SSH
```bash
ssh -A -L 5900:localhost:5901 -L 8888:localhost:8889 {server_name}
```
## Setup base env 
Check your machine's CUDA version and set version in Dockerfile and base_env.def
### Singularity
```bash
singularity build --fakeroot base_env.sif base_env.def
singularity shell --nv base_env.sif
```

### Docker
```bash
cd dlresearch_template
docker build -t dlresearch .
docker run --gpus all --rm -it --shm-size=48gb -p 5901:5900 -p 8889:8888 --mount type=bind,src=$PWD,dst=/root/dlresearch_template  --name `whoami`_dlresearch dlresearch
```

## Install packages for experiments
```bash
python3 -m venv ~/venv/dmcontrol_env
source ~/venv/dmcontrol_env/bin/activate
pip3 install --upgrade pip
pip3 install jupyterlab wandb matplotlib
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install dm_control==1.0.5
export MUJOCO_GL=egl
export EGL_DEVICE_ID=0
cd dlresearch_template
pip3 install -e .
# Check jupyter lab
# nohup jupyter lab --port 8888 --ip=0.0.0.0 --allow-root >> jupyer.log &
## acess with token via your local machine's browser
# cat jupyer.log | grep 127.0.0.1:8888 | tail -n 1
```

## Check simulator
```bash
export DISPLAY=:0
Xvfb :0 -screen 0 1400x900x24 &
x11vnc -display :0 -forever -noxdamage > /dev/null 2>&1 &
icewm-session &
# You can check remote machine via VNC viewer
python experiments/pointmass/env_check.py
```