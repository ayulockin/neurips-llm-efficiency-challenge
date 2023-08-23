# NeurIPS Large Language Model Efficiency Challenge: 1 LLM + 1GPU + 1Day

This repository showcase a working pipeline that can be used for the [NeurIPS Large Language Model Efficiency Challenge: 1 LLM + 1GPU + 1Day](https://llm-efficiency-challenge.github.io/index) challenge. The aim of this challenge is to come up with strategies to finetune a large language model either on 1 Nvidia 4090 or 1 A100 (40 GB) inorder to improve the performance on a subset of HELM benchmark and on a set of secret holdout tasks.

Coming up with strategies is one thing but one needs to first have a working pipeline to fit atleast a 7B LLM in one GPU. This itself is a challenge. This repo is created to outline the setup and gotchas and is built on the shoulder of giants.

# Setup

The most important thing is to get your hands on a A100 (40 GB) or a 4090. Note that in this challenege these two are separate tracks.

### 1. Create a fresh environment

I will be using conda but you can use your choice of environment management tool.

```
conda create -n neurips-llm python==3.10.0
```

### 2. Clone this repository

```
git clone --recurse-submodule https://github.com/ayulockin/neurips-llm-efficiency-challenge
```

Note the usage of `--recurse-submodule` here. The repo uses my [fork](https://github.com/ayulockin/lit-gpt) of [`lit-gpt`](https://github.com/Lightning-AI/lit-gpt) as a submodule. My fork is instrumented with [Weights and Biases](https://wandb.ai/site) experiment tracking and model versioning capabilities. This is also in sync with the upstream repo (lit-gpt). 

### 3. Install PyTorch 2.1 (nightly)

`lit-gpt` requires PyTorch 2.1 (nightly) to work. Let's install this:

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

This will download CUDA driver version 11.8 if it is compatible with your installed CUDA runtime version. We will revisit CUDA again.

### 4. Install other libraries

Let's install other requirements by lit-gpt

```
pip install -r requirements.txt tokenizers sentencepiece huggingface_hub
```

# CUDA gotchas and Flash Attention

In order to finetune a large model (7B) efficiently, [flash attention](https://github.com/Dao-AILab/flash-attention) is a must imo. In this section we will install flash attention 2.0. 

This library requires CUDA runtime >= 11.4 and PyTorch >= 1.12.

<details open>
<summary>Resolving CUDA</summary>
<br>
The CUDA runtime and driver are two different APIs. You can check the runtime version using `nvcc --version` and the driver version using `nvidia-smi`.

If your runtime is less than 11.4, you need to update it to 11.4 or above. This runtime is also dependent on the OS (eg: Debian 10 supports till 11.4).

If you have to update the cuda runtime, follow the steps:


1. Remove cuda from your system.
```
sudo apt-get --purge remove "cublas*" "cuda*"
```

2. Google Nvidia Toolkit 11.x download. You will find the appropriate url with steps listed there. In my case, I was on Debian 10 and thus could install 11.4. The official instructions page for CUDA 11.4 can be found [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local).

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-debian10-11-4-local/7fa2af80.pub
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```
</details>

To download Flash Attention here are the required steps:

```
pip install packaging
pip install packaging uninstall -y ninja && pip install ninja

MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

> An A100 will typically come with 83.6 GB of usable RAM. `ninja` will do parallel compilation jobs that could exhaust the amount of RAM. Set the max number of jobs using `MAX_JOBS`. A `MAX_JOBS=4` will take 30+ minutes to compile flash attention, while `MAX_JOBS=8` might take 20ish minutes (with 35ish GB of RAM usage). On an A100, `MAX_JOBS` of 16 might work (haven't tested).

# Data



generate

```
python lit-gpt/generate/base.py --checkpoint_dir /home/ayushthakur/llm/neurips-llm-efficiency-challenge/checkpoints/tiiuae/falcon-7b --prompt "Tell me an interesting fun fact about earth:"
```

fine-tune

```
python lit-gpt/finetune/lora.py --data_dir data/dolly/tiiuae/falcon-7b --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --out_dir out/lora/falcon-7b
```


Update CUDA runtime to 11.4 or above

```

```

Flash Attention



Prepare data

```
python lit-gpt/scripts/prepare_dolly.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```

Download the model checkpoint

```
python lit-gpt/scripts/download.py --repo_id openlm-research/open_llama_3b --token <HuggingFace Token>
python lit-gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```


ElutherAI Eval Harness

```
python lit-gpt/eval/lm_eval_harness.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b --precision "bf16-true" --eval_tasks [truthfulqa_mc --batch_size 4 --save_filepath "results-openllama-3b.json"
```

Run docker server
```
sudo docker run --rm --gpus all -p 9000:80 toy_submission
```

nvidia container toolkit

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

docker build

```
sudo docker build -t toy_submission . 
```