# neurips-llm-efficiency-challenge


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

generate

```
python lit-gpt/generate/base.py --checkpoint_dir /home/ayushthakur/llm/neurips-llm-efficiency-challenge/checkpoints/tiiuae/falcon-7b --prompt "Tell me an interesting fun fact about earth:"
```

fine-tune

```
python lit-gpt/finetune/lora.py --data_dir data/dolly/tiiuae/falcon-7b --checkpoint_dir checkpoints/tiiuae/falcon-7b --precision bf16-true --out_dir out/lora/falcon-7b
```

ElutherAI Eval Harness

```
python lit-gpt/eval/lm_eval_harness.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b --precision "bf16-true" --eval_tasks [truthfulqa_mc --batch_size 4 --save_filepath "results-openllama-3b.json"
```

Basic installation

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121


```


Update CUDA runtime to 11.4 or above

```
sudo apt-get --purge remove "cublas*" "cuda*"
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-debian10-11-4-local/7fa2af80.pub
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```

Flash Attention

```
pip install packaging
pip install packaging uninstall -y ninja && pip install ninja
MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

Prepare data

```
python lit-gpt/scripts/prepare_dolly.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```

Download the model checkpoint

```
python lit-gpt/scripts/download.py --repo_id openlm-research/open_llama_3b --token <HuggingFace Token>
python lit-gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
```

Setup Repo

```
git clone --recurse-submodule https://github.com/ayulockin/neurips-llm-efficiency-challenge
```
