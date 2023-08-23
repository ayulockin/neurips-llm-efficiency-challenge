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
cd neurips-llm-efficiency-challenge
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
cd lit-gpt
pip install -r requirements.txt tokenizers sentencepiece huggingface_hub
cd ..
```

# CUDA gotchas and Flash Attention

In order to finetune a large model (7B) efficiently, [flash attention](https://github.com/Dao-AILab/flash-attention) is a must imo. In this section we will install flash attention 2.0. 

This library requires CUDA runtime >= 11.4 and PyTorch >= 1.12.

<details>
<summary>Update CUDA to 11.4 or above</summary>
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

# Model

We start by downloading the model and preparing the model so that it can be consumed by `lit-gpt`'s finetuning pipeline.

```
python lit-gpt/scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --token <HuggingFace Token>
python lit-gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

> Note: In order to use Llama 2 model you need to fill a small form in the HuggingFace [model card page](https://huggingface.co/meta-llama/Llama-2-7b-hf) for this model. The permission is usually granted in 1-2 days. Tip: Please provide the same email id that you used to create your HuggingFace account.

> Note: You can generate your HuggingFace access token [here](https://huggingface.co/settings/tokens).

# Data

Download the dataset and prepare it using a convenient script provided by `lit-gpt`. Below I am downloading the [`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

```
python lit-gpt/scripts/prepare_dolly.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

> Note: The tokenizer used by the model checkpoint is used to tokenize the dataset. The dataset will be split into train and test set in the `.pt` format.

You are not only limited to the `databricks-dolly-15k` dataset. You can also download and prepare [`RedPajama-Data-1T`](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) dataset.

```
python lit-gpt/scripts/prepare_redpajama.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```

Follow [these steps](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/finetune_lora.md#tune-on-your-dataset) to create your own data preparation script.

> Tip: You will ideally want to combine datasets from varying benchmarks and sample them properly.

## Validate your setup

At this point, before going ahead, let's validate if our setup is working.

```
python lit-gpt/generate/base.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --prompt "Tell me an interesting fun fact about earth:"
```

# Finetune

`lit-gpt` provides a few resource constrainted finetuning strategies like lora, qlora, etc., out of the box.

1. LoRA finetuning

```
lit-gpt/finetune/lora.py --data_dir data/dolly/meta-llama/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision bf16-true --out_dir out/lora/llama-2-7b
```

2. QLoRA finetuning

To finetune with QLoRA, you will have to install the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library.

```
pip install bitsandbytes
```

Finetune with QLoRA by passing the `--quantize` flag to the `lora.py` script

```
lit-gpt/finetune/lora.py --data_dir data/dolly/meta-llama/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision bf16-true --out_dir out/lora/llama-2-7b --quantize "bnb.nf4"
```

# Evaluation (EluetherAI LM Eval Harness)

Once you have a working setup for finetuing, coming up with finetuing strategies is going to be one of the most important task but this will be guided by an even bigger task - thinking through the evaluation strategy.

As per the organizers of this competition:

> The evaluation process in our competition will be conducted in two stages. In the first stage, we will run a subset of HELM benchmark along with a set of secret holdout tasks. The holdout tasks will consist of logic reasoning type of multiple-choice Q&A scenarios as well as conversational chat tasks. Submissions will be ranked based on their performance across all tasks. The ranking will be determined by the geometric mean across all evaluation tasks.

We cannot do anything about the secret hold out tasks. But we can try to improve the finetuned model on a subset of the HELM benchmark.

We can also consider using other benchmarks like EluetherAI's [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). You can find the tasks available in this benchmark [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md). A few tasks here can be considered your validation set since there is an overlap between `lm-evaluation-harness` and HELM.

Install this library and perform evaluation with the base checkpoint and the LoRA finetuned checkpoint.

1. Install `lm-evaluation-harness`

```
cd ..
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd neurips-llm-efficiency-challenge
```

2. Evaluate using the base checkpoint

```
python lit-gpt/eval/lm_eval_harness.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b.json"
```

3. Evaluate the finetuned checkpoint

```
python lit-gpt/eval/lm_eval_harness_lora.py --lora_path out/lora/llama-2-7b/lit_model_lora_finetuned.pth --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b.json"
```

> Here `out/lora/<model-name>/lit_model_lora_finetuned.pth` is something we get after finetuning the model.

> Notice that for base model evaluation we used the `lm_eval_harness.py` script while for finetuned model evaluation we are using `lm_eval_harness_lora.py`. If you are using other strategy, you might have to modify the scipt to cater to your strategy.

# Setting up submission pipeline

The organizers of this challenge, have provided an useful toy example to demonstrate the submission steps. Check out the official repo [here](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/blob/master/toy-submission/README.md).

I have copied the files required to setup the submisison pipeline to this repo to simplify things.

### Installing Nvidia Container Toolkit

There are a lot of details mentioned in the [official documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to install Nvidia container toolkit.

Here are the steps that worked for me:


Setup the GPG key (don't ignore this step)

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
```

Run the following commands:

```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build the docker image and run

```
docker build -t submission .
sudo docker run --rm --gpus all -p 8080:80 toy_submission
```

If everything works out, you will have a successful running HTTP server. The steps mentioned in the `Dockerfile` should be self satisfactory.

# Evaluation (HELM)

A subset (unknown) will be used for the leaderboard evaluation. We can choose to use the [official HELM repository](https://github.com/stanford-crfm/helm) but it is not configured to hit the endpoints of the HTTP server. We will instead be using this [fork](https://github.com/drisspg/helm/tree/neruips_client) (`neruips_client` branch - note the typo) of the HELM repo.

```
cd ..
git clone https://github.com/drisspg/helm.git@neruips_client
```

Create a new environment because this repo uses different versions of multiple repositories.

```
conda create -n helm-eval python==3.10.0
conda activate helm-eval
cd helm
pip install -e .
```

**Note**: If 8080 is already in use, do the following: change the base url that's hardcoded from `http://localhost:8080` (to avoid conflict with Jupyter Notebook, most probable) to `http://localhost:9000` in line 30 of this file - `helm/src/helm/proxy/clients/http_model_client.py`.

Run the following lines to benchmark on the `mmlu` (subset of HELM) benchmark:

```
cd ..
cd neurips-llm-efficiency-challenge
echo 'entries: [{description: "mmlu:model=neurips/local,subject=college_computer_science", priority: 4}]' > run_specs.conf
helm-run --conf-paths run_specs.conf --suite v1 --max-eval-instances 1000
helm-summarize --suite v1
```

Check out the various benchmarks that are present in this benchmark [here](https://crfm.stanford.edu/helm/latest/)

You might encounter this issue - `OSError: /opt/conda/envs/helm-eval/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.11: symbol cublasLtGetStatusString version libcublasLt.so.11 not defined in file libcublasLt.so.11 with link time reference`. If so do the following steps:

To solve this check out this [Stack OverFlow answer](https://stackoverflow.com/a/74828501/8663152).

After successful evaluation, you can find the results in the `benchmark_output` directory. I am working on logging the results to Weights and Biases for easy comparison and tracking.

# Final Thoughts

I hope the documented steps will expedite the setting up process so that more time can be spent on doing ML.

The steps were tested on a GCP Compute Engine VM with A100 (40 GB) GPU. If you have a different setup and if something doesn't work, feel free to open an issue or raise a PR.
