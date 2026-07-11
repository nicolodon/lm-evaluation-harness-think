#!/bin/bash
#SBATCH --job-name=mmlu_GRM_vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:3
#SBATCH --partition=a100
#SBATCH --output=/scratch.hpc/nicolo.donati11/results_mmlu_pro/logs/grm_vllm_%N_%j.log
#SBATCH --error=/scratch.hpc/nicolo.donati11/results_mmlu_pro/logs/grm_vllm_%N_%j.err
#SBATCH --chdir=/scratch.hpc/nicolo.donati11/

# ── GRM-2.5 / GRM-2.6-Plus with vLLM tensor parallelism across 3x A100 GPUs ──
# vLLM splits the model layers across GPUs using tensor parallelism,
# giving a real speedup (not pipeline parallelism).
# Change MODEL below to switch between GRM-2.5 and GRM-2.6-Plus.

MODEL="OrionLLM/GRM-2.5"
# MODEL="OrionLLM/GRM-2.6-Plus"

export APPTAINER_CACHEDIR=/scratch.hpc/nicolo.donati11/apptainer_cache
export HF_HOME=/scratch.hpc/nicolo.donati11/hf_cache
export HF_TOKEN=${HF_TOKEN:-"YOUR_HF_TOKEN_HERE"}
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /scratch.hpc/nicolo.donati11/results_mmlu_pro/logs

echo "=== Job started at $(date) ==="
echo "Running model: ${MODEL} with vLLM (tensor_parallel_size=3) on $(hostname)"
nvidia-smi

SUBCATEGORIES=(
    "computer_science" "biology" "business" "chemistry" "economics"
    "engineering" "health" "history" "law" "math" "other"
    "philosophy" "physics" "psychology"
)

# Derive a safe directory name from the model name
MODEL_DIR=$(echo "${MODEL}" | tr '/' '_')

for CAT in "${SUBCATEGORIES[@]}"; do
    echo "=========================================================="
    echo "  Evaluating: mmlu_pro_${CAT}"
    echo "=========================================================="
    
    apptainer run --nv \
        --bind /scratch.hpc/nicolo.donati11:/scratch.hpc/nicolo.donati11 \
        /scratch.hpc/nicolo.donati11/lm_eval_venv.sif \
        --model vllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,tensor_parallel_size=3,max_model_len=8192,think_start_token="<think>",think_end_token="</think>",enable_thinking=True \
        --gen_kwargs "temperature=1,do_sample=True,top_p=0.95,top_k=20,max_gen_toks=2048" \
        --apply_chat_template \
        --tasks "mmlu_pro_${CAT}" \
        --batch_size auto \
        --output_path "/scratch.hpc/nicolo.donati11/results_mmlu_pro/${MODEL_DIR}/${CAT}" \
        --log_samples
        
    echo "Finished mmlu_pro_${CAT} at $(date)"
    echo "Saved results in /scratch.hpc/nicolo.donati11/results_mmlu_pro/${MODEL_DIR}/${CAT}"
done

echo "=== All evaluations finished at $(date) ==="
