# 激活conda环境 - 使用conda activate命令
source /mnt/cloud-disk/conda-tool/bin/activate searchr1

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
num_gpus=8
data_name="nq_hotpotqa_train_autorefine"
export DATA_DIR="./data/${data_name}"

wandb_token="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"
WAND_PROJECT="AutoRefine"
export WANDB_MODE="online"
export WANDB_API_KEY=$wandb_token
export VLLM_ATTENTION_BACKEND=XFORMERS

export BASE_MODEL='/jizhicfs/zyiii/Model/Qwen2.5-7B'
export EXPERIMENT_NAME="$data_name-autorefine-qwen2.5-7b-again"


mkdir -p log/
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    reward_model.reward_style="F1" \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/valid_500.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=256 \
    data.max_prompt_length=6656 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    max_turns=3 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=false \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.refine_lambda=-1 \
    actor_rollout_ref.actor.refine_score=0.1 \
    actor_rollout_ref.actor.format_score=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_llds=true \
    actor_rollout_ref.actor.llds_coef=0.05 \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=8 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=2 \
    2>&1 | tee log/$EXPERIMENT_NAME.log