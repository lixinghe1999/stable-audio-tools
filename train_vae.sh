# To use TensorBoard instead of wandb, add: --logger tensorboard
# Then visualize with: tensorboard --logdir ./checkpoints
torchrun --nproc_per_node=2 \
  train.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_3_0_vae.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/dataset_train.json \
  --val-dataset-config stable_audio_tools/configs/dataset_configs/dataset_val.json \
  --strategy ddp_find_unused_parameters_true \
  --batch-size 4 \
  --num-workers 8 \
  --checkpoint-every 2000 \
  --save-dir ./checkpoints \
  --name stable_audio_3_vae_ft \