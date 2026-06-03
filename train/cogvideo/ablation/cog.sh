CUDA_VISIBLE_DEVICES=1 python examples/cogvideo/ablation/infer_sft_CogI2V_pipe_long.py --val_s 18 --val_e 36 &
CUDA_VISIBLE_DEVICES=2 python examples/cogvideo/ablation/infer_sft_CogI2V_pipe_long.py --val_s 36 --val_e 54 &
CUDA_VISIBLE_DEVICES=6 python examples/cogvideo/ablation/infer_sft_CogI2V_pipe_long.py --val_s 54 --val_e 72 &