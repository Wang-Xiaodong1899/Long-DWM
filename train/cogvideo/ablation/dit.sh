CUDA_VISIBLE_DEVICES=1 python infer_sft_DiT_only_pipe_long.py --val_s 18 --val_e 36 &
CUDA_VISIBLE_DEVICES=2 python infer_sft_DiT_only_pipe_long.py --val_s 36 --val_e 54 &
CUDA_VISIBLE_DEVICES=3 python infer_sft_DiT_only_pipe_long.py --val_s 54 --val_e 72 &
CUDA_VISIBLE_DEVICES=4 python infer_sft_DiT_only_pipe_long.py --val_s 72 --val_e 90 &
CUDA_VISIBLE_DEVICES=5 python infer_sft_DiT_only_pipe_long.py --val_s 90 --val_e 108 &
CUDA_VISIBLE_DEVICES=6 python infer_sft_DiT_only_pipe_long.py --val_s 108 --val_e 126 &
CUDA_VISIBLE_DEVICES=7 python infer_sft_DiT_only_pipe_long.py --val_s 126 --val_e 150 &