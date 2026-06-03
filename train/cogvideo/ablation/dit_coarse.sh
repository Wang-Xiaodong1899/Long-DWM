CUDA_VISIBLE_DEVICES=0 python infer_sft_DiT_coarse_pipe_long.py --val_s 3 --val_e 18 &
CUDA_VISIBLE_DEVICES=1 python infer_sft_DiT_coarse_pipe_long.py --val_s 21 --val_e 36 &
CUDA_VISIBLE_DEVICES=2 python infer_sft_DiT_coarse_pipe_long.py --val_s 39 --val_e 54 &
CUDA_VISIBLE_DEVICES=3 python infer_sft_DiT_coarse_pipe_long.py --val_s 57 --val_e 72 &
CUDA_VISIBLE_DEVICES=4 python infer_sft_DiT_coarse_pipe_long.py --val_s 75 --val_e 90 &
CUDA_VISIBLE_DEVICES=5 python infer_sft_DiT_coarse_pipe_long.py --val_s 93 --val_e 108 &
CUDA_VISIBLE_DEVICES=6 python infer_sft_DiT_coarse_pipe_long.py --val_s 111 --val_e 126 &
CUDA_VISIBLE_DEVICES=7 python infer_sft_DiT_coarse_pipe_long.py --val_s 129 --val_e 150 &