from train import run_cross_validation
import torch
import argparse 
import os
import torch.distributed as dist

def get_args():
    parser = argparse.ArgumentParser("HTv2 arguments")
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=3e-2)
    parser.add_argument('--max_n_epochs', type=int, default=50,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience', type=int, default=10,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--experiment_name', type=str, default="accuracy_run",
                        help="Experiment name that you currently running")
    parser.add_argument('--n_steps', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--window_mode', type=str, default="sliding", choices=["sliding", "random_song"],
                        help="sliding uses overlapping windows; random_song samples one random segment per train song per epoch.")
    parser.add_argument('--label_mode', type=str, default="quality27",
                        choices=["quality27", "full_chord", "structured_full_chord"],
                        help="quality27 predicts rootless chord qualities; full_chord uses a flat 301-way head; structured_full_chord uses six component heads decoded into 301 chords.")
    parser.add_argument('--embed_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--class_weighting', type=str, default="inverse_sqrt",
                        choices=["none", "inverse_sqrt"])
    parser.add_argument('--checkpoint_metric', type=str, default="chord_acc",
                        choices=["chord_acc", "macro_chord_acc"])
    parser.add_argument('--min_delta', type=float, default=1e-4)

    parser.add_argument('--change_loss_weight', type=float, default=0.1)
    parser.add_argument('--max_change_pos_weight', type=float, default=50.0)
    parser.add_argument('--boundary_teacher_forcing_epochs', type=int, default=15)

    parser.add_argument('--augment', action="store_true")
    parser.add_argument('--noise_std', type=float, default=0.01)
    parser.add_argument('--gain_min', type=float, default=0.9)
    parser.add_argument('--gain_max', type=float, default=1.1)
    parser.add_argument('--time_mask_width', type=int, default=8)
    parser.add_argument('--freq_mask_width', type=int, default=12)
    parser.add_argument('--pitch_shift_bins', type=int, default=0)
    parser.add_argument('--pitch_shift_semitones', type=int, default=0,
                        help="Train-only CQT roll in semitones when --augment is enabled; labels are unchanged for quality-only targets.")
    parser.add_argument('--use_signal_decay', action="store_true",
                        help="Apply train-only linear signal decay when --augment is enabled.")
    parser.add_argument('--signal_decay_min', type=float, default=0.4,
                        help="Minimum final-frame gain for signal decay.")
    parser.add_argument('--signal_decay_max', type=float, default=0.9,
                        help="Maximum final-frame gain for signal decay.")

    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold_ids', type=str, default="",
                        help="Comma-separated fold ids, e.g. '0' or '0,1'. Overrides --num_folds.")
    parser.add_argument('--paper_compare', action="store_true",
                        help="Print ChordFormer-style accframe/accclass aliases and paper reference numbers.")
    parser.add_argument('--root_dir', type=str, default=None)
    return parser.parse_args()


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, world_size, rank, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    distributed, world_size, rank, local_rank, device = init_distributed()

    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if rank == 0:
            print(f"Device : {device}")
            if distributed:
                print(f"Distributed : world_size={world_size}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = args.root_dir or os.path.join(script_dir, "chord_data_1217")

        _ = run_cross_validation(
            root_dir=root_dir,
            device=device,
            args=args
        )
    finally:
        cleanup_distributed()
