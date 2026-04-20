from train import run_cross_validation
import torch
import argparse 
import os

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
    parser.add_argument('--root_dir', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = args.root_dir or os.path.join(script_dir, "chord_data_1217")
    
    _ = run_cross_validation(
        root_dir=root_dir,
        device=device,  
        args=args
    )
