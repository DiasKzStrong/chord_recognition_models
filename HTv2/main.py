from train import run_cross_validation
import torch
import argparse 

def get_args():
    parser = argparse.ArgumentParser("HTv2 arguments")
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--max_n_epochs', type=int, default=50,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience', type=int, default=10,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--experiment_name',type=str, help="Experiment name that you currently running")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    root_dir = "./chord_data_1217"
    args = get_args()
    
    _ = run_cross_validation(
        root_dir=root_dir,
        device=device,  
        args=args
    )