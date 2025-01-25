import argparse
import torch
from torch.utils.data import DataLoader
from utils import MambaBlock, TransformerModule, UpSampleFunction, DownSampleFunction
from model import DiffusionModel
from Dataload import MedicalImageDataset
from RadiomicsExtractor import load_radiomics_features
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os
import numpy as np

def calculate_metrics(pred, target):
    """Calculate PSNR and SSIM between the predicted and target images."""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    psnr = peak_signal_noise_ratio(target_np, pred_np, data_range=target_np.max() - target_np.min())
    ssim = structural_similarity(target_np, pred_np, data_range=target_np.max() - target_np.min())

    return psnr, ssim

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    dataset = MedicalImageDataset(args.target_dir, args.condition_dir, args.mask_dir)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load radiomics features
    radiomics_features = load_radiomics_features(args.condition_dir, args.mask_dir)
    radiomics_features = torch.tensor(radiomics_features, dtype=torch.float32).to(device)

    fold_psnr_results = []
    fold_ssim_results = []

    # Iterate over each fold's model
    for fold in range(args.num_folds):
        fold_model_path = os.path.join(args.model_dir, f"riven_mcdm_model_fold_{fold + 1}.pth")
        if not os.path.exists(fold_model_path):
            print(f"Model for fold {fold + 1} not found, skipping...")
            continue

        # Initialize the model
        model_structure = nn.Identity()  # Replace with your actual model structure if needed
        model = DiffusionModel(num_timesteps=args.T, model=model_structure).to(device)
        
        # Load the pretrained model weights for the current fold
        model.load_state_dict(torch.load(fold_model_path, map_location=device))
        model.eval()

        # Initialize the required modules
        mamba_blocks = [MambaBlock(args.input_dim, args.hidden_dim) for _ in range(8)]
        transformer = TransformerModule(args.img_feature_dim, args.radiomics_feature_dim, args.num_heads, args.num_layers)
        upsample = UpSampleFunction(scale_factor=2)
        downsample = DownSampleFunction(scale_factor=0.5)

        total_psnr = 0
        total_ssim = 0
        num_samples = 0

        # Testing phase for the current fold
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                x_0 = batch['target'].to(device)
                X_C = batch['condition'].to(device)

                # Extract the corresponding radiomics features (R) for the current batch
                R = radiomics_features[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size].to(device)

                # Perform MCCS sampling
                x_hat_0, variance = model.mccs_sampling(x_0, j=args.j, I=args.i, C=args.C,
                                                        X_C=X_C, R=R,
                                                        mamba_blocks=mamba_blocks, 
                                                        transformer=transformer,  
                                                        upsample=upsample, 
                                                        downsample=downsample)
                
                # Calculate PSNR and SSIM
                for idx in range(x_hat_0.shape[0]):
                    pred = torch.tensor(x_hat_0[idx])
                    target = x_0[idx]

                    psnr, ssim = calculate_metrics(pred, target)
                    total_psnr += psnr
                    total_ssim += ssim
                    num_samples += 1

        # Calculate average PSNR and SSIM for the current fold
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples

        fold_psnr_results.append(avg_psnr)
        fold_ssim_results.append(avg_ssim)

        print(f"Fold {fold + 1}: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}")

    # Calculate overall mean and standard deviation
    mean_psnr = np.mean(fold_psnr_results)
    std_psnr = np.std(fold_psnr_results)
    mean_ssim = np.mean(fold_ssim_results)
    std_ssim = np.std(fold_ssim_results)

    print(f"Overall PSNR: {mean_psnr:.4f} ± {std_psnr:.4f}")
    print(f"Overall SSIM: {mean_ssim:.4f} ± {std_ssim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Riven-MCDM Model using MCCS with Cross-Validation")

    # Data paths
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target images directory")
    parser.add_argument("--condition_dir", type=str, required=True, help="Path to the condition images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask images directory")

    # Model directory (where all fold models are stored)
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where fold models are stored")

    # Output path (optional, if you want to save the predictions)
    parser.add_argument("--output_dir", type=str, help="Directory to save the output results")

    # Model and testing parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--T", type=int, default=1000, help="Total number of diffusion steps")
    parser.add_argument("--j", type=int, default=30, help="Number of compressed sampling steps")
    parser.add_argument("--i", type=int, default=5, help="Number of Monte Carlo iterations")
    parser.add_argument("--C", type=int, default=5, help="Constant to control the sampling interval size")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for cross-validation")

    # Other parameters (e.g., input/hidden dimensions for the modules)
    parser.add_argument("--input_dim", type=int, default=128, help="Input dimension for MambaBlock")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for MambaBlock")
    parser.add_argument("--img_feature_dim", type=int, default=512, help="Image feature dimension for Transformer")
    parser.add_argument("--radiomics_feature_dim", type=int, default=128, help="Radiomics feature dimension for Transformer")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in Transformer")

    args = parser.parse_args()

    main(args)