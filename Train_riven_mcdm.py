import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from Dataload import MedicalImageDataset
from utils import MambaBlock, TransformerModule, UpSampleFunction, DownSampleFunction
from model import DiffusionModel
from RadiomicsExtractor import load_radiomics_features

from sklearn.model_selection import KFold

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    batch_size_per_gpu = args.batch_size // num_gpus

    # Load the dataset
    dataset = MedicalImageDataset(args.target_dir, args.condition_dir, args.mask_dir)
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    # Load radiomics features
    radiomics_features = load_radiomics_features(args.condition_dir, args.mask_dir)
    radiomics_features = torch.tensor(radiomics_features, dtype=torch.float32).to(device)

    # Training and Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{args.num_folds}")

        # Create training and validation sets using SubsetRandomSampler
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=val_sampler, num_workers=4)

        # Initialize the model and optimizer
        model_structure = nn.Identity()  # Replace with your actual model structure if needed
        model = DiffusionModel(num_timesteps=args.T, model=model_structure).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Initialize required modules
        mamba_blocks = [MambaBlock(args.input_dim, args.hidden_dim) for _ in range(8)]
        transformer = TransformerModule(args.img_feature_dim, args.radiomics_feature_dim, args.num_heads, args.num_layers)
        upsample = UpSampleFunction(scale_factor=2)
        downsample = DownSampleFunction(scale_factor=0.5)

        for epoch in range(args.num_epochs):
            model.train()
            total_loss = 0.0

            # Training phase
            for batch_idx, batch in enumerate(train_loader):
                x_0 = batch['target'].to(device)
                X_C = batch['condition'].to(device)

                # Extract the corresponding radiomics features (R) for the current batch
                R = radiomics_features[train_idx[batch_idx*batch_size_per_gpu:(batch_idx+1)*batch_size_per_gpu]].to(device)

                optimizer.zero_grad()

                # Forward diffusion process
                x_t, noise, t = model.forward(x_0)

                # Calculate Riven-MCDM loss
                epsilon_pred = model.model(x_t, t, X_C, R)
                loss = nn.functional.mse_loss(epsilon_pred, noise)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {avg_loss}")

            # Validation phase
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_idx, batch in enumerate(val_loader):
                    x_0 = batch['target'].to(device)
                    X_C = batch['condition'].to(device)

                    # Extract the corresponding radiomics features (R) for the current batch
                    R = radiomics_features[val_idx[batch_idx*batch_size_per_gpu:(batch_idx+1)*batch_size_per_gpu]].to(device)

                    # Perform MCCS sampling and evaluate the model
                    x_hat_0, variance = model.mccs_sampling(x_0, j=args.j, I=args.i, C=args.C,
                                                            X_C=X_C, R=R,
                                                            mamba_blocks=mamba_blocks, 
                                                            transformer=transformer,  
                                                            upsample=upsample, 
                                                            downsample=downsample)

                    # Optionally compute validation loss or other metrics based on x_hat_0
                    # val_loss += some_metric_function(x_hat_0, expected_output)

            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            print(f"Validation Loss: {avg_val_loss}")

        # Save the model for the current fold
        torch.save(model.state_dict(), f"riven_mcdm_model_fold_{fold + 1}.pth")
        print(f"Model for fold {fold + 1} saved.")

    print("Training completed for all folds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Riven-MCDM Model")

    # Data paths
    parser.add_argument("--target_dir", type=str, required=True, help="Path to the target images directory")
    parser.add_argument("--condition_dir", type=str, required=True, help="Path to the condition images directory")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to the mask images directory")

    # Model and training parameters
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
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