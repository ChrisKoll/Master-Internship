"""
Collection of functions used in the model training loop.
"""

# === Libraries ===
import torch
from tqdm import tqdm

# === Functions ===


def train(
    model,
    dataloader,
    optimizer,
    prev_updates,
    device,
    model_type: str = None,
    writer=None,
):
    """
    Trains the model on the given data.

    Args:
        model (nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The data loader.
        optimizer (torch.optim.Optim): The optimizer.
        prev_updates (int): Number of previous updates.
        device (str): Device.
        writer: The TensorBoard writer.
    """
    model.train()  # Set the model to training mode

    for batch_idx, data in enumerate(tqdm(dataloader)):
        n_upd = prev_updates + batch_idx

        data = data.to(device)

        optimizer.zero_grad()  # Zero the gradients

        output = model(data)  # Forward pass
        loss = output.loss

        loss.backward()

        if n_upd % 100 == 0:
            # Calculate and log gradient norms
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)

            print(
                f"Step {n_upd:,} (N samples: {n_upd * dataloader.batch_size:,}), Loss: {loss.item():.4f} Grad: {total_norm:.4f}"
            )

            if writer is not None:
                global_step = n_upd
                writer.add_scalar("Loss/Train", loss.item(), global_step)
                writer.add_scalar("GradNorm/Train", total_norm, global_step)
                if model_type == "vae":
                    writer.add_scalar(
                        "Loss/Train/BCE", output.loss_recon.item(), global_step
                    )
                    writer.add_scalar(
                        "Loss/Train/KLD", output.loss_kl.item(), global_step
                    )

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # Update the model parameters

    return prev_updates + len(dataloader)


def test(model, dataloader, cur_step, device, model_type: str = None, writer=None):
    """
    Tests the model on the given data.

    Args:
        model (nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The data loader.
        cur_step (int): The current step.
        writer: The TensorBoard writer.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Testing"):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=True)  # Forward pass

            test_loss += output.loss.item()

    test_loss /= len(dataloader)
    print(f"====> Test set loss: {test_loss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/Test", test_loss, global_step=cur_step)
        if model_type == "vae":
            writer.add_scalar(
                "Loss/Test/BCE", output.loss_recon.item(), global_step=cur_step
            )
            writer.add_scalar(
                "Loss/Test/KLD", output.loss_kl.item(), global_step=cur_step
            )

        # Log random samples from the latent space
        # z = torch.randn(16, latent_dim).to(device)
        # samples = model.decode(z)
