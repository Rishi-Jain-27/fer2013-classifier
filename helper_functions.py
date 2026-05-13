
# Import required libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
from typing import Dict, List, Optional
import subprocess
import threading
import time

###-----Image Plotting-----###
def plot_image(
        img: torch.Tensor,
        class_names: list[str],
        label: int,
        cmap: str = "gray",
        axis_state: bool = False
        ) -> None:
    """
    Displays a PyTorch tensor as an image using Matplotlib.

    This function handles grayscale (1-channel), RGB (3-channel), and 2D tensors.
    It automatically moves the tensor to the CPU and adjusts dimensions 
    from PyTorch format [C, H, W] to Matplotlib format [H, W, C] if necessary.

    Args:
        img (torch.Tensor): The image tensor to plot. Can be shape (H, W), 
            (1, H, W), or (3, H, W).
        class_names (list[str]): A list of strings containing the human-readable 
            labels for the dataset.
        label (int): The integer index of the label for this specific image.
        cmap (str, optional): The Matplotlib colormap to use for grayscale 
            images. Defaults to "gray".
        axis_state (bool, optional): Whether to display the x and y axes. 
            Defaults to False (hidden).

    Returns:
        None
    """
    # Ensure the tensor is on CPU
    img_plot = img.cpu()

    # Logic to handle different tensor shapes
    if len(img_plot.shape) == 3 and img_plot.shape[0] == 1:
        plt.imshow(img_plot.squeeze(), cmap=cmap)
    elif len(img_plot.shape) == 3 and img_plot.shape[0] == 3:
        # Convert [C, H, W] -> [H, W, C]
        plt.imshow(img_plot.permute(1, 2, 0), cmap=cmap)
    else:
        plt.imshow(img_plot, cmap=cmap)
    
    plt.title(class_names[label])

    if not axis_state:
        plt.axis("off")
    else:
        plt.axis("on")

###-----Reproducibility-----###
def set_seeds(
        seed: int = 42
        ) -> None:
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.

    This function initializes seeds for the basic Python `random` module, 
    NumPy, and PyTorch (both CPU and CUDA). It also configures CuDNN to 
    ensure deterministic behavior, which is critical for reproducible 
    deep learning experiments.

    Args:
        seed: The integer value to use as the seed. Defaults to 42.

    Returns:
        None
    """
    # Standard Python and NumPy seeds
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] All seeds set to: {seed}")

###-----CNN Model Training-----###
def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """
    Performs a single training step (one epoch) over a given DataLoader.

    This function puts the model into training mode, iterates through the 
    provided dataset in batches, calculates the loss and accuracy, 
    performs backpropagation, and updates the model parameters.

    Args:
        model: The PyTorch neural network to be trained.
        dataloader: A DataLoader instance providing the training data batches.
        loss_fn: The PyTorch loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The PyTorch optimizer (e.g., torch.optim.Adam).
        device: The target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        A tuple containing (train_loss, train_accuracy), where both values 
        are averages calculated across all batches in the dataloader.
        Example: (0.4521, 0.8902)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0.0, 0.0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Performs a single evaluation step (one epoch) over a given DataLoader.

    This function sets the model to evaluation mode and uses inference mode 
    to disable gradient computation, reducing memory consumption and 
    speeding up forward passes. It calculates the average loss and 
    accuracy over the entire test/validation set.

    Args:
        model: The PyTorch neural network to be evaluated.
        dataloader: A DataLoader instance providing the test/validation data.
        loss_fn: The PyTorch loss function used to calculate test loss.
        device: The target device to compute on (e.g., "cuda" or "cpu").

    Returns:
        A tuple containing (test_loss, test_accuracy), where both values 
        are averages calculated across all batches in the dataloader.
        Example: (0.3210, 0.9150)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0.0, 0.0

    # Turn on inference mode context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1) # argmax has same result as softmax in this case
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    writer: SummaryWriter,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
    checkpoint_metric: str = "accuracy"
) -> Dict[str, List[float]]:
    """
    Trains and evaluates a PyTorch model while logging metrics to TensorBoard.

    This function runs a full supervised training loop for a classification
    model. For each epoch, it performs one training step and one evaluation
    step, logs the resulting loss and accuracy values to TensorBoard, stores
    the results in a dictionary, and optionally saves the best model checkpoint.

    The function also automatically starts a TensorBoard server and exposes it
    through a Cloudflare tunnel by calling:

        start_tensorboard_tunnel(log_dir="runs", port=6008)

    before training begins.

    Checkpointing behavior:
        - If checkpoint_metric="accuracy", the model is saved whenever test
          accuracy improves.
        - If checkpoint_metric="loss", the model is saved whenever test loss
          decreases.

    Args:
        model:
            The PyTorch model to train.

        train_dataloader:
            DataLoader containing the training dataset.

        test_dataloader:
            DataLoader containing the validation or test dataset.

        optimizer:
            PyTorch optimizer used to update model parameters.

        loss_fn:
            Loss function used to calculate prediction error.

        epochs:
            Number of complete passes through the training dataset.

        writer:
            TensorBoard SummaryWriter used to log loss, accuracy, and the
            model graph.

        device:
            Target device where the model and tensors should be placed.

            Example:
                torch.device("cuda") or torch.device("cpu")

        checkpoint_dir:
            Directory where best model checkpoints should be saved.

            Defaults to:
                "checkpoints"

        checkpoint_metric:
            Metric used to decide when to save the best model.

            Use:
                "accuracy" to save the model with the highest test accuracy.
                "loss" to save the model with the lowest test loss.

            Defaults to:
                "accuracy"

    Returns:
        A dictionary containing metric history across all epochs.

        Format:
            {
                "train_loss": [...],
                "train_acc": [...],
                "test_loss": [...],
                "test_acc": [...]
            }

    Raises:
        ValueError:
            If checkpoint_metric is not "accuracy" or "loss".

    Notes:
        The model graph is added to TensorBoard using one sample batch from the
        training dataloader. If graph tracing fails, training continues and a
        warning is printed.

        The TensorBoard writer is flushed after every epoch so that metrics
        appear while training is still running.
    """

    if checkpoint_metric not in ["accuracy", "loss"]:
        raise ValueError("checkpoint_metric must be either 'accuracy' or 'loss'.")

    start_tensorboard_tunnel(log_dir="runs", port=6008)

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    model.to(device)

    if checkpoint_metric == "accuracy":
        best_metric = 0.0
        checkpoint_mode = "max"
        checkpoint_path = os.path.join(checkpoint_dir, "best_accuracy_model.pth")
    else:
        best_metric = float("inf")
        checkpoint_mode = "min"
        checkpoint_path = os.path.join(checkpoint_dir, "best_loss_model.pth")

    example_batch_X, _ = next(iter(train_dataloader))
    example_batch_X = example_batch_X.to(device)

    try:
        writer.add_graph(model, input_to_model=example_batch_X)
        print(f"[INFO] Model graph traced with input shape: {example_batch_X.shape}")
    except Exception as e:
        print(f"[WARNING] Could not add model graph to TensorBoard: {e}")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        writer.add_scalars(
            main_tag="Loss",
            tag_scalar_dict={"train": train_loss, "test": test_loss},
            global_step=epoch
        )

        writer.add_scalars(
            main_tag="Accuracy",
            tag_scalar_dict={"train": train_acc, "test": test_acc},
            global_step=epoch
        )

        writer.flush()

        if checkpoint_metric == "accuracy":
            current_metric = test_acc
        else:
            current_metric = test_loss

        best_metric = save_best_model(
            model=model,
            target_metric=current_metric,
            best_metric=best_metric,
            checkpoint_path=checkpoint_path,
            mode=checkpoint_mode,
            extra_info={
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "optimizer_state_dict": optimizer.state_dict()
            }
        )

    writer.close()

    return results

def save_best_model(
    model: torch.nn.Module,
    target_metric: float,
    best_metric: float,
    checkpoint_path: str,
    mode: str = "max",
    extra_info: Optional[dict] = None
) -> float:
    """
    Saves a PyTorch model checkpoint when the current metric improves.

    This function is designed for experiment checkpointing during training.
    It compares the current metric against the best metric seen so far and
    saves the model only if the current metric is better.

    Common use cases:
        - Save when validation/test accuracy is highest.
        - Save when validation/test loss is lowest.

    Args:
        model:
            The PyTorch model being trained.

        target_metric:
            The current value of the metric being monitored.
            Example: current test accuracy or current test loss.

        best_metric:
            The best value of the monitored metric seen so far.

        checkpoint_path:
            File path where the model checkpoint should be saved.
            Example: "checkpoints/best_model.pth"

        mode:
            Determines whether improvement means a larger or smaller value.

            Use:
                "max" for metrics where higher is better, such as accuracy.
                "min" for metrics where lower is better, such as loss.

            Defaults to "max".

        extra_info:
            Optional dictionary containing additional information to save
            inside the checkpoint, such as epoch, optimizer state, loss,
            accuracy, or experiment name.

            Example:
                {
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "optimizer_state_dict": optimizer.state_dict()
                }

    Returns:
        The updated best metric.

        If the current metric improved, this returns target_metric.
        Otherwise, it returns the unchanged best_metric.

    Raises:
        ValueError:
            If mode is not "max" or "min".
    """

    if mode not in ["max", "min"]:
        raise ValueError("mode must be either 'max' or 'min'.")

    if mode == "max":
        improved = target_metric > best_metric
    else:
        improved = target_metric < best_metric

    if improved:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_metric": target_metric,
            "mode": mode
        }

        if extra_info is not None:
            checkpoint.update(extra_info)

        torch.save(checkpoint, checkpoint_path)

        print(
            f"[INFO] Saved new best model to {checkpoint_path} | "
            f"best_metric: {target_metric:.4f}"
        )

        return target_metric

    return best_metric

###-----Experiment Tracking-----###
def create_writer(
    experiment_name: str, 
    model_name: str, 
    extra: Optional[str] = None
) -> SummaryWriter:
    """
    Creates a torch.utils.tensorboard.SummaryWriter instance with a structured log directory.

    The log directory follows the hierarchy: 
    runs/YYYY-MM-DD/experiment_name/model_name/extra

    Args:
        experiment_name: Name of the overall experiment (e.g., 'data_augmentation_test').
        model_name: Name of the specific model architecture (e.g., 'resnet50').
        extra: Optional string for additional sub-directory info (e.g., 'lr_0.001_epochs_50').

    Returns:
        A SummaryWriter object pointing to the newly created log directory.
    """
    # Get timestamp of current date
    # Returns current date in YYYY-MM-DD format
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Construct the log directory path
    if extra:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    
    return SummaryWriter(log_dir=log_dir)

def start_tensorboard_tunnel(
        log_dir: str = "runs",
        port: int = 6008):
    
    """
    Starts TensorBoard locally and exposes it through a public Cloudflare tunnel.

    This function is useful in notebook or remote environments where localhost
    is not directly accessible. It starts a TensorBoard server on the specified
    port, then launches a Cloudflare tunnel that provides a public URL.

    The function performs the following steps:
        1. Downloads the cloudflared binary if it does not already exist.
        2. Kills any existing TensorBoard or cloudflared processes.
        3. Starts TensorBoard using the provided log directory and port.
        4. Starts a background Cloudflare tunnel.
        5. Prints the public TensorBoard URL when it becomes available.

    Args:
        log_dir:
            Path to the TensorBoard log directory.

            Example:
                "runs"

        port:
            Local port where TensorBoard should run.

            Example:
                6008

    Returns:
        None.

    Notes:
        This function assumes a Linux-like environment because it uses wget,
        chmod, pkill, and the Linux AMD64 cloudflared binary.

        In Google Colab or similar hosted notebook environments, this avoids
        needing ngrok authentication.
    """

    print("[INFO] Setting up Cloudflare tunnel for TensorBoard...")

    # 1. Download cloudflared if needed
    if not os.path.exists("cloudflared"):
        print("[INFO] Downloading cloudflared binary...")
        subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared",
            shell=True,
            check=False
        )
        subprocess.run("chmod +x cloudflared", shell=True, check=False)

    # 2. Kill old TensorBoard / cloudflared processes
    subprocess.run("pkill -f tensorboard || true", shell=True, check=False)
    subprocess.run("pkill -f cloudflared || true", shell=True, check=False)

    # 3. Start TensorBoard
    tb_cmd = [
        "tensorboard",
        "--logdir", log_dir,
        "--port", str(port),
        "--host", "0.0.0.0"
    ]

    tb_log = open("/tmp/tb.log", "w")

    subprocess.Popen(
        tb_cmd,
        stdout=tb_log,
        stderr=tb_log
    )

    print(f"[INFO] TensorBoard started at http://localhost:{port}")

    # 4. Start Cloudflare tunnel and print public URL
    def run_tunnel():

        """
        Starts the Cloudflare tunnel process and prints the public TensorBoard URL.

        This inner function is run in a background daemon thread so that the main
        training process can continue without blocking.

        It launches cloudflared with the local TensorBoard URL and continuously
        reads the process output until it finds a public trycloudflare.com URL.
        Once found, it prints the URL so the user can open TensorBoard in a
        browser.

        Args:
            None.

        Returns:
            None.

        Notes:
            Cloudflared often writes status messages to stdout or stderr.
            This function redirects stderr into stdout so both streams can be
            searched together.

            The function stops reading once the first public tunnel URL is found.
        """

        tunnel_cmd = [
            "./cloudflared",
            "tunnel",
            "--url",
            f"http://localhost:{port}"
        ]

        proc = subprocess.Popen(
            tunnel_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        if proc.stdout:
            for line in proc.stdout:
                if "trycloudflare.com" in line:
                    parts = line.strip().split()
                    url = next((p for p in parts if "trycloudflare.com" in p), None)

                    if url:
                        print("\n" + "=" * 60)
                        print(f"TENSORBOARD PUBLIC URL: {url}")
                        print("=" * 60 + "\n")
                    break

    threading.Thread(target=run_tunnel, daemon=True).start()

    # Give TensorBoard and the tunnel a moment to initialize
    time.sleep(2)

###-----Plot Loss Curves###
def plot_loss_curves(
        results: Dict[str, List[float]]
        ):
    """
    Plots training and test loss and accuracy curves from a results dictionary.

    Args:
        results: A dictionary containing lists of metrics.
            Expects: {"train_loss": [...], "test_loss": [...], 
                      "train_acc": [...], "test_acc": [...]}
    """
    # Extract data from results dictionary
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Calculate number of epochs
    epochs = range(len(results["train_loss"]))

    # Setup the figure with two subplots
    plt.figure(figsize=(15, 7))

    # --- Plot 1: Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # --- Plot 2: Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.show()
