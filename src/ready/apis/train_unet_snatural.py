"""
Train pipeline for UNET
"""

import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
from loguru import logger
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms

from ready.models.unet import UNet
from ready.utils.datasets import Rti_Eyes_Dataset
from ready.utils.metrics import evaluate
from ready.utils.utils import create_data_loaders, set_data_directory, evaluate_model

torch.cuda.empty_cache()


def save_checkpoint(state, path):
    """
    Save checkpoint method
    """
    torch.save(state, path)
    print(f"Checkpoint saved at {path}")


def norm_image(hot_img):
    """
    Normalise image
    """
    return torch.argmax(hot_img, dim=0)


def sanity_check(trainloader, neural_network, cuda_available):
    """
    Sanity check for trainloader and model
    """
    f, ax = plt.subplots(5, 3)

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        outputs = neural_network(images[0].unsqueeze(0))
        no = norm_image(outputs[0]).squeeze(0)
        print(
            f"   SANITY_CHECK images[0].shape: {images[0].shape}"
        )
        print(
            f"   SANITY_CHECK labels[0].shape: {labels[0].shape}"
        )
        print(
            f"   SANITY_CHECK outputs.shape: {outputs.shape}"
        )
        print(
            f"   SANITY_CHECK outputs[0].shape: {outputs[0].shape}"
        )
        print(f"   SANITY_CHECK no.shape: {no.shape}")

        print(
            f"   SANITY_CHECK no[no == 0].size(): {no[no == 0].size()}, \
                                no[no == 1].size(): {no[no == 1].size()}, \
                                no[no == 2].size(): {no[no == 2].size()}, \
                                no[no == 3].size(): {no[no == 3].size()}"
        )

        ax[0, 0].imshow(
            (images[0].permute(1, 2, 0) * 255).to(torch.long).squeeze(0).cpu()
        )
        ax[0, 1].imshow(labels[0].cpu())
        ax[1, 1].imshow(labels[0].cpu() > 0)
        ax[2, 1].imshow(labels[0].cpu() > 1)
        ax[3, 1].imshow(labels[0].cpu() > 2)
        ax[4, 1].imshow(labels[0].cpu() > 3)
        ax[0, 2].imshow(no.cpu())
        ax[1, 2].imshow(no.cpu() > 0)
        ax[2, 2].imshow(no.cpu() > 1)
        ax[3, 2].imshow(no.cpu() > 2)
        ax[4, 2].imshow(no.cpu() > 3)
        plt.show()

        break

if __name__ == "__main__":
    """
    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    #CHECK save loss
    """
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    starttime = time.time()

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path

    TRAIN_SET_RATIO = config.datasets_splitting_ratios.train_set
    VALIDATION_SET_RATIO = config.datasets_splitting_ratios.validation_set
    TEST_SET_RATIO = config.datasets_splitting_ratios.test_set

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUG_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    if not os.path.exists(FULL_MODEL_PATH):
        os.mkdir(FULL_MODEL_PATH)

    cuda_available = torch.cuda.is_available()
    logger.info(f"cuda_available: {cuda_available}")

    full_dataset = Rti_Eyes_Dataset(
        FULL_DATA_PATH+"/",
        transform=transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ]),
        target_transform=transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST, antialias=True),
        ]),
    )

    logger.info(f"Length of full_dataset: {len(full_dataset)}")

    batch_size = config.model_hyperparameters.batch_size
    num_workers = config.model_hyperparameters.num_workers
    learning_rate = config.model_hyperparameters.learning_rate
    run_epoch = config.model_hyperparameters.epochs

    SEED = 42
    data_splitting_ratios = [TRAIN_SET_RATIO, VALIDATION_SET_RATIO, TEST_SET_RATIO]

    trainloader, validationloader, testloader = create_data_loaders(
        full_dataset=full_dataset,
        data_splitting_ratios=data_splitting_ratios,
        seed=SEED,
        batch_size=batch_size,
        num_workers=num_workers)

    logger.info(f"trainloader.batch_size: {trainloader.batch_size}")

    model = UNet(nch_in=3, nch_out=4)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    epoch = None

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print(f"Epoch {i}:")
        sum_loss = 0.0
        epoch_metrics = []

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if j % 10 == 0 or j == 1:
                batch_metrics = evaluate(output, labels, n_classes=4, average="weighted")
                epoch_metrics.append(batch_metrics)
            if j == 200:
                break

            if j % 10 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size}")

        print(f"Average loss @ epoch: {sum_loss / (j*trainloader.batch_size)}")

        avg_metrics = {
            key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            for key in epoch_metrics[0]
        }
        print("Training Metrics:")
        for key, value in avg_metrics.items():
            logger.info(f"Average training {key} @ epoch: {value:.4f}")

        # Validation Section
        model.eval()
        val_sum_loss = 0.0
        val_epoch_metrics = []
        with torch.no_grad():
            for vj, vdata in enumerate(validationloader, 1):
                vimages, vlabels = vdata
                if cuda_available:
                    vimages = vimages.cuda()
                    vlabels = vlabels.cuda()

                voutput = model(vimages)
                vloss = loss_fn(voutput, vlabels)
                val_sum_loss += vloss.item()

                if vj % 10 == 0 or vj == 1:
                    vbatch_metrics = evaluate(voutput, vlabels, n_classes=4, average="weighted")
                    val_epoch_metrics.append(vbatch_metrics)
                    print(f"Validation Loss at {vj} mini-batch {vloss.item()/validationloader.batch_size}")

                if vj == 200:
                    break
        model.train()

        print(f"Average validation loss @ epoch: {val_sum_loss / (vj*validationloader.batch_size)}")

        val_avg_metrics = {
            key: sum(m[key] for m in val_epoch_metrics) / len(val_epoch_metrics)
            for key in val_epoch_metrics[0]
        }
        print("Validation Metrics:")
        for key, value in val_avg_metrics.items():
            logger.info(f"Average validation {key} @ epoch: {value:.4f}")

    print("Training complete. Saving checkpoint...")
    current_time_stamp= datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    PATH = FULL_MODEL_PATH+"/"+datetime.now().strftime("%d-%b-%Y")
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    model_name = PATH+"/_weights_" + current_time_stamp + ".pth"
    torch.save(model.state_dict(), model_name)
    logger.info(f"Saved PyTorch Model State to {model_name}")

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")

    device = torch.device("cuda" if cuda_available else "cpu")
    logger.info("Commencing Evaluation")
    test_accuracy = evaluate_model(model=model, test_loader=testloader, device=device)
    logger.info(f"Model Accuracy: {test_accuracy: .4f}%")
    logger.info("Completed!")
