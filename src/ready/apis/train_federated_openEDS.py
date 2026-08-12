

import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
#from torchvision import transform
from loguru import logger
from omegaconf import OmegaConf

from ready.models.unet import UNet
from ready.utils.datasets import EyeDataset
from ready.utils.utils import set_data_directory

#torch.set_num_threads(1)    #reduce number of processing threads to avoid deadlocks when using DataLoader with num_workers > 0
#torch.set_num_interop_threads(1)
#torch.backends.mkldnn.enabled = False    

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
        )  # torch.Size([3, 400, 640])
        print(
            f"   SANITY_CHECK labels[0].shape: {labels[0].shape}"
        )  # torch.Size([400, 640])
        print(
            f"   SANITY_CHECK outputs.shape: {outputs.shape}"
        )  # torch.Size([1, 4, 400, 640])
        print(
            f"   SANITY_CHECK outputs[0].shape: {outputs[0].shape}"
        )  # torch.Size([4, 400, 640])
        print(f"   SANITY_CHECK no.shape: {no.shape}")  # torch.Size([400, 640])

        print(
            f"   SANITY_CHECK no[no == 0].size(): {no[no == 0].size()}, \
                                no[no == 1].size(): {no[no == 1].size()}, \
                                no[no == 2].size(): {no[no == 2].size()}, \
                                no[no == 3].size(): {no[no == 3].size()}"
        )

        # TOSAVE_PLOTS_TEMPORALY?
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
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    starttime = time.time()  # print(f'Starting training loop at {startt}')

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path

    PRETRAINED_MODEL = config.pretrained_model.models_folder_path

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUG_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    FULL_PRETRAINED_MODEL_PATH = os.path.join(Path.home(), PRETRAINED_MODEL)
    if not os.path.exists(FULL_MODEL_PATH):
        os.mkdir(FULL_MODEL_PATH)

    cuda_available = torch.cuda.is_available()
    logger.info(f"cuda_available: {cuda_available}")

    trainset = EyeDataset(
        #Change back to original path when done testing with sample data from repo
        #FULL_GITHUG_DATA_PATH+"/sample-frames/val3frames"
        #FULL_DATA_PATH+"/openEDS/openEDS"
        FULL_DATA_PATH
    )
    logger.info(f"Length of trainset: {len(trainset)}")

    batch_size = config.model_hyperparameters.batch_size
    num_workers = config.model_hyperparameters.num_workers
    learning_rate = config.model_hyperparameters.learning_rate
    run_epoch = config.model_hyperparameters.epochs


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    logger.info(f"trainloader.batch_size: {trainloader.batch_size}")


    #load most recent model global path
    model = UNet(nch_in=3, nch_out=4, nch_ker=16)
    weighted_files = list(Path(FULL_PRETRAINED_MODEL_PATH).rglob("*.pth"))
    if weighted_files:
        latest_modification = max(weighted_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"loading {latest_modification}")
        print(f"Loading: {latest_modification}")
        model.load_state_dict(torch.load(latest_modification))
    else:
        logger.info("No .pth files found")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 1, 0.8, 10]).float())

    if cuda_available:
        model.cuda()
        loss_fn.cuda()

    epoch = None

    for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
        print(f"Epoch {i}:")
        sum_loss = 0.0

        for j, data in enumerate(trainloader, 1):
            images, labels = data
            if cuda_available:
                images = images.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            output = model(images)  # torch.Size([8, 4, 400, 640])
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if j % 100 == 0 or j == 1:  # if j % 2 == 0 or j == 1:
                print(f"Loss at {j} mini-batch {loss.item()/trainloader.batch_size}")

        print(f"Average loss @ epoch: {sum_loss / (j*trainloader.batch_size)}")

    print("Training complete. Saving checkpoint...")
    current_time_stamp= datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    PATH = FULL_MODEL_PATH+"/"+datetime.now().strftime("%d-%b-%Y")
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    openEDS_weights = os.path.join(Path.home(), "Scratch/scratch/ccaekqu/datasets/ready/ready/federated/openEDS_weights.pth")
    os.makedirs(os.path.dirname(openEDS_weights), exist_ok=True)
    torch.save(model.state_dict(), openEDS_weights)
    logger.info(f"Saved openEDS weight to {openEDS_weights}")

    endtime = time.time()
    elapsedtime = endtime - starttime
    print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")
