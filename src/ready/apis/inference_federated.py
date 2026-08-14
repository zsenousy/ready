import os
from argparse import ArgumentParser
from pathlib import Path
from xml.parsers.expat import model

import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import OmegaConf
import subprocess
import torchvision.transforms.v2 as transforms

from ready.models.unet import UNet
from ready.utils.datasets import EyeDataset, MobiousDataset, Rti_Eyes_Dataset
from argparse import Namespace
from ready.utils.metrics import evaluate
from ready.utils.utils import (HOME_PATH, create_data_loaders, evaluate_model,
                               loss_values_file_writer,
                               performance_file_writer,
                               sanity_check_trainloader,
                               test_accuracy_file_writer, training_loop,
                               validation_loop)

# TODO
# Make sure we have a common path for models to avoid looking where the model path is!


def main(args):
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    #DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    #GITHUB_DATA_PATH = config.dataset.github_data_path
    INFERENCE_RESULTS = config.dataset.inference_results
    MOBIOUS_DATA_PATH = config.dataset.mobious_data_path
    OPENEDS_DATA_PATH = config.dataset.openEDS_data_path
    RTI_EYES_DATA_PATH = config.dataset.rti_eyes_data_path

    #FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    #FULL_GITHUB_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    FULL_MOBIOUS_DATA_PATH = os.path.join(Path.home(), MOBIOUS_DATA_PATH)
    FULL_OPENEDS_DATA_PATH = os.path.join(Path.home(), OPENEDS_DATA_PATH)
    FULL_RTI_DATA_PATH = os.path.join(Path.home(), RTI_EYES_DATA_PATH)
    FULL_INFERENCE_RESULTS = os.path.join(Path.home(), INFERENCE_RESULTS)



    #input_model_name=config.model.input_model_name
    #model_name = input_model_name[:-4]
    #logger.info(f"model_name {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logger.info(f"CUDA is available")
        #import onnxruntime


    weighted_files = list(Path(FULL_MODEL_PATH).rglob("*.pth"))
    model = UNet(nch_in=3, nch_out=4, nch_ker=16)
    model = model.to(device)
    if weighted_files:
        latest_modification = max(weighted_files, key = lambda f: f.stat().st_mtime)
        print(f"Loading: {latest_modification}")
        model.load_state_dict(torch.load(latest_modification, map_location=device))
        model.eval()
        # Normalise weights to reasonable range
        with torch.no_grad():
            for name, param in model.named_parameters():
                max_val = param.data.abs().max()
                if max_val > 100:  # only normalise if weights are too large
                    param.data = param.data / max_val * 1
    else:
        logger.info("No weights found") #debugging

    #if cuda_available:
        #### ONNX model
      #  onnx_checkpoint_path = str(FULL_MODEL_PATH) + '/' + str(model_name) + "-sim.onnx"
      #  ort_session = onnxruntime.InferenceSession(
       #     onnx_checkpoint_path, providers=["CPUExecutionProvider"]
      #  )

        # UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
        #def to_numpy(tensor):
            #return (
               # tensor.detach().cpu().numpy()
               # if tensor.requires_grad
              #  else tensor.cpu().numpy() )

    transform = transforms.Compose([
        transforms.Resize((128, 128), antialias=True)
    ])

    target_transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True)
    ])


    openEDS_dataset = EyeDataset(FULL_OPENEDS_DATA_PATH, transform=transform, target_transform=target_transform)
    openEDS_loader = torch.utils.data.DataLoader(openEDS_dataset, batch_size=1, shuffle=True, num_workers=0)
        
    openEDS_miou, openEDS_dice, openEDS_hausdorff, openEDS_accuracy = [], [], [], []
        
    with torch.no_grad():
        for j, data in enumerate(openEDS_loader, 1):
            print(f"openEDS batch {j}")
            logger.info(f"openEDS batch {j}")
            images, labels = data
            if cuda_available:
                images, labels = images.cuda(), labels.cuda()
            image = images[0].unsqueeze(0)
            label = labels[0].unsqueeze(0)
        
            output = model(image)
            temperature = 1.0
            pred = torch.argmax(F.softmax(output / temperature, dim=1), dim=1)
        
            if j <=  25:
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(image.squeeze(0).permute(1, 2, 0).cpu())
                ax[0].set_title("openEDS - Original Image")
                ax[1].imshow(label.squeeze(0).cpu())
                ax[1].set_title("OpenEDS - Mask")
                ax[2].imshow(pred.squeeze(0).cpu())
                ax[2].set_title("Model Prediction")
                plt.savefig(f"{FULL_INFERENCE_RESULTS}/inference_results/openEDS/openEDS_result{j}.png")
                plt.close()
            logger.info(f" openeds done")
            print(f"openeds done")
        
            metrics = evaluate(output, labels, n_classes=4)
            openEDS_miou.append(metrics['miou'])
            openEDS_dice.append(metrics['dice'])
            openEDS_hausdorff.append(metrics['hausdorff_distance'])
            openEDS_accuracy.append(metrics['accuracy'])

    mobious_dataset = MobiousDataset(FULL_MOBIOUS_DATA_PATH, transform=transform, target_transform=target_transform)
    mobious_loader = torch.utils.data.DataLoader(mobious_dataset, batch_size=1, shuffle=True, num_workers=0)
        
    mobious_miou, mobious_dice, mobious_hausdorff, mobious_accuracy = [], [], [], []
        
    with torch.no_grad():
        for j, data in enumerate(mobious_loader, 1):
            images, labels = data
            if cuda_available:
                images, labels = images.cuda(), labels.cuda()
            image = images[0].unsqueeze(0)
            label = labels[0].unsqueeze(0)
        
            output = model(image)
            temperature = 1.0
            pred = torch.argmax(F.softmax(output / temperature, dim=1), dim=1)
        
                #save at least first 5 images for each dataset
            if j <= 25:      
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(image.squeeze(0).permute(1, 2, 0).cpu() / 255)
                ax[0].set_title("Mobious - Original Image")
                ax[1].imshow(label.squeeze(0).cpu())
                ax[1].set_title("Mobious - Mask")
                ax[2].imshow(pred.squeeze(0).cpu())
                ax[2].set_title("Model Prediction")
                plt.savefig(f"{FULL_INFERENCE_RESULTS}/inference_results/mobious/mobious_result{j}.png")
                plt.close()
            logger.info(f" mobious done")
            print(f"mobious done")
        
            metrics = evaluate(output, labels, n_classes=4)
            mobious_miou.append(metrics['miou'])
            mobious_dice.append(metrics['dice'])
            mobious_hausdorff.append(metrics['hausdorff_distance'])
            mobious_accuracy.append(metrics['accuracy'])




    rti_dataset = Rti_Eyes_Dataset(FULL_RTI_DATA_PATH, transform=transform, target_transform=target_transform)
    rti_loader = torch.utils.data.DataLoader(rti_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    rti_miou, rti_dice, rti_hausdorff, rti_accuracy = [], [], [], []
    
    with torch.no_grad():
        for j, data in enumerate(rti_loader, 1):
            images, labels = data
            if cuda_available:
                images, labels = images.cuda(), labels.cuda()
            image = images[0].unsqueeze(0)
            label = labels[0].unsqueeze(0)
    
            output = model(image)

            temperature = 1.0
            pred = torch.argmax(F.softmax(output / temperature, dim=1), dim=1)
    
            if j <= 25:
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                img_display = image.squeeze(0).permute(1, 2, 0).cpu()
                img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
                ax[0].imshow(img_display)
                ax[0].set_title("RTI-Eyes - Original Image")
                ax[1].imshow(label.squeeze(0).cpu())
                ax[1].set_title("RTI-Eyes - Mask")
                ax[2].imshow(pred.squeeze(0).cpu())
                ax[2].set_title("Model Prediction")
                plt.savefig(f"{FULL_INFERENCE_RESULTS}/inference_results/rti_eyes/rti_result{j}.png")
                plt.close()
            logger.info(f" rit done")
            print(f"rit done")
    
            metrics = evaluate(output, labels, n_classes=4)
            rti_miou.append(metrics['miou'])
            rti_dice.append(metrics['dice'])
            rti_hausdorff.append(metrics['hausdorff_distance'])
            rti_accuracy.append(metrics['accuracy'])
    

    

    



    logger.info(f"\n########### FEDERATED INFERENCE RESULTS ##########")
    logger.info(f"{'Dataset':<12} {'mIoU':<10} {'Dice':<10} {'Hausdorff':<10} {'Accuracy':<10}")
    logger.info(f"{'Mobious':<12} {sum(mobious_miou)/len(mobious_miou):<10.4f} {sum(mobious_dice)/len(mobious_dice):<10.4f} {sum(mobious_hausdorff)/len(mobious_hausdorff):<10.4f} {sum(mobious_accuracy)/len(mobious_accuracy):<10.4f}")
    logger.info(f"{'openEDS':<12} {sum(openEDS_miou)/len(openEDS_miou):<10.4f} {sum(openEDS_dice)/len(openEDS_dice):<10.4f} {sum(openEDS_hausdorff)/len(openEDS_hausdorff):<10.4f} {sum(openEDS_accuracy)/len(openEDS_accuracy):<10.4f}")
    logger.info(f"{'RTI-Eyes':<12} {sum(rti_miou)/len(rti_miou):<10.4f} {sum(rti_dice)/len(rti_dice):<10.4f} {sum(rti_hausdorff)/len(rti_hausdorff):<10.4f} {sum(rti_accuracy)/len(rti_accuracy):<10.4f}")

if __name__ == "__main__":

    parser = ArgumentParser(description="Plot inference for models pth")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
    args = parser.parse_args()
    main(args)
