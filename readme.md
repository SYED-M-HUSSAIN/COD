# Unified Camouflaged Object Detection Dataset

Welcome to the Unified Camouflaged Object Detection Dataset!

## Introduction

This open-source dataset is designed to advance the field of camouflaged object detection by consolidating and standardizing various benchmark datasets into a single, accessible resource. The goal of this project is to provide a comprehensive collection of camouflaged object images to foster collaboration and accelerate research in the domain of object detection and segmentation.

## Dataset Overview

- **Unified Benchmark Dataset Compilation:** We have collected and organized five benchmark camouflaged datasets, including Camo, NC4K, MoCa, COD10k, and CAD. The primary contributions of this dataset are:
  - A comprehensive and extensive collection of camouflaged object instances.
  - Removal of fragmentation and standardization of datasets.
  - Open-source accessibility to facilitate research and development in the field.

- **Transition from Binary Masks to YOLO Format Labels:** We have transitioned from binary masks to YOLO (You Only Look Once) format labels. This transition enables the training of single-stage object detection models and expands the utility of the dataset beyond traditional segmentation tasks.

- **Data Split into Standard Training, Validation, and Test Sets:** The dataset has been meticulously organized into standardized partitions for model development and evaluation:
  - Training (75%): Intended for model training and development.
  - Validation (15%): Used for hyperparameter tuning and model performance evaluation.
  - Test (15%): Reserved for the final evaluation of model performance and
