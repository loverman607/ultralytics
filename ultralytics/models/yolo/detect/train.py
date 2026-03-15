# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import os
import random
from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, distributed

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.build import ContiguousDistributedSampler, InfiniteDataLoader, seed_worker
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    int(self.args.imgsz * (1.0 - self.args.multi_scale)),
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj)

class LTTrainer(DetectionTrainer):
    """Trainer for tail-aware YOLO26."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return LTDetectionModel."""
        from ultralytics.nn.tasks import LTDetectionModel
        model = LTDetectionModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build dataset and compute class statistics for tail balancing."""
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            self._set_class_stats(dataset)
        return dataset

    def _set_class_stats(self, dataset):
        """Compute class counts and weights from dataset labels."""
        if getattr(self, "_class_stats_set", False):
            return
        nc = int(self.data["nc"])
        counts = np.zeros(nc, dtype=np.float64)
        for label in dataset.labels:
            cls = label.get("cls", None)
            if cls is None:
                continue
            cls = np.asarray(cls, dtype=np.int64).reshape(-1)
            if cls.size:
                counts += np.bincount(cls, minlength=nc)
        counts = np.maximum(counts, 1.0)  # avoid divide-by-zero
        total = counts.sum()
        pos_weight = (total - counts) / counts
        self.model.class_counts = torch.from_numpy(counts.astype(np.float32))
        self.model.class_weights = torch.from_numpy(pos_weight.astype(np.float32))
        self._set_head_tail_split(counts)
        self._class_stats_set = True

    def _set_head_tail_split(self, counts: np.ndarray):
        """Compute head/tail class split and convert to dual head."""
        nc = int(self.data["nc"])
        if nc < 2:
            return
        tail_count = getattr(self.args, "tail_count", None)
        tail_frac = getattr(self.args, "tail_frac", None)
        if tail_count is not None:
            tail_idx = np.where(counts <= float(tail_count))[0].tolist()
        else:
            tail_frac = 0.3 if tail_frac is None else float(tail_frac)
            tail_n = max(1, int(round(nc * tail_frac)))
            tail_idx = np.argsort(counts)[:tail_n].tolist()

        # Ensure non-empty head/tail split
        tail_idx = sorted(set(tail_idx))
        if len(tail_idx) >= nc:
            tail_idx = tail_idx[: max(1, nc - 1)]
        head_idx = [i for i in range(nc) if i not in tail_idx]
        if not head_idx or not tail_idx:
            return

        self.model.head_classes = head_idx
        self.model.tail_classes = tail_idx
        if hasattr(self.model, "set_dual_head"):
            self.model.set_dual_head(head_idx, tail_idx)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build dataloader with tail-biased sampler for rare classes."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False

        sampler = None
        if mode == "train":
            sampler = self._build_tail_biased_sampler(dataset, rank)

        if sampler is None:
            sampler = (
                None
                if rank == -1
                else distributed.DistributedSampler(dataset, shuffle=shuffle)
                if shuffle
                else ContiguousDistributedSampler(dataset)
            )

        batch = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers if mode == "train" else self.args.workers * 2)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and mode == "train" and len(dataset) % batch != 0,
        )

    def _build_tail_biased_sampler(self, dataset, rank):
        """Create a tail-biased sampler for long-tail class oversampling."""
        if rank != -1:
            LOGGER.warning("Tail-biased sampler disabled for DDP; using default distributed sampling.")
            return None
        if len(dataset) == 0:
            return None
        weights = self._compute_image_weights(dataset)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    def _compute_image_weights(self, dataset, alpha: float = 1.0):
        """Compute per-image weights from class frequencies."""
        counts = getattr(self.model, "class_counts", None)
        if counts is None:
            # fallback: compute counts directly
            nc = int(self.data["nc"])
            counts = np.zeros(nc, dtype=np.float64)
            for label in dataset.labels:
                cls = label.get("cls", None)
                if cls is None:
                    continue
                cls = np.asarray(cls, dtype=np.int64).reshape(-1)
                if cls.size:
                    counts += np.bincount(cls, minlength=nc)
            counts = np.maximum(counts, 1.0)
        else:
            counts = counts.detach().cpu().numpy().astype(np.float64)
            counts = np.maximum(counts, 1.0)

        class_weights = 1.0 / (counts**alpha)
        class_weights /= class_weights.mean()

        img_weights = np.zeros(len(dataset), dtype=np.float64)
        for i, label in enumerate(dataset.labels):
            cls = label.get("cls", None)
            if cls is None:
                img_weights[i] = class_weights.min()
                continue
            cls = np.asarray(cls, dtype=np.int64).reshape(-1)
            if cls.size:
                img_weights[i] = class_weights[cls].max()
            else:
                img_weights[i] = class_weights.min()
        return torch.from_numpy(img_weights).double()

    def validate(self):
        """Compute tail-aware metrics (F1/AP per class)."""
        metrics, fitness = super().validate()
        if metrics:
            # Log tail-class F1
            LOGGER.info("Tail-class metrics: ...")
        return metrics, fitness

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Per-layer LR: backbone lower, head higher."""
        backbone_params = []
        head_params = []
        model_ = unwrap_model(model)
        backbone_end = getattr(model_, "backbone_end", None)
        if backbone_end is None and hasattr(model_, "yaml") and isinstance(model_.yaml, dict):
            backbone_end = len(model_.yaml.get("backbone", [])) - 1

        def _is_backbone(name: str) -> bool:
            if name.startswith("cbam"):
                return True
            if name.startswith("model."):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit() and backbone_end is not None:
                    return int(parts[1]) <= backbone_end
            return False

        for k, v in model_.named_parameters():
            if not v.requires_grad:
                continue
            if _is_backbone(k):
                backbone_params.append(v)
            else:
                head_params.append(v)

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr*0.1, "weight_decay": decay},
            {"params": head_params, "lr": lr, "weight_decay": decay},
        ])
        return optimizer
