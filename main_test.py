import inspect
import logging
import os
import wandb

import hydra
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.linear import parse_cfg
from solo.data.test_dataloader import prepare_test_data
from solo.data.pretrain_dataloader import NCropAugmentation, build_transform_pipeline
from solo.methods.bagoffeatures import BagOfFeaturesModel
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True



#@hydra.main(version_base="1.2")
def main(config_path, rel_model_path=None):#='scripts/linear/eurosat_msi/frossl_hat.yaml'
    if rel_model_path:
        ckpt_path = "trained_models\linear\\" +rel_model_path
    else:
        with open("last_ckpt.txt", "r") as f:
            ckpt_path = f.read().strip()
        # delete last_ckpt file
        os.remove("last_ckpt.txt")
    print(ckpt_path)
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    # initialize backbone
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        # remove fc layer
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()
        elif cfg.data.dataset[-3:] == "msi": #TODO adapt kernel size, ..
            backbone.conv1 = nn.Conv2d(
                13, 64, kernel_size=7, stride=2, padding=3, bias=False
            )


    # state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    # for k in list(state.keys()):
    #     if "encoder" in k:
    #         state[k.replace("encoder", "backbone")] = state[k]
    #         logging.warn(
    #             "You are using an older checkpoint. Use a new one as some issues might arrise."
    #         )
    #     if "backbone" in k:
    #         state[k.replace("backbone.", "")] = state[k]
    #     del state[k]
    # backbone.load_state_dict(state, strict=False)
    # logging.info(f"Loaded {ckpt_path}")

    # check if mixup or cutmix is enabled
    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=cfg.data.num_classes,
        )
        # smoothing is handled with mixup label transform
        loss_func = SoftTargetCrossEntropy()
    elif cfg.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    if cfg.pretrain_method == "empssl":
        aug_cfg = cfg.augmentations[0]

        train_pipeline = NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), 20
                )
        val_pipeline =  NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), 20
                )
        modelClass = BagOfFeaturesModel
    else:
        modelClass = LinearModel
        train_pipeline = None
        val_pipeline = None

    model = modelClass(backbone, loss_func=loss_func, mixup_func=mixup_func, cfg=cfg)

    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=False)
    # make_contiguous(model)

    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    test_loader = prepare_test_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=0,#cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
        train_pipeline = train_pipeline,
        val_pipeline = val_pipeline,
        precompute_features=cfg.precompute,
        model=model,
    )

    # if cfg.data.format == "dali":
    #     assert (
    #         _dali_avaliable
    #     ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."

    #     assert not cfg.auto_augment, "Auto augmentation is not supported with Dali."

    #     dali_datamodule = ClassificationDALIDataModule(
    #         dataset=cfg.data.dataset,
    #         train_data_path=cfg.data.train_path,
    #         val_data_path=cfg.data.val_path,
    #         num_workers=cfg.data.num_workers,
    #         batch_size=cfg.optimizer.batch_size,
    #         data_fraction=cfg.data.fraction,
    #         dali_device=cfg.dali.device,
    #     )

    #     # use normal torchvision dataloader for validation to save memory
    #     dali_datamodule.val_dataloader = lambda: val_loader

    # # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    # ckpt_path, wandb_run_id = None, None
    # if cfg.auto_resume.enabled and cfg.resume_from_checkpoint is None:
    #     auto_resumer = AutoResumer(
    #         checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
    #         max_hours=cfg.auto_resume.max_hours,
    #     )
    #     resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
    #     if resume_from_checkpoint is not None:
    #         print(
    #             "Resuming from previous checkpoint that matches specifications:",
    #             f"'{resume_from_checkpoint}'",
    #         )
    #         ckpt_path = resume_from_checkpoint
    # elif cfg.resume_from_checkpoint is not None:
    #     ckpt_path = cfg.resume_from_checkpoint
    #     del cfg.resume_from_checkpoint

    # callbacks = []

    # if cfg.checkpoint.enabled:
    #     ckpt = Checkpointer(
    #         cfg,
    #         logdir=os.path.join(cfg.checkpoint.dir, "linear"),
    #         frequency=cfg.checkpoint.frequency,
    #         keep_prev=cfg.checkpoint.keep_prev,
    #     )
    #     callbacks.append(ckpt)

    # # wandb logging
    # if cfg.wandb.enabled:
    #     wandb_logger = WandbLogger(
    #         name=cfg.name,
    #         project=cfg.wandb.project,
    #         entity=cfg.wandb.entity,
    #         offline=cfg.wandb.offline,
    #         resume="allow" if wandb_run_id else None,
    #         id=wandb_run_id,
    #     )
    #     wandb_logger.watch(model, log="gradients", log_freq=100)
    #     wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

    #     # lr logging
    #     lr_monitor = LearningRateMonitor(logging_interval="step")
    #     callbacks.append(lr_monitor)

    # trainer_kwargs = OmegaConf.to_container(cfg)
    # # we only want to pass in valid Trainer args, the rest may be user specific
    # valid_kwargs = inspect.signature(Trainer.__init__).parameters
    # trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    # trainer_kwargs.update(
    #     {
    #         "logger": wandb_logger if cfg.wandb.enabled else None,
    #         "callbacks": callbacks,
    #         "enable_checkpointing": False,
    #         "strategy": DDPStrategy(find_unused_parameters=False)
    #         if cfg.strategy == "ddp"
    #         else cfg.strategy,
    #     }
    # )
    # trainer = Trainer(**trainer_kwargs)

    # if cfg.data.format == "dali":
    #     trainer.fit(model, ckpt_path=ckpt_path, datamodule=dali_datamodule)
    # else:
    #     trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    metrics = [] 
    for idx, batch in enumerate(test_loader):
        if idx == 0:
            metrics = model.test_step(batch, batch_idx=None, update_validation_step_outputs=False)
        else:
            metric = model.test_step(batch, batch_idx=None, update_validation_step_outputs=False)
            for k, item in metrics.items():
                item.append(metric[k])
    for k, item in metrics.items():
        metrics[k] = sum(item)/len(idx+1)
    wandb.init(project="eurosat_msi_test_accuracies", name=ckpt_path, mode="online")
    wandb.log(metrics)
    wandb.finish()

if __name__ == "__main__":
    main()