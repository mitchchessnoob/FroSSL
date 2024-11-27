from solo.utils.lars import LARS
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
def create_optimizer_and_scheduler(model, configs):
    classifier_params = []
    other_params = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {'params': other_params, 'lr': configs.optimizer.lr},
        {'params': classifier_params, 'lr': configs.optimizer.classifier_lr}
    ]

    # Initialize LARS optimizer
    optimizer = LARS(
        param_groups,
        lr=configs.optimizer.lr,
        weight_decay=configs.optimizer.weight_decay,
        eta=configs.optimizer.kwargs.eta,
        clip_lr=configs.optimizer.kwargs.clip_lr,
        exclude_bias_n_norm=configs.optimizer.kwargs.exclude_bias_n_norm
    )

    # Create scheduler
    if configs.scheduler.name == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=configs.scheduler.warmup_epochs,  # You might want to add these to your config
            max_epochs=configs.max_epochs,    # You might want to add these to your config
            warmup_start_lr=0.0
        )

    return optimizer, scheduler
