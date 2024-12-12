from solo.losses.frossl import multiview_frossl_loss_func
import numpy as np
import torch
import wandb
def train_step(model, labeled_loader, unlabeled_loader, optimizer, scheduler, criterion, device, epoch, configs):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    labeled_iter = iter(labeled_loader)
    #since D_u should have more saples, we iterate over it accounting that D_l loader might have to restart
    for unlabeled_views in unlabeled_loader:
        try:
            x = next(labeled_iter)

        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x = next(labeled_iter)

        labeled_data, labels = x[1], x[-1]
        n_views = len(labeled_data)
        labeled_data = torch.cat(labeled_data, dim=0)  # Shape: (batch_size*n_views, 3, 224, 224)

        # Repeat the labels n_views times
        labels = labels.repeat(n_views)

        labeled_data = labeled_data.to(device)

        labels = labels.to(device)
        views = [v.to(device) for v in unlabeled_views[1]]

        optimizer.zero_grad()

        # Forward pass for labeled data to get logits
        logits = model(labeled_data)

        supervised_loss = criterion(logits, labels)

        # Forward pass for unlabeled data to get embeddings
        features = []
        for view in views:
          feature, _ = model(view, return_features=True)
          features.append(feature)

        # Calculate SSL loss with unlabeled embeddings
        ssl_loss = multiview_frossl_loss_func(features, configs.method_kwargs.invariance_weight)

        # total loss
        loss = supervised_loss + configs.ssl_weight*ssl_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'Train_total_loss_step': loss.item(),
            'Train_supervised_loss_step': supervised_loss.item(),
            'Train_ssl_loss_step': ssl_loss.item()
        })

        _, predicted = logits.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    wandb.log({
            'epoch': epoch,
            'Train_total_loss_epoch': total_loss/len(unlabeled_loader),
            "Train_accuracy_epoch": accuracy
        })

    if configs.scheduler.name == "warmup_cosine":
        scheduler.step()
    return total_loss / len(unlabeled_loader), accuracy
