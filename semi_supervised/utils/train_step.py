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
    for unlabeled_views in unlabeled_loader:
        try:
            x = next(labeled_iter)

        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x = next(labeled_iter)

        labeled_data, labels = x[1], x[-1]
        n_views = len(labeled_data)
        labeled_data = torch.cat(labeled_data, dim=0)  # Shape: (32*n, 3, 224, 224)

        # Repeat the labels n times
        labels = labels.repeat(n_views)

        # Move data to device
        labeled_data = labeled_data.to(device)
        # labeled_data = [labeled_data[i].to(device) for i in range(len(labeled_data))]
        labels = labels.to(device)
        views = [v.to(device) for v in unlabeled_views[1]]

        optimizer.zero_grad()

        # Forward pass for labeled data
        logits = model(labeled_data)
        # logits = [model(labeled_data[i], return_features=False) for i in range(len(labeled_data))]
        supervised_loss = criterion(logits, labels)#np.sum(criterion(logits[i], labels) for i in range(len(logits)))

        # Forward pass for unlabeled data
        features = []
        for view in views:
          feature, _ = model(view, return_features=True)
          features.append(feature)

        # Calculate SSL loss
        ssl_loss = multiview_frossl_loss_func(features, configs.method_kwargs.invariance_weight)

        # Combined loss
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
        # predicted = [logits[i].max(1) for i in range(len(logits))]
        _, predicted = logits.max(1)
        # _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()#sum([predicted[i].eq(labels).sum().item() for i in range(len(predicted))])

    accuracy = 100. * correct / total
    wandb.log({
            'epoch': epoch,
            'Train_total_loss_epoch': total_loss/len(unlabeled_loader),
            "Train_accuracy_epoch": accuracy
        })

    if configs.scheduler.name == "warmup_cosine":
        scheduler.step()
    return total_loss / len(unlabeled_loader), accuracy
