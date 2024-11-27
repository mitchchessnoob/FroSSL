def evaluate(model, test_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)

    wandb.log({
        'epoch': epoch,
        'test_loss': avg_loss,
        'test_accuracy': accuracy
    })

    return avg_loss, accuracy
