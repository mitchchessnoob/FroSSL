def main():
  model = segmentation_model.cuda()
  num_epochs = 5
  criterion = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(
      filter(lambda p: p.requires_grad, model.parameters()),
      lr=0.01,
      momentum=0.9,
      weight_decay=1e-4
  )
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,78,eta_min=0.001)
  best_acc = 0
  if wandb.api.api_key is None:
    wandb.login(key="57e49312fc462a736d24abd32cc7891d91258b76")
  try:
      wandb.init(project="Weakly-Supervised Object Localization", id="cifar10 resnet18 not pretrained",\
                 resume="auto", settings=wandb.Settings(init_timeout=300),\
                 config= {"learning_rate": 0.01,
                          "architecture": "resnet18 not pretrained",
                          "dataset": "CIFAR-10",
                          "epochs": num_epochs,
                          },
                 mode = "online"
  
                 )
  
      losses = []
      acces = []
      v_losses = []
      v_acces = []
  
      for epoch in tqdm(range(num_epochs)):
          epoch_loss = 0.0
          acc = 0.0
          var = 0.0
          model.train()
          train_pbar = train_iter
          for i, (x, _label) in enumerate(train_pbar):
              x = x.cuda(non_blocking=True)
              _label = _label.cuda(non_blocking=True)
              label = F.one_hot(_label, num_classes=10).float()
              seg_out = model(x)
  
              attn = attention(seg_out)
              # Smooth Max Aggregation
              logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
              loss = criterion(logit, label)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              lr_scheduler.step()
              epoch_loss += loss.item()
              acc += (logit.argmax(-1)==_label).sum()
  
              wandb.log({
                'train_loss_step': loss,
                'train_accuracy_step': acc/(i*len(_label))
              })
              del x, _label, label, seg_out, attn, logit, loss
              #train_pbar.set_description('Accuracy: {:.3f}%'.format(100*(logit.argmax(-1)==_label).float().mean()))
  
          avg_loss = epoch_loss / (i + 1)
          losses.append(avg_loss)
          avg_acc = acc.cpu().detach().numpy() / (len(trainset))
          acces.append(avg_acc)
          wandb.log({
                'epoch': epoch,
                'train_loss_epoch': avg_loss,
                'train_accuracy_epoch': avg_acc
              })
          model.eval()
  
          epoch_metrics = {
              'loss': [],
              'accuracy': [],
              'mean_coverage': [],
              'peak_response': [],
              'smoothness': [],
              'peak_to_background': [],
              'attention_contiguity': [],
              'class_attention_std': [],
              'activation_consistency': []
          }
          test_pbar = tqdm(test_iter)
          for i, (x, _label) in enumerate(test_pbar):
              x = x.cuda(non_blocking=True)
              _label = _label.cuda(non_blocking=True)
              label = F.one_hot(_label, num_classes = 10).float()
              seg_out = model(x)
              attn = attention(seg_out)
              logit = torch.log(torch.exp(seg_out*0.5).mean((-2,-1)))*2
              loss = criterion(logit, label)
  
              acc = (logit.argmax(-1)==_label).sum()/len(_label)
  
              attention_stats = calculate_attention_metrics(attn, seg_out, _label)
  
  
              wandb.log({
                  'step': i + epoch * len(test_iter),
                  'test/step_loss': loss.item(),
                  'test/step_accuracy': acc,
                  'attention/step_mean_coverage': attention_stats['mean_coverage'],
                  'attention/step_peak_response': attention_stats['peak_response'],
                  'attention/step_smoothness': attention_stats['smoothness'],
                  'attention/step_peak_to_background': attention_stats['peak_to_background'],
                  'attention/step_contiguity': attention_stats['attention_contiguity'],
                  'attention/step_class_std': attention_stats['class_attention_std'],
                  'attention/step_activation_consistency': attention_stats['activation_consistency']
              })
  
              for key in attention_stats:
                  epoch_metrics[key].append(attention_stats[key])
              epoch_metrics['loss'].append(loss.item())
              epoch_metrics['accuracy'].append(acc)
  
              test_pbar.set_description(
                  f'Test Acc: {100*acc:.2f}% | Loss: {loss.item():.4f}'
              )
  
              del  _label, label, logit, loss, acc
              torch.cuda.empty_cache()
  
  
          plt.close('all')
  
          epoch_averages = {
              key: np.mean(safe_tensor_to_numpy(values))
              for key, values in epoch_metrics.items()
          }
  
          # Log epoch metrics to wandb
          wandb.log({
              'epoch': epoch,
              'test/epoch_loss': epoch_averages['loss'],
              'test/epoch_accuracy': epoch_averages['accuracy'],
              'attention/epoch_mean_coverage': epoch_averages['mean_coverage'],
              'attention/epoch_peak_response': epoch_averages['peak_response'],
              'attention/epoch_smoothness': epoch_averages['smoothness'],
              'attention/epoch_peak_to_background': epoch_averages['peak_to_background'],
              'attention/epoch_contiguity': epoch_averages['attention_contiguity'],
              'attention/epoch_class_std': epoch_averages['class_attention_std'],
              'attention/epoch_activation_consistency': epoch_averages['activation_consistency']
          })
  
          if epoch_averages['accuracy'] > best_acc:
            best_acc = epoch_averages['accuracy']
            torch.save(model.state_dict(), './model_not_pretrained.pth')
  
          conf = torch.max(nn.functional.softmax(seg_out, dim=1), dim=1)[0]
          hue = (torch.argmax(seg_out, dim=1).float() + 0.5)/10
          x -= x.min()
          x /= x.max()
          gs_im = x.mean(1)
          gs_mean = gs_im.mean()
          gs_min = gs_im.min()
          gs_max = torch.max((gs_im-gs_min))
          gs_im = (gs_im - gs_min)/gs_max
          hsv_im = torch.stack((hue.float(), attn.squeeze().float(), gs_im.float()), -1)
          im = hsv_to_rgb(hsv_im.cpu().detach().numpy())
          ex = make_grid(torch.tensor(im).permute(0,3,1,2), normalize=True, nrow=25)
          attns = make_grid(attn, normalize=False, nrow=25)
          attns = attns.cpu().detach()
          inputs = make_grid(x, normalize=True, nrow=25).cpu().detach()
          display.clear_output(wait=True)
          plt.figure(figsize=(20,8))
          plt.imshow(np.concatenate((inputs.numpy().transpose(1,2,0),ex.numpy().transpose(1,2,0), attns.numpy().transpose(1,2,0)), axis=0))
  
          plt.xticks(np.linspace(18,324,10), classes)
          plt.xticks(fontsize=20)
          plt.yticks([])
          plt.title('CIFAR10 Epoch:{:02d}, Train:{:.3f}, Test:{:.3f}'.format(epoch, avg_acc, epoch_averages['accuracy']), fontsize=20)
          display.display(plt.gcf())
  
          torch.cuda.empty_cache()
          gc.collect()
      wandb.finish()
  except Exception as e:
      wandb.finish()
      raise e
  
