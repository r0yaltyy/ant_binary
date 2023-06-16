import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib import style

traindir = '/home/alex/ant_binary/train2'
testdir = '/home/alex/ant_binary/validation2'

#transformations
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),                                
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

#datasets
train_data = datasets.ImageFolder(traindir,transform=train_transforms)
test_data = datasets.ImageFolder(testdir,transform=test_transforms)

#dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)

def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step



device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18()

#freeze all params
for params in model.parameters():
  params.requires_grad_ = False

#add a new final layer
nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)

model = model.to(device)

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(model.fc.parameters()) 

#train step
train_step = make_train_step(model, optimizer, loss_fn)




losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []
epoch_test_losses_plt = []
epoch_train_losses_plt = []

n_epochs = 100
early_stopping_tolerance = 3
early_stopping_threshold = 0.03
accuracy = []

plt.ion()
plt.show(block=False)
for epoch in range(n_epochs):
  epoch_loss = 0
  for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate ove batches
    x_batch , y_batch = data
    x_batch = x_batch.to(device) #move to gpu
    y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
    y_batch = y_batch.to(device) #move to gpu

    loss = train_step(x_batch, y_batch)
    epoch_loss += loss/len(trainloader)
    losses.append(loss)
    
  epoch_train_losses.append(epoch_loss)
  epoch_train_losses_plt.append(epoch_loss.cpu().data.numpy())
  print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

  #validation doesnt requires gradient
  with torch.no_grad():
    cum_loss = 0
    for x_batch, y_batch in testloader:
      x_batch = x_batch.to(device)
      y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
      y_batch = y_batch.to(device)

      #model to eval mode
      model.eval()

      yhat = model(x_batch)
      val_loss = loss_fn(yhat,y_batch)
      cum_loss += loss/len(testloader)
      val_losses.append(val_loss.item())


    epoch_test_losses.append(cum_loss)
    epoch_test_losses_plt.append(cum_loss.cpu().data.numpy())
    print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
    
    best_loss = min(epoch_test_losses)
    
    #save best model
    if cum_loss <= best_loss:
      best_model_wts = model.state_dict()
    
    #early stopping
    early_stopping_counter = 0
    if cum_loss > best_loss:
      early_stopping_counter +=1

    if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
      print("/nTerminating: early stopping")
      break #terminate training
  
  yhat = model(x_batch)
  acc = (torch.argmax(yhat, 1) == y_batch).float().mean()
  acc = float(acc)
  print("Model accuracy: %.2f%%" % (acc*100))
  accuracy.append(acc)
  plt.cla()
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.plot(epoch_train_losses_plt, color='red', label='train', linestyle='-')
  plt.plot(epoch_test_losses_plt, color='blue', label='test', linestyle='--')
  plt.plot(accuracy, color='green', label='accuracy', linestyle='-.')
  plt.legend(loc="best")
  plt.gcf().canvas.draw_idle()
  plt.gcf().canvas.start_event_loop(0.3)

#load best model
model.load_state_dict(best_model_wts)


def inference(test_data):
  idx = torch.randint(1, len(test_data), (1,))
  sample = torch.unsqueeze(test_data[idx][0], dim=0).to(device)

  if torch.sigmoid(model(sample)) < 0.5:
    print("Prediction : Agrressive")
  else:
    print("Prediction : Not agrressive")


  #plt.imshow(test_data[idx][0].permute(1, 2, 0))

inference(test_data)

'''yhat = model(x_batch)
acc = (torch.argmax(yhat, 1) == y_batch).float().mean()
acc = float(acc)*100
print("Model accuracy: %.2f%%" % acc)
print("Accuracy for each epoch: ", accuracy)
'''
plt.savefig('nn_loss.png')
wait = input("Press Enter to continue.")
