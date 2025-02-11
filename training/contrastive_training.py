import sys
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/transforms')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/loader')
sys.path.insert(1, 'D:/Python script/transformer/lsfb_transfo/models')
import BackboneConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import  encoder
from lightly.loss import NTXentLoss
import torch.optim as optim


class ContrastiveTraining():

    def __init__(self,model,epochs,train_loader):
        self.model = model
        self.criterion = NTXentLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = epochs
        self.train_loader = train_loader
        
    @staticmethod    
    def plot_loss(epoch_losses):
        
         plt.figure(figsize=(8, 6))
         plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Training Loss')
         plt.xlabel('Epoch')
         plt.ylabel('Loss')
         plt.title('Training Loss per Epoch')
         plt.grid(True)
         plt.legend()
         plt.savefig('Training_loss_simclr_others_essai_4.png')
         plt.show()
    
                
    def train(self):
        epoch_losses = [] 
        for epoch in range(self.epochs):
           running_loss = 0.0
           for i, (data, target) in enumerate(tqdm(self.train_loader)):
              data = data.to('cuda')
              z1, z2 =  self.model.forward(data)
              loss = self.criterion(z1, z2)
              loss.backward()
              self.optimizer.step()
              running_loss += loss.item()
              epoch_loss = running_loss / len(self.train_loader)
           epoch_losses.append(epoch_loss)
        self.plot_loss(epoch_losses)
        return self.model.backbone
        