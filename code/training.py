import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from net1d import Net1D
from inception import Inception, InceptionBlock

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# weighted binary cross entropy loss
def bce_loss_class_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)
    return loss

# calculate loss # maon - training
def process_batch(batch_size, data, label, model, optimizer, weight, feature, inference=False):
    optimizer.zero_grad()
    prediction_results, att = model(data, feature)
    
    loss_function = nn.BCELoss()
    loss = 0
    corrects = 0
    label_len = [min(812, (label[b] == 3).nonzero(as_tuple=True)[0][0] if (label[b] == 3).any() else 812) for b in range(batch_size)]
    
    for batch in range(batch_size):
        batch_pred = prediction_results[batch, :, :label_len[batch]].clone().transpose(1, 0)
        batch_label = label[batch, :label_len[batch]].view([-1]).clone()
        
        loss += loss_function(batch_pred.squeeze(), batch_label)
        corrects += ((prediction_results >= 0.5) == torch.squeeze(batch_label)).sum()
    
    if not inference:
        loss.backward()
        optimizer.step()
    
    pred_label = (prediction_results >= 0.5).long().squeeze()
    return label, pred_label, prediction_results, loss.item(), corrects, sum(label_len), att.squeeze()

class ATT_inception_time(nn.Module):
    # model.py
    pass

# training
def train(dataset, test_dataset, train_feature, test_feature, max_epoch, BATCH_SIZE, LR, weight):
    net = ATT_inception_time(input_size=3, output_size=3).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    
    for epoch in range(max_epoch):
        net.train()
        train_stats = train_epoch(net, dataset, train_feature, optimizer, BATCH_SIZE, weight)
        
        net.eval()
        test_stats = test_epoch(net, test_dataset, test_feature, optimizer, BATCH_SIZE, weight)
        
        print_epoch_stats(epoch, train_stats, test_stats)
        save_model_and_attention(net, epoch, test_stats['attention'])

# training epoch
def train_epoch(model, dataset, feature, optimizer, batch_size, weight):
    epoch_loss, corrects, total_samples = 0, 0, 0
    for (input_, label), t_feature in zip(dataset, feature):
        input_, label = input_.to(device).float(), label.to(device).float()
        _, _, _, loss, batch_corrects, batch_samples, _ = process_batch(batch_size, input_, label, model, optimizer, weight, t_feature)
        epoch_loss += loss
        corrects += batch_corrects
        total_samples += batch_samples
    
    return {
        'loss': epoch_loss / len(dataset),
        'accuracy': corrects.float() / float(total_samples)
    }

# validation epoch
def test_epoch(model, dataset, feature, optimizer, batch_size, weight):
    epoch_loss, corrects, total_samples = 0, 0, 0
    true_label, true_pred = [], []
    
    with torch.no_grad():
        for (input_, label), t_feature in zip(dataset, feature):
            input_, label = input_.to(device).float(), label.to(device).float()
            label_result, pred_label, _, loss, batch_corrects, batch_samples, attention = process_batch(batch_size, input_, label, model, optimizer, weight, t_feature, inference=True)
            
            epoch_loss += loss
            corrects += batch_corrects
            total_samples += batch_samples
            
            true_label.extend(label_result.cpu().numpy()[0])
            true_pred.extend(pred_label.cpu().numpy())
    
    cohen = metrics.cohen_kappa_score(true_pred, true_label)
    f1 = f1_score(true_label, true_pred, average=None)
    
    return {
        'loss': epoch_loss / len(dataset),
        'accuracy': corrects.float() / float(total_samples),
        'cohen': cohen,
        'f1_score': f1,
        'attention': attention
    }

# training result print
def print_epoch_stats(epoch, train_stats, test_stats):
    print(f"Epoch: {epoch}")
    print(f"Train - Loss: {train_stats['loss']:.4f}, Accuracy: {train_stats['accuracy']:.4f}")
    print(f"Test - Loss: {test_stats['loss']:.4f}, Accuracy: {test_stats['accuracy']:.4f}")
    print(f"Cohen's Kappa: {test_stats['cohen']:.4f}")
    print(f"F1 Score: {test_stats['f1_score']}")
    print("=" * 70)

# model saving
def save_model_and_attention(model, epoch, attention):
    path = "PATH/RESULT/"
    torch.save(model.state_dict(), f"{path}Epoch_{epoch}_model.pth")
    np.save(f"{path}Epoch_{epoch}_Attention.npy", attention.cpu().numpy())

if __name__ == "__main__":
    # dataset_load.py
    train(dataset, test_dataset, train_feature, test_feature, max_epoch, BATCH_SIZE, LR, weight)