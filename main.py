import torch 
from model.protopnet import ProtoPNet
import train
import push
from datasets import train_loader, test_loader, train_push_loader
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__": 

    model = ProtoPNet(224).cuda()

    warm_optimizer = torch.optim.Adam([
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.prototype_vectors, 'lr': 3e-3},
    ])

    joint_optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},
        {'params': model.prototype_vectors, 'lr': 3e-3},
    ])

    last_layer_optimizer = torch.optim.Adam([
        {'params': model.last_layer.parameters(), 'lr': 1e-4}
    ])

    push_epochs = [(i+1) * 10 for i in range(10)]

    # warm 
    train.warm_only(model) 

    for _ in range(5):
        train.train(model, train_loader, test_loader, warm_optimizer)
    
    for joint_epoch in range(100): 
        train.joint(model) 
        train.train(model, train_loader, test_loader, joint_optimizer) 

        if joint_epoch in push_epochs: 
            train.last_only(model)
            push.push(model, train_push_loader, test_loader)

    # push last time and generate visuals
    train.last_only(model)
    push.push(model, train_push_loader, test_loader, visualize=True)

    train.last_only(model) 
    for _ in range(5): 
        train.train(model, train_loader, test_loader, last_layer_optimizer) 
