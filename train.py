from model import ProtoPNet
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch
from tqdm import tqdm
import time

def warm_only(model: ProtoPNet):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True

    print("warm")


def joint(model: ProtoPNet):
    for p in model.backbone.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    print("joint")

def last_only(model: ProtoPNet):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    print('last layer')

def train(model: ProtoPNet, train_loader: DataLoader, test_loader: DataLoader, optimizer: Optimizer): 

    train_total_correct = 0  
    train_total_cross_entropy = 0 
    train_total_cluster_cost = 0 
    train_total_separation_cost = 0

    for _, (image, label) in enumerate(tqdm(train_loader, desc="train")): 
        label_cpu = label
        image = image.cuda(); label = label.cuda()

        with torch.enable_grad(): 
            logits, min_distances = model(image) # logits: (B, 200), min_distances: (B, 200, 2000)

            ce_loss = torch.nn.functional.cross_entropy(logits, label) 

            max_dist = 128 # shouldn't be hardcoded, idk why it's set to what it is in original repo 
            _, predicted = torch.max(logits.data, 1)
            train_total_correct += (predicted == label).sum().item()

            # calculate cluster cost by looking at minimum distances between prototype and nearest patch. 
            # 1. Filter prototype_class_identity: (2000, 200) -> (2000, B). It consists of rows that represent each prototype. 
            # 2. If we transpose it, then (B, 2000). Given the ith sample from batch, matrix[i] is a vector that 
            # shows all prototypes associated with it of form: [0 ... 0 1 ... 1 0 ... 0]
            prototypes_of_correct_class = torch.t(model.prototype_class_identity[:, label_cpu]).cuda() # (B, 2000)
            # prototypes_of_correct_class[i][j] = 1 if jth prototype corresponds to class in ith sample of batch

            # we have the min_distances of shape (B, 2000). We don't care about the distances from ith sample to a prototype that isn't associated with it, 
            # so we just mask it by doing A = min_distances * prototypes_of_correct_class, of shape (B, 2000) 
            # A[i][j] is the distance from the jth prototype to the class that the prototype corresponds to. 
            # invert them and then compute the max (so min distance) across the prototype dimension. This ignores the majority of distances which are 0. 
            # and finds the prototype that has the min distance. 
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)

            # we can't just take the min of the masked min_distances directly since it will always be 0 
            # we want to minimize only over the elements that are not masked
            # this is why we take the max of inverted distances, mask them out, and then invert again. 
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            l1_cost = model.last_layer.weight.norm(p=1)

            total_loss = ce_loss + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1_cost

            train_total_cross_entropy += ce_loss.item() 
            train_total_cluster_cost += cluster_cost.item()  
            train_total_separation_cost += separation_cost.item()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        del image
        del label
        del predicted
        del min_distances

    # You should divide by the num batches rather than num samples.  

    print(f"  Cross Entropy : {train_total_cross_entropy/len(train_loader)}")
    print(f"  Cluster       : {train_total_cluster_cost/len(train_loader)}")
    print(f"  Seperation    : {train_total_separation_cost/len(train_loader)}")
    print(f"  Train Acc     : {train_total_correct/len(train_loader.dataset) * 100}")
    print(f"  L1 Cost       : {model.last_layer.weight.norm(p=1)}")

    test_total_correct = 0  

    for _, (image, label) in enumerate(tqdm(test_loader, desc="test")): 
        image = image.cuda(); label = label.cuda()
        with torch.no_grad(): 
            logits, min_distances = model(image)  

            ce_loss = torch.nn.functional.cross_entropy(logits, label) 
            _, predicted = torch.max(logits.data, 1)
            test_total_correct += (predicted == label).sum().item()


    print(f"  Test Acc: {test_total_correct/len(test_loader.dataset) * 100}")


