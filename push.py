import numpy as np
from torch.utils.data import DataLoader
from model import ProtoPNet
import torch
from tqdm import tqdm
from model.receptive_field import compute_rf_prototype
import cv2
import os 
import matplotlib.pyplot as plt

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

def push(model: ProtoPNet, train_push_loader: DataLoader, test_loader: DataLoader, visualize: bool = False): 

    model.eval()
    prototype_shape = model.prototype_shape
    n_prototypes, _, proto_h, proto_w = prototype_shape

    # global_min_proto_dist[i] closest distance from prototype to any patch in any sample of corresponding class
    global_min_proto_dist = np.full(n_prototypes, np.inf)  # (2000)

    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(prototype_shape)   # (2000, 128, 1, 1)

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)

    for push_iter, (search_batch, search_y) in enumerate(tqdm(train_push_loader, desc="push")): 

        start_index_of_search_batch = push_iter * train_push_loader.batch_size

        with torch.no_grad(): 
            search_batch = search_batch.cuda() 

            # we will need access to the patches themselves, along with ALL distances 
            # from each patch to each prototype, BEFORE minimizing (so we can track index) 
            # latent features, distances
            protoL_input_torch, proto_dist_torch = model.push_forward(search_batch) # (B, 128, 7, 7), (B, 2000, 7, 7) 

        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

        del protoL_input_torch, proto_dist_torch

        # we want to focus on each class, so keep a mapping from each class index to the sample index of the batch
        class_to_img_index_dict = {key: [] for key in range(model.num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)


        # now iterate through prototypes 
        for j in range(n_prototypes): 
            target_class = torch.argmax(model.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue

            # take the distances from prototypes to all patches in a batch  
            # filter out the samples in batch by target class, say B = 70 and there are K samples left corresponding to prototype j 
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]] # (K, 2000, 7, 7)

            # now just focus on the prototype j 
            proto_dist_j = proto_dist_j[:,j,:,:]    # (K, 7, 7)
            # proto_dist_j[a, b, c] = distance of prototype j on the (b, c)th patch of sample a in our batch 

            # now just find the minimum distance, and update the global dist with the correct index and patch if it is smaller 
            batch_min_proto_dist_j = np.amin(proto_dist_j) 
            if batch_min_proto_dist_j < global_min_proto_dist[j]: 
                # this is the specific index (a, b, c) \in (K, 7, 7) corresponding to new min
                batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))

                # change the argmin index from the index among images of the target class to the index in the entire search batch
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]] 

                # location of best patch so far 
                img_index_in_batch, fmap_height_start_index, fmap_width_start_index = batch_argmin_proto_dist_j
                fmap_height_end_index, fmap_width_end_index = fmap_height_start_index + 1, fmap_width_start_index + 1 

                # now grab the specific patch of shape (128, 1, 1) from the batch 
                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                       :,
                                                       fmap_height_start_index:fmap_height_end_index,
                                                       fmap_width_start_index:fmap_width_end_index]
                # finally update the global minimmizers
                global_min_proto_dist[j] = batch_min_proto_dist_j
                global_min_fmap_patches[j] = batch_min_fmap_patch_j 

            """
            Everything past this is visualization. Can ignore this if you don't want to worry about implementing visualization. 
            """

            if visualize: 
                # get the receptive field boundary of the image patch that generates the representation
                rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, model.proto_layer_rf_info)
                
                # get the whole image
                original_img_j = search_batch[rf_prototype_j[0]]
                original_img_j = original_img_j.clone().cpu().numpy()
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]
                
                # crop out the receptive field
                rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                          rf_prototype_j[3]:rf_prototype_j[4], :]
                
                # save the prototype receptive field information
                proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
                proto_rf_boxes[j, 1] = rf_prototype_j[1]
                proto_rf_boxes[j, 2] = rf_prototype_j[2]
                proto_rf_boxes[j, 3] = rf_prototype_j[3]
                proto_rf_boxes[j, 4] = rf_prototype_j[4]
                if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                    proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                # find the highly activated region of the original image
                proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
                proto_act_img_j = 128 - proto_dist_img_j
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                 interpolation=cv2.INTER_CUBIC)
                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

                # save the prototype boundary (rectangular boundary of highly activated region)
                proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
                proto_bound_boxes[j, 1] = proto_bound_j[0]
                proto_bound_boxes[j, 2] = proto_bound_j[1]
                proto_bound_boxes[j, 3] = proto_bound_j[2]
                proto_bound_boxes[j, 4] = proto_bound_j[3]
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

                # overlay (upsampled) self activation on original image and save the result
                rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                plt.imsave(os.path.join("saved", f"prototype_{j}.png"), overlayed_original_img_j, vmin=0.0, vmax=1.0)
                    
    # now that we have all patches for which prototypes should get pushed to, push. 
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    model.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())

    test_total_correct = 0  

    for _, (image, label) in enumerate(tqdm(test_loader, desc="test")): 
        image = image.cuda(); label = label.cuda()
        with torch.no_grad(): 
            logits, min_distances = model(image)  

            ce_loss = torch.nn.functional.cross_entropy(logits, label) 
            _, predicted = torch.max(logits.data, 1)
            test_total_correct += (predicted == label).sum().item()


    print(f"  Test Acc: {test_total_correct/len(test_loader.dataset) * 100}")

