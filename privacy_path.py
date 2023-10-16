import torch
import torch.nn.functional as F
from tqdm import tqdm

# Class consistency Loss
def class_loss(label_classifer, imgs, label, device):
    loss = 0
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1,3,1,1)
    label = torch.tensor(label,device=device)
    for img in imgs.split(4):
        out = label_classifer(img)
        loss +=  F.cross_entropy(out,label.repeat(out.shape[0]))
    return loss

# Privacy preserving Loss
def privacy_loss(id_net, imgs):
    loss = 0
    if imgs.shape[1] == 1:
        imgs = imgs.repeat(1,3,1,1)
    for img in imgs.split(4):
        out = id_net(img)
        loss +=  F.kl_div(F.log_softmax(out, dim=1),torch.ones(out.shape[1],device=out.device)/out.shape[1],reduction='batchmean') 
    return loss

# Equidistance Loss
def equidistance_loss(learned_path, w_a, w_b):
    temp = [w_a.squeeze(0),learned_path, w_b.squeeze(0)]
    full_path = torch.cat(temp)
    distances = F.pairwise_distance(full_path[torch.arange(0,full_path.shape[0]-1)],
                                    full_path[torch.arange(1,full_path.shape[0])])
    loss = distances.square().sum()
    return loss

# returns a linear path between w_a and w_b
def get_linear_path(w_a, w_b, n_steps, z_dim=512):
    device = w_a.device
    assert w_a.device == w_b.device
    linear_path_weights = torch.linspace(0, 1, n_steps, device=device) # [steps, 1, z_dim]
    linear_path = torch.zeros([n_steps, z_dim], device=device)
    for i,weight in enumerate(linear_path_weights):
        int_w = w_a + weight*(w_b - w_a)
        linear_path[i] = int_w
    return linear_path


def get_privacy_path(w_a, w_b, id_net, label_classifier, G, n_steps, label, lr = 0.1, optim_steps = 500, verbose=False):
    '''
    w_a: starting point of the path
    w_b: ending point of the path
    id_net: privacy preserving network
    label_classifier: classifier network
    G: GAN generator
    n_steps: number of steps in the path
    label: label of the generated images
    lr: learning rate for the optimizer
    optim_steps: number of optimization steps
    verbose: print loss at each step
    '''
    device = w_a.device
    assert w_a.device == w_b.device
    #init path to linear path
    privacy_path = get_linear_path(w_a,w_b,n_steps)

    # starting and ending points are fixed
    privacy_path = privacy_path[torch.arange(1,privacy_path.shape[0]-1)]
    privacy_path.requires_grad_(True)

    # Optimize path
    optimizer = torch.optim.Adam([privacy_path], betas=(0.9, 0.999), lr=lr)
    for i in tqdm(range(optim_steps), desc='Optimizing privacy path', leave=False, position=1, disable=not verbose):
        optimizer.zero_grad()
        imgs = G.synthesis(privacy_path.unsqueeze(1).repeat([1, G.mapping.num_ws, 1]), noise_mode='const')        
        imgs = imgs.clamp(-1, 1)
        loss_eq = equidistance_loss(privacy_path, w_a, w_b)
        loss_privacy = privacy_loss(id_net, imgs)
        loss_class = class_loss(label_classifier, imgs, label, device=device)
        loss = loss_privacy*0.1 + loss_eq + loss_class
        if verbose:
            print("Step {} dist_loss: {} classifier_loss: {} label_loss: {} total_loss: {}".format(i,loss_eq.item(),loss_privacy.item(),loss_class.item(),loss.item())) 
        loss.backward()
        optimizer.step()

    privacy_path = torch.cat([w_a.squeeze(0),privacy_path,w_b.squeeze(0)])
    return privacy_path
