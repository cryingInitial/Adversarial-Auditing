import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools

from models import Models
from utils.data import load_data
from utils.audit import compute_eps_lower_from_mia
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from copy import deepcopy
import scienceplots
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_models', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eps', type=float, default=10.0)
    parser.add_argument('--target', type=str, default='blank')
    parser.add_argument('--num_trials', type=int, default=10)

    args = parser.parse_args()

    return args

def get_attack_results(target='blank', seeds=[0], eps=4.0, methods=['subtract', 'fisher', 'bhattacharyya'], alpha=0.2, epochs=10, lr=0.1, num_models=256):
    dataset = 'mnist'
    os.makedirs(f'results/{dataset}_{target}/white_{eps}', exist_ok=True)

    directories = []
    for seed in seeds:
        directories.append(f'exp_data/{dataset}_{target}_100/seed{seed}/{dataset}_cnn_eps{eps}') 

    models_in_whole, models_out_whole = load_models(directories, seeds, dataset)

    np.random.shuffle(models_in_whole)
    np.random.shuffle(models_out_whole)

    models_in = models_in_whole[:num_models]
    models_out = models_out_whole[:num_models]

    models_in_eval = models_in_whole[num_models:2*num_models]
    models_out_eval = models_out_whole[num_models:2*num_models]

    print(f'Loaded {len(models_in)} in models and {len(models_out)} out models')
    print(f'Loaded {len(models_in_eval)} in eval models and {len(models_out_eval)} out eval models')

    criterion = nn.CrossEntropyLoss()
    base_eps, Subtract_eps, Ours_eps, Fisher_eps, Bhattacharyya_eps = 0, 0, 0, 0 ,0

    base_eps = get_eps_from_loss(models_in_eval, models_out_eval, dataset, criterion)

    for method in methods:
            
        fixed_input = torch.randn(1, 1, 28, 28).to('cuda') if dataset == 'mnist' else torch.randn(1, 3, 32, 32).to('cuda')
        y = torch.tensor([9]).to('cuda')

        X = torch.nn.Parameter(torch.zeros(1, 1, 28, 28).to('cuda'), requires_grad=True) if dataset == 'mnist' else torch.nn.Parameter(torch.ones(1, 3, 32, 32).to('cuda'), requires_grad=True)
        optimizer_input = optim.Adam([X], lr=lr)

        models_train_in = models_in
        models_train_out = models_out
        
        for epoch in tqdm(range(epochs)):
            optimizer_input.zero_grad()
                
            in_outputs = []; out_outputs = []
            for model_in, model_out in zip(models_train_in, models_train_out):
                    
                in_output = model_in(X.to('cuda'))
                out_output = model_out(X.to('cuda'))

                in_outputs.append(in_output)
                out_outputs.append(out_output)

            losses = []
            losses_ins = []
            losses_outs = []
            mean_of_out_outputs_loss = sum([criterion(out_output, y) for out_output in out_outputs]) / len(out_outputs)
            in_losses = torch.stack([criterion(in_output, y) for in_output in in_outputs])
            out_losses = torch.stack([criterion(out_output, y) for out_output in out_outputs])
            
            if method == 'subtract':
                for in_output, out_output in zip(in_outputs, out_outputs):
                        
                    loss = criterion(in_output, y) - criterion(out_output, y)
                    losses.append(loss)

                loss = sum(losses)
            
            elif method == "fisher":
                in_mean, in_var = get_gaussian(in_losses)
                out_mean, out_var = get_gaussian(out_losses)

                loss = -((in_mean - out_mean).pow(2) / (in_var + out_var + 1e-8)).sum()
                    
            elif method == 'bhattacharyya':
                in_mu, in_var = get_gaussian(in_losses)
                out_mu, out_var = get_gaussian(out_losses)

                distance = 1/4 * torch.log(1/4 * (in_var/out_var + out_var/in_var + 2)) + 1/4 * ((in_mu - out_mu)**2 / (in_var + out_var))
                loss = -distance

            loss.backward()
            
            optimizer_input.step()
            X.data.clamp_(0, 1)
            
            print(f'epoch: {epoch}, loss: {loss.item()}')

            final_X = X.detach()
            final_y = None

        os.makedirs(f'results/{dataset}_{target}/white_{eps}', exist_ok=True)
        if method == 'subtract': 
            Subtract_eps = get_eps_from_loss(models_in_eval,models_out_eval,dataset,criterion,inputX=final_X,inputy=final_y)

        elif method == 'fisher':
            Fisher_eps = get_eps_from_loss(models_in_eval,models_out_eval,dataset,criterion,inputX=final_X,inputy=final_y)

        elif method == 'bhattacharyya':
            Bhattacharyya_eps = get_eps_from_loss(models_in_eval,models_out_eval,dataset,criterion,inputX=final_X,inputy=final_y)

    os.makedirs(f'results/{dataset}_{target}/white_{eps}', exist_ok=True)
    with open(f'results/{dataset}_{target}/white_{eps}/epochs{epochs}.txt', 'a') as f:
        f.write(f'[Num Models {num_models} LOG] Base eps: {base_eps}, Subtract_eps: {Subtract_eps}, Ours_eps: {Ours_eps}, fisher_eps:{Fisher_eps}, bhattacharyya_eps:{Bhattacharyya_eps} lr: {lr}\n')

def get_gaussian(losses):
    mu = torch.mean(losses)
    sigma = torch.std(losses)
    var = sigma ** 2
    return mu, var


def get_eps_from_loss(models_in, models_out, dataset, criterion, inputX=None, inputy=None):

    in_outputs = []
    out_outputs = []

    for model_in, model_out in zip(models_in, models_out):
        inpX = inputX if inputX is not None else torch.zeros(1, 1, 28, 28).cuda() if dataset == 'mnist' else torch.zeros(1, 3, 32, 32).cuda()
        inpY = inputy if inputy is not None else torch.tensor([9]).cuda()
        
        if inputX is not None: inpX = torch.clamp(inpX, min=0, max=1)
        if inputy is not None: inpY = torch.nn.functional.softmax(inpY, dim=1)

        in_output = model_in(inpX)
        out_output = model_out(inpX)

        in_target = deepcopy(inpY)
        out_target = deepcopy(inpY)

        in_outputs.append(-criterion(in_output, in_target).item())
        out_outputs.append(-criterion(out_output, out_target).item())

    mia_scores = np.concatenate([in_outputs, out_outputs])
    mia_labels = np.concatenate([np.ones_like(in_outputs), np.zeros_like(out_outputs)])
    _, emp_eps_loss = compute_eps_lower_from_mia(mia_scores, mia_labels, 0.05, 1e-5, 'GDP', n_procs=1)

    in_outputs = - np.array(in_outputs)
    out_outputs = - np.array(out_outputs)

    return emp_eps_loss

def load_models(directories, comb, dataset):
    models_in = []
    models_out = []
    directories = [directories[i] for i in comb]
    for directory in directories:
        for model_loc in os.listdir(f'{directory}/models'):
            if dataset == 'mnist': model = Models['cnn'](in_shape=(1, 1, 28, 28), out_dim=10).to('cuda')

            model.load_state_dict(torch.load(os.path.join(f'{directory}/models', model_loc),weights_only=True))
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            if 'in' in model_loc: models_in.append(model)
            elif 'out' in model_loc: models_out.append(model)

    print(f'Loaded {len(models_in)} in models and {len(models_out)} out models')

    return models_in, models_out

if __name__ == '__main__':
    args = get_args()
    for _ in range(args.num_trials):
        get_attack_results(
            target=args.target,
            eps=args.eps,
            epochs=args.epochs,
            lr=args.lr,
            num_models=args.num_models,
        )