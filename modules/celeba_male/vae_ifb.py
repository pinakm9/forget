import torch
from tqdm import tqdm
import numpy as np
import os, sys
import time
from torch.autograd import grad
import copy


sys.path.append(os.path.abspath('../modules'))
import utility as ut
import vae_loss as vl
import vae_train as vt
import vae_ortho as vo
import vae_viz as viz
import classifier as cl


def get_processor(net, old_net, z_e, trainable_params, identifier, z_random, weights, optim, all_classes, forget_class):
    device = next(net.parameters()).device
    def process_batch():
        
        # real_img_retain = net.encode(z)
        # real_img_forget = old_net.encode(z)

        kl_weight, uniformity_weight = weights
        
        # time_0 = time.time()
        optim.zero_grad()

        # reconstructed_forget, mu_forget, logvar_forget = net(real_img_forget)
        # reconstructed_retain, mu_retain, logvar_retain = net(real_img_retain)

        
        # rec_forget = vl.reconstruction_loss(reconstructed_forget, real_img_forget)
        # rec_retain = vl.reconstruction_loss(reconstructed_retain, real_img_retain)
        # kl_forget = vl.kl_div(mu_forget, logvar_forget)
        # kl_retain = vl.kl_div(mu_retain, logvar_retain)
        # time_1 = time.time()


        # generated_img = net.decode(z_random)
        # logits = identifier(generated_img)
        # uniformity = vl.uniformity_loss_surgery(logits, all_classes=all_classes, forget_class=forget_class)
        # time_2 = time.time()

        
        # loss_forget = rec_forget + kl_weight * kl_forget + uniformity_weight * uniformity
        # gf = torch.cat([g.view(-1) for g in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
        # time_3 = time.time()
        
        # loss_retain = rec_retain + kl_weight * kl_retain + uniformity_weight * uniformity
        # gr = torch.cat([g.view(-1) for g in grad(outputs=loss_retain, inputs=trainable_params, retain_graph=True)]) 


        # Remove the component of gr that is parallel to g1.
        # (An epsilon is added to avoid division by zero.)
        # gfgr = gf @ gr
        # gfgf = gf @ gf

        # gr = gr - gf * gfgr / gfgf 
        time_4 = time.time()

        z = torch.randn(z_random.shape).to(device)
        z_hat = vl.hat(z, z_e)
        s = vl.sim(z, z_e)
        gz = net.decode(z)
        fz = old_net.decode(z)
        fz_hat = old_net.decode(z_hat)

        l_recon = vl.L_recon(gz, fz, s).mean()
        l_unlearn = vl.L_unlearn(gz, fz_hat, s).mean()
        l_percep = vl.L_percep(gz, fz_hat, s).mean() 

        loss = l_recon + 1. * (l_unlearn + l_percep)
        loss.backward()
        optim.step()

        time_final = time.time() 

        elapsed_time = (time_final - time_4)
        # orth = gfgr**2 / (gfgf * (gr @ gr))
        return loss.item(), np.nan, np.nan,  np.nan, np.nan, gz, identifier(gz), elapsed_time

    return process_batch





def train(model, folder, num_steps, batch_size, latent_dim=512, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, all_classes=[0, 1], forget_class=1,\
          img_ext='jpg', classifier_path="../../data/CelebA/cnn/cnn_10.pth",  data_path="../../data/CelebA/dataset", max_data=None, **viz_kwargs):
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=0.,\
           all_classes=all_classes, forget_class=forget_class, img_ext=img_ext, classifier_path=classifier_path, train_mode='ascent', data_path=data_path, max_data=max_data)
    old_net = copy.deepcopy(net).to(device)
    # Freeze all parameters in the copied model
    for param in old_net.parameters():
        param.requires_grad = False 
    z_e = cl.find_target_latent_direction(net, identifier, latent_dim)
    process_batch = get_processor(net, old_net, z_e, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight), optim, all_classes, forget_class)    
    log_results = vo.get_logger(identifier, csv_file, log_interval)
    save = vt.get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = vt.get_collector(sample_dir, collect_interval, grid_size, img_ext)
    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for (img_retain, _), (img_forget, _) in zip(dataloader['retain'], dataloader['forget']):
            global_step += 1
            # img_retain = img_retain.to(device)
            # img_forget = img_forget.to(device)  
            # -- Process a single batch
            loss, rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch()
            # loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss 
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz_kwargs.update({"folder": folder})
    viz.summarize_training(**viz_kwargs) 