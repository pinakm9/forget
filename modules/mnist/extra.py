def get_processor_retain(net, trainable_params, identifier, z_random, weights, optim, all_digits, forget_digit):
    digits = all_digits
    def process_batch(img_retain, img_forget):
        kl_weight, orthogonality_weight, uniformity_weight = weights
        img_forget = img_forget.view(img_forget.shape[0], -1).to(net.device)
        img_retain = img_retain.view(img_retain.shape[0], -1).to(net.device)
        # batch_size = img_retain.shape[0]
        
        
        start_time = time.time()
        reconstructed_forget, mu_forget, logvar_forget = net(img_forget)
        reconstructed_retain, mu_retain, logvar_retain = net(img_retain)

        generated_img = net.decoder(z_random)
        logits = identifier(generated_img)
        uniformity = vl.uniformity_loss_surgery(logits, all_digits=all_digits, forget_digit=forget_digit)

    
        rec_forget = vl.reconstruction_loss(reconstructed_forget, img_forget)
        rec_retain = vl.reconstruction_loss(reconstructed_retain, img_retain)
        kl_forget = vl.kl_div(mu_forget, logvar_forget)
        kl_retain = vl.kl_div(mu_retain, logvar_retain)

        loss_forget = rec_forget + kl_weight * kl_forget 
        loss_retain = rec_retain + kl_weight * kl_retain 

        gf = torch.cat([x.view(-1) for x in grad(outputs=loss_forget, inputs=trainable_params, retain_graph=True)])
        gr = torch.cat([x.view(-1) for x in grad(outputs=loss_retain, inputs=trainable_params, retain_graph=True)])
        orth = (gf @ gr)**2 / ((gf @ gf) * (gr @ gr))

        loss = loss_retain + orthogonality_weight * orth + uniformity_weight * uniformity 
        optim.zero_grad()
        loss.backward()
        optim.step()
        elapsed_time = time.time() - start_time

        return (rec_forget + rec_retain).item(), (kl_forget + kl_retain).item(),  uniformity.item(), orth.item(), generated_img, logits, elapsed_time

    return process_batch




def train_to_forget(model, folder, num_steps, batch_size, latent_dim=2, save_steps=None, collect_interval='epoch', log_interval=10,\
          kl_weight=1., uniformity_weight=1e4, orthogonality_weight=1e5, all_digits=list(range(10)), forget_digit=1,\
          img_ext='jpg', classifier_path="../data/MNIST/classifiers/MNISTClassifier.pth"):
    # ---------------------------------------------------
    # Setup
    # ---------------------------------------------------
    net, dataloader, optim, z_random, identifier, sample_dir, checkpoint_dir, epoch_length, epochs,\
    num_steps, save_steps, collect_interval, log_interval, csv_file, device, grid_size \
    = vt.init(model, folder, num_steps, batch_size, latent_dim=latent_dim, save_steps=save_steps, collect_interval=collect_interval,\
           log_interval=log_interval, kl_weight=kl_weight, uniformity_weight=uniformity_weight, orthogonality_weight=orthogonality_weight,\
           all_digits=all_digits, forget_digit=forget_digit, img_ext=img_ext, classifier_path=classifier_path, train_mode='orthogonal')
    process_batch = get_processor_retain(net, ut.get_trainable_params(net), identifier, z_random, (kl_weight, uniformity_weight, orthogonality_weight), optim, all_digits, forget_digit)    
    log_results = get_logger(identifier, csv_file, log_interval)
    save = vt.get_saver(net, save_steps, checkpoint_dir, epoch_length)
    collect_samples = vt.get_collector(sample_dir, collect_interval, grid_size, img_ext)   
    # ---------------------------------------------------
    # Main training loop
    # ---------------------------------------------------
    global_step = 0
    for _ in tqdm(range(1, epochs + 1), desc="Epochs"):
        for (img_retain, _), (img_forget, _) in zip(dataloader['retain'], dataloader['forget']):
            global_step += 1
            # -- Process a single batch
            rec_loss, kl_loss, unif_loss, orth_loss, generated_img, logits, elapsed_time = process_batch(img_retain, img_forget)
            loss = rec_loss + kl_weight * kl_loss + uniformity_weight * unif_loss + orthogonality_weight * orth_loss
            real_img, _ = next(iter(dataloader['original']))
            real_img = real_img.view(real_img.shape[0], -1).to(device)
            log_results(step=global_step, losses=[rec_loss, kl_loss, unif_loss, orth_loss, loss], elapsed_time=elapsed_time, real_img=real_img, generated_img=generated_img, logits=logits)
            save(step=global_step)
            collect_samples(generated_img, step=global_step)
    viz.summarize_training(folder)
