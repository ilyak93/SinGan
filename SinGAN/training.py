import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize



def train(opt, Gs, Zs, reals, NoiseAmp):

    real_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real = imresize(real_, opt.scale1, opt)
    reals = functions.creat_reals_pyramid(real, reals, opt)
    nfc_prev = 0

    #path = 'TrainedModels/animation_input/scale_factor=0.750000,alpha=10'
    #Gs = torch.load('%s/Gs.pth' % (path))
    #Zs = torch.load('%s/Zs.pth' % (path))
    #NoiseAmp = torch.load('%s/NoiseAmp.pth' % (path))
    #in_s = torch.load('TrainedModels/animation_input/scale_factor=0.750000,alpha=10/in_s.pt')

    while scale_num < opt.stop_scale + 1:
        opt.attn = False

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        functions.im_save('real_scale', opt.outf, reals[scale_num], vmin=0, vmax=1)
        #plt.imsave('%s/real_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        torch.cuda.empty_cache()  # added to try to save memory
        if scale_num >= 0:
            opt.attn = True
        opt.cur_real_shape = reals[len(Gs)].shape
        D_curr, G_curr = init_models(opt)
        if (nfc_prev == opt.nfc):
            G_prev = torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1))
            G_curr_params = G_curr.state_dict()
            filtered_G_prev = {k: v for k, v in G_prev.items() if k in G_curr_params and v.shape == G_curr_params[k].shape}
            G_curr_params.update(filtered_G_prev)
            G_curr.load_state_dict(G_curr_params)
            D_prev = torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1))
            D_curr_params = D_curr.state_dict()
            filtered_D_prev = {k: v for k, v in D_prev.items() if k in D_curr_params and v.shape == D_curr_params[k].shape}
            D_curr_params.update(filtered_D_prev)
            D_curr.load_state_dict(D_curr_params)

        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def train_single_scale(netD, netG, reals, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    real = reals[len(Gs)]
    opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2] + (opt.ker_size - 1) * (opt.num_layer)
        opt.nzy = real.shape[3] + (opt.ker_size - 1) * (opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))
    #real = m_image(real)

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device, dtype=torch.float32)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    errG2norecplot = []
    errG2recplot = []
    D_real2plot = []
    D_fake2plot = []
    D_penality = []
    z_opt2plot = []
    
    real_batch_sz = 1
    
    for epoch in range(opt.niter):
        oom = False
        try:
            if (Gs == []) & (opt.mode != 'SR_train'):
                z_opt = functions.generate_noise([1, opt.nzx, opt.nzy], real.shape[0], device=opt.device)
                z_opt = m_noise(z_opt.expand(real.shape[0], 3, opt.nzx, opt.nzy))
                noise_ = functions.generate_noise([1, opt.nzx, opt.nzy], real.shape[0], device=opt.device)
                noise_ = m_noise(noise_.expand(real.shape[0], 3, opt.nzx, opt.nzy))
            else:
                noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], real.shape[0], device=opt.device)
                noise_ = m_noise(noise_)
            
            

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                # train with real
                netD.zero_grad()

                if opt.mode == 'train_gif_rnn':
                    d_state = netD.init_hidden(real_batch_sz)
                    output, _, _ = netD(real, d_state)
                    output = output.to(opt.device)
                else :
                    output = netD(real).to(opt.device)

                # D_real_map = output.detach()
                errD_real = -output.mean()  # -a
                errD_real.backward(retain_graph=True)
                D_x = -errD_real.item()

                # train with fake
                if (j == 0) & (epoch == 0):
                    if (Gs == []) & (opt.mode != 'SR_train'):
                        prev = torch.full([real.shape[0], opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.float32)
                        in_s = prev
                        prev = m_image(prev)
                        z_prev = torch.full([real.shape[0], opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device, dtype=torch.float32)
                        z_prev = m_noise(z_prev)
                        opt.noise_amp = 1
                    elif opt.mode == 'SR_train':
                        z_prev = in_s
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init * RMSE
                        z_prev = m_image(z_prev)
                        prev = z_prev
                    else:
                        prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                        prev = m_image(prev)
                        z_prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
                        criterion = nn.MSELoss()
                        #plt.imshow(real.permute([0, 2, 3, 1])[0])
                        #plt.show()
                        #real_rmse = real[:, :, 5:-5, 5:-5]
                        #plt.imshow(real_rmse.permute([0,2,3,1])[0])
                        #plt.show()
                        #plt.imshow(z_prev.permute([0, 2, 3, 1])[0])
                        #plt.show()
                        #RMSE = torch.sqrt(criterion(real_rmse, z_prev))
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init * RMSE
                        z_prev = m_image(z_prev)
                else:
                    prev = draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
                    prev = m_image(prev)

                if opt.mode == 'paint_train':
                    prev = functions.quant2centers(prev, centers)
                    plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

                if (Gs == []) & (opt.mode != 'SR_train'):
                    noise = noise_
                else:
                    noise = opt.noise_amp * noise_ + prev

                if opt.mode == 'train_gif_rnn':
                    g_states = netG.init_hidden(real_batch_sz)
                    d_state = netD.init_hidden(real_batch_sz)
                    fake, _ = netG(noise.detach(), prev, g_states)
                    output, _, _ = netD(fake.detach(), d_state)
                else:
                    fake = netG(noise.detach(), prev)
                    output = netD(fake.detach())
                errD_fake = output.mean()
                errD_fake.backward(retain_graph=True)
                D_G_z = output.mean().item()

                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device, opt)
                gradient_penalty.backward()

                D_penality.append(gradient_penalty)
                errD = errD_real + errD_fake + gradient_penalty
                optimizerD.step()

            errD2plot.append(errD.detach())

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(opt.Gsteps):
                netG.zero_grad()
                if opt.mode == 'train_gif_rnn':
                    d_state = netD.init_hidden(real_batch_sz)

                    output, _, _ = netD(fake, d_state)
                else :
                    output = netD(fake)
                # D_fake_map = output.detach()
                errG = -output.mean()
                errG.backward(retain_graph=True)
                if alpha != 0:
                    loss = nn.MSELoss()
                    if opt.mode == 'paint_train':
                        z_prev = functions.quant2centers(z_prev, centers)
                        plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                    Z_opt = opt.noise_amp * z_opt + z_prev
                    if opt.mode == 'train_gif_rnn':
                        g_states = netG.init_hidden(real_batch_sz)
                        fake, _ = netG(Z_opt.detach(), z_prev, g_states)
                    else:
                        fake = netG(Z_opt.detach(), z_prev)
                    rec_loss = alpha * loss(fake, real)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = z_opt
                    rec_loss = 0
                optimizerG.step()
                if opt.mode == 'train_gif_rnn':
                    g_states = netG.init_hidden(real_batch_sz)
                    fake, _ = netG(noise.detach(), prev, g_states)
                else:
                    fake = netG(noise.detach(), prev)

            errG2norecplot.append(errG.detach())
            errG2recplot.append(rec_loss)
            errG2plot.append(errG.detach() + rec_loss)
            D_real2plot.append(D_x)
            D_fake2plot.append(D_G_z)
            z_opt2plot.append(rec_loss)

            if epoch % 25 == 0 or epoch == (opt.niter - 1):
                print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            
            if epoch % 500 == 0 or epoch == (opt.niter - 1):
                functions.im_save('fake_sample', opt.outf, fake.detach(), vmin=0, vmax=1)
                #plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                if opt.mode == 'train_gif_rnn':
                    g_states = netG.init_hidden(real_batch_sz)
                    G_opt, _ = netG(Z_opt.detach(), z_prev, g_states)
                else:
                    G_opt = netG(Z_opt.detach(), z_prev)
                G_opt = G_opt.detach()
                functions.im_save('G(z_opt)', opt.outf, G_opt, vmin=0, vmax=1)
                #plt.imsave('%s/G(z_opt).png' % (opt.outf),
                #           functions.convert_image_np(netG(Z_opt.detach(), z_prev, g_states).detach()), vmin=0, vmax=1)
                # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
                # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
                # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
                # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
                # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

                torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

                print('Generator loss:')
                plt.plot(list(range(0, len(errG2plot))), errG2plot)
                plt.show()
                print('Discriminator real loss:')
                plt.plot(list(range(0, len(D_real2plot))), D_real2plot)
                plt.show()
                print('Discriminator fake loss:')
                plt.plot(list(range(0, len(D_fake2plot))), D_fake2plot)
                plt.show()
                print('Reconstruction loss:')
                plt.plot(list(range(0, len(errG2recplot))), errG2recplot)
                plt.show()

            schedulerD.step()
            schedulerG.step()
        except:
            oom = True
            
        if oom == True:
            alloc = torch.cuda.memory_allocated()
            print(alloc)
            functions.im_save('fake_sample', opt.outf, fake.detach(), vmin=0, vmax=1)
            #plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            if opt.mode == 'train_gif_rnn':
                g_states = netG.init_hidden(real_batch_sz)
                G_opt, _ = netG(Z_opt.detach(), z_prev, g_states)
            else:
                G_opt = netG(Z_opt.detach(), z_prev)
            G_opt = G_opt.detach()
            functions.im_save('G(z_opt)', opt.outf, G_opt, vmin=0, vmax=1)
            #plt.imsave('%s/G(z_opt).png' % (opt.outf),
            #           functions.convert_image_np(netG(Z_opt.detach(), z_prev, g_states).detach()), vmin=0, vmax=1)
            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

            print('Generator loss:')
            plt.plot(list(range(0, len(errG2plot))), errG2plot)
            plt.show()
            print('Discriminator real loss:')
            plt.plot(list(range(0, len(D_real2plot))), D_real2plot)
            plt.show()
            print('Discriminator fake loss:')
            plt.plot(list(range(0, len(D_fake2plot))), D_fake2plot)
            plt.show()
            print('Reconstruction loss:')
            plt.plot(list(range(0, len(errG2recplot))), errG2recplot)
            plt.show()
            break
        

    functions.save_networks(netG, netD, z_opt, opt)
    return z_opt, in_s, netG


def draw_concat(Gs, Zs, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
            #pad_noise = int(((opt.ker_size - 1) * 2) / 2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G.use_cuda = False
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise],
                                                 real_curr.shape[0], device = opt.device)
                    z = z.expand(real_curr.shape[0], 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise(
                        [opt.nc_z, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], real_curr.shape[0],
                        device=opt.device)
                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * z + G_z
                if opt.mode == 'train_gif_rnn':
                    g_states = G.init_hidden(1)
                    G_z, _ = G(z_in.detach(), G_z, g_states)
                else:
                    G_z = G(z_in.detach(), G_z)
                G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp * Z_opt + G_z
                if opt.mode == 'train_gif_rnn':
                    g_states = G.init_hidden(1)
                    G_z, _ = G(z_in.detach(), G_z, g_states)
                else:
                    G_z = G(z_in.detach(), G_z)
                    G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                    G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                    # if count != (len(Gs)-1):
                    #    G_z = m_image(G_z)
                    count += 1
    return G_z


def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr, G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num + 1], Gs[:scale_num],
                                                      Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

            G_curr = functions.reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def init_models(opt):
    # generator initialization:
    netG = models.ConvLSTMGenerator6(opt).to(opt.device)
    total_params = sum(p.numel() for p in netG.parameters())
    train_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print(f'number of parameters of generator: total={total_params} train={train_params}')
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    total_params = sum(p.numel() for p in netD.parameters())
    train_params = sum(p.numel() for p in netD.parameters() if p.requires_grad)
    print(f'number of parameters of generator: total={total_params} train={train_params}')
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
