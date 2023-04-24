# SinGAN

![image](https://user-images.githubusercontent.com/50303550/234036154-fdceeb57-43b8-4670-83b0-118e63364117.png)

As a final project in the course “Digital Image Processing” (Computer Science faculty, Technion - Israel Institute for Technology) we were given list of papers, one of which we had to choose to try to use it for a different task or improve performance. The idea of this project isn’t to came with an article, but to investigate in details the paper and try your own ideas, analyze and report the outcomes. 

I had chosen the paper “SinGAN: Learning a Generative Model from a Single Natural Image” (Tamar Rott Shaham, Tali Dekel, Tomer Michaeli). That was a natural choice for me as it was a great opportunity to refresh and expand my theoretical knowledge and practical skills in the domain of deep learning, which I’m a great enthusiast of.  

(*) Links for the original work:

[Project](https://tamarott.github.io/SinGAN.htm) | [Arxiv](https://arxiv.org/pdf/1905.01164.pdf) | [CVF](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf) | [Supplementary materials](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Shaham_SinGAN_Learning_a_ICCV_2019_supplemental.pdf) | [Talk (ICCV`19)](https://youtu.be/mdAcPe74tZI?t=3191) 

One idea I’ve chosen to implement is adding an attention mechanism to the GAN on each of the pyramidal levels for an attempt to include the ability for the network to perceive features across the image (as attentions mechanism do) and measuring the performance with the same measurements the original SinGAN was measured, SIFID (a variant of Frechet Inception Distance) and RMSE.

In some cases better SIFID was achieved for a particular architecture with attention on some examples, although visually it wasn’t noticeable and it was too memory-expensive to test it on the whole test dataset SinGAN was tested on.

Another idea I’ve also examined in this project, is to create multiple animations from one  in the same manner SinGAN creates multiple similar images from one. The difference from the paper approach is to add sequence-memory to the method instead of only do a walk in learned z-space of one image. 
For this, I’ve used RNN/LSTM-like architectures and expanded the pyramidal SinGAN architecture.  

In both implementation a noticeable bottleneck for the runs was a lack of GPU memory, which I handled by investigating less costly attentions mechanism for the first idea, and pruning training on the finest scales of the pyramidal training for the second (among other technical tweaks). 
The report at first briefly introduce the work and the motivation for it, then overlooks and explains to some level the used mechanisms and networks, later reproduces closely enough the results of the paper on one image of originally 50. 

Here are some demonstrations of the main results:

Image Generation (with similiar results to the reproduced results and a little bit better):

![image](https://user-images.githubusercontent.com/50303550/234034959-9700a50a-9cfa-4628-845f-dc1625b56235.png)

![image](https://user-images.githubusercontent.com/50303550/234035068-2f243c76-3e19-46cd-ba04-1775f9a3870f.png)

![image](https://user-images.githubusercontent.com/50303550/234035092-449d003e-2967-47e1-a213-8bae938ed026.png)

Super-Resolution (ours gets better NIQE results)::

![image](https://user-images.githubusercontent.com/50303550/234035203-a05888b7-ecff-41bc-ac89-e9a3e62e0b9c.png)

Finally, here are presented some gifs we created in the proposed approach (with the memory restrictions we had):

The original (regular and reverted):

![](https://github.com/ilyak93/SinGan/blob/main/report/real/waterfall-regular.gif)

![](https://github.com/ilyak93/SinGan/blob/main/report/real/waterfall-rev.gif)

Generated:

![](https://github.com/ilyak93/SinGan/blob/main/report/fake/waterfall-regular-fake.gif)


![](https://github.com/ilyak93/SinGan/blob/main/report/fake/waterfall-rev-fake.gif)







