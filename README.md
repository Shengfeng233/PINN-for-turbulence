# PINN-for-turbulence
A pytorch implementation of several approaches using PINN to slove turbulent flow

So far, there are three promising approaches to solve turbulent flow using physics informed neural network(see reference1-3), including using NS equation, RANS euqation with turbulent eddy viscosity, RANS equation with reynolds stress. This repository implemented these methods separately in different branches.

For simplicity, this repository only considers the 2d incompressible flow. We have tried 3d flow, but it seems to be extremely difficult and tricky to train a 3d flow. Through experiments, we discovered the difficulty mainly comes from the continuity equation, which shows a completely different feature compared to momentum equations during backpropagation. For a 2d incompressible flow, the trick lies in the flow function. By introducing the flow function, the continuity equation will be automatically satisfied, so we do not need to calculate the continuity equation in the loss function, which greatly reduces the training difficulty.

The test case for this repository is the wake of a 2d flow past a circular cylinder calculated with kw-sst(Re=3900).

# Note
1. All the data in the wake region is used in this repository. However, if you try different sparsity, you will find it is still trainable.

2. The learning rate schedule adopted in this repository is Cosine Annealing Warm Restart With Decay(see reference4).

3. Both the data and the equations are non-dimensional. 

4. Further approaches for solving turbulent flow are under study.

5. The annotates are in Chinese.

6. The code is still being optimized, it would be appreciated to contact me (xu_shengfeng1220@163.com) if you find any unreasonably written code.

# Reference
1. Raissi M, Perdikaris P, Karniadakis G E. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations[J]. Journal of Computational physics, 2019, 378: 686-707.
2. Xu H, Zhang W, Wang Y. Explore missing flow dynamics by physics-informed deep learning: The parameterized governing systems[J]. Physics of Fluids, 2021, 33(9): 095116.
3. Eivazi H, Tahani M, Schlatter P, et al. Physics-informed neural networks for solving Reynolds-averaged Navier–Stokes equations[J]. Physics of Fluids, 2022, 34(7): 075117.
4. https://github.com/saadnaeem-dev/pytorch-linear-warmup-cosine-annealing-warm-restarts-weight-decay
