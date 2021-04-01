# fMRINeuralODE

The code is split into two different files one for training the Neural ODE network and the other to test it. The reason was in order to checkpoint and save the trained network
which can take a very long time to run. The variables then can be loaded and tested very quickly. The code loads all the relevant data which was too big to load into github, 
but included 447 hcp release of resting and task data processed such that it was reduced to 66 ROIs based on the DK atlas.  
