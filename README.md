# ASPCNet

We have relased the code of Adaptive Spatial Pattern Capsule Network (ASPCNet) algorithm, And the paper has been submitted to [*NEUROCOMPUTING*](https://www.sciencedirect.com/science/article/pii/S092523122200220X). 

Please cite us if our project is helpful to you

J. Wang, X. Tan, J. Lai, and J. Li. ASPCNet: Deep adaptive spatial pattern capsule network for hyperspectral image classification. *Neurocomputing*. 2022(486). 47-60.

If you have any questions, please contact us. e-mail: jinping_wang@foxmail.com



Requirements

     python3
     cuda=9
     cudnn=7
     tensorflow-gpu==1.12
     keras==2.2.4
     numpy 
     scipy


******
Validate_set is an optional parameter. If the value of Validate_set is set, the early_stop operation should be set as follows:

     callback = callbacks.EarlyStopping(monitor='val_acc',
                                        min_delta=0,
                                        patience=args.patience,
                                        verbose=1,
                                        mode='auto',
                                        restore_best_weights=True)

Otherwise, it should be judged based on the training accuracy or training loss, and be set as:

    callback = callbacks.EarlyStopping(monitor='acc',
                                       min_delta=0,
                                       patience=args.patience,
                                       verbose=1,
                                       mode='auto',
                                       restore_best_weights=True)
******
If you want to run this code on the other datasets, please directly replace the ********.mat* file in the *data* folder.

     --> data
        --> ****.mat
     --> img
        --> result
     --> ASP.py
     --> ASPCaps.py
     --> ASPCNet.py
     --> util.py
******

Quick start

     data source files and introduce can been found in http://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
     
     
   
