# Notes 

**DEPRECATED WITH CLEANUP** 

## Description 

This document collects the observations I have made based on the models presented in the repository
and their performance logged in [logs](examples/logs/) and shown in [plots](examples/plots/). 

## Legend 

The following is a legend of the plot and log labels taken from the original code. These are how I interpret the meaning of these models as named in the original code. 

- `teacher_training`: this is the original (smaller) model that is trained originally. 
- `wider_student`: this is a wider version of `teacher_training` that is then *further trained*.
- `wider_deeper_student`: this is a wider and deeper version of `teacher_training` that is then *further trained*.
- `wider_teacher`: this is a wider version of `teacher_training` that is *trained from scratch*.
- `wider_deeper_teacher`: this is a wider and deeper version of `teacher_training` that is *trained from scratch*.

## Observations on MNIST 

See [here](examples/plots/mnist).

**NEEDSWORK** 

<!-- - `small` (`teacher_training`) performs the worst, this makes sense as this is the smallest model. 
- `wider_teacher` and `wider_student` perform the same in the long term. This is not surprising, given that we give the models extensive time to learn. However, it is important to note that, as expected, `wider_student` reaches high accuracy much faster. 
- `wider_deeper_teacher` and `wider_deeper_student` exhibit a similar relationship.
- It doesn't seem that `wider_deeper_student` starts at instantaneously as good as `teacher_training`, which according to the paper, it should.  -->

## Observations on CIFAR10

See [here](examples/plots/cifar10).

**NEEDSWORK** 

<!-- - `teacher_training` performs the worst as in the MNIST experiment. 
- Similar to above, `wider_teacher` and `wider_student` again perform similarly in the long term. However, `wider_student` is faster to reach higher accuracy.
- Interestingly enough, `wider_deeper_teacher` is noticeably better in the long term. However, `wider_deeper_student` does again reach higher accuracy faster. 
- This experiment is a bit jumpy, so it may be better to run for a longer period. 
- It doesn't seem that `wider_deeper_student` and `wider_student` start as instanteously as good as `teacher_training`, which according to the paper, they should. -->