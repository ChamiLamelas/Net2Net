import tracing
import copy
import torch.nn as nn


# idea of just transferring the corresponding teacher weights to a student is
# mentioned in this paper: https://arxiv.org/abs/2009.09152
def deeper_weight_transfer(teacher, student):
    teacher_layers = tracing.get_all_learnable_layers(teacher)
    student_layers = tracing.get_all_learnable_layers(student)
    for hierarchy, parent in student_layers.items():
        teacher_parent = teacher_layers[hierarchy]
        teacher_layer = getattr(teacher_parent, hierarchy[-1])
        setattr(parent, hierarchy[-1], copy.deepcopy(teacher_layer))
