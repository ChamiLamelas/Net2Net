import tracing
import copy
import torch.nn as nn

# idea of just transferring the corresponding teacher weights to a student is
# mentioned in this paper: https://arxiv.org/abs/2009.09152
def deeper_weight_transfer(teacher, student):
    teacher_table = tracing.LayerTable(teacher)
    student_table = tracing.LayerTable(student)
    for ls in student_table:
        curr = student_table.get(ls["hierarchy"])
        if isinstance(curr, nn.Conv2d) or isinstance(curr, nn.Linear):
            teacher_layer = teacher_table.get(ls["hierarchy"])
            if isinstance(teacher_layer, nn.Sequential):
                teacher_layer = teacher_layer[0]
            student_table.set(ls["hierarchy"], copy.deepcopy(teacher_layer))
