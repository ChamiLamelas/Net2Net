import tracing
import copy


def deeper_weight_transfer(teacher, student):
    teacher_layers = tracing.get_all_important_layer_hierarchies(teacher)
    student_layers = tracing.get_all_important_layer_hierarchies(student)
    for hierarchy, parent in student_layers.items():
        teacher_parent = teacher_layers[hierarchy]
        teacher_layer = getattr(teacher_parent, hierarchy[-1])
        setattr(parent, hierarchy[-1], copy.deepcopy(teacher_layer))
