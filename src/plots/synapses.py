import torch
import matplotlib.pyplot as plt


def cosine_similarity(tensor_a, tensor_b):
    # Validate that the tensors are of same size
    if tensor_a.size() != tensor_b.size():
        raise ValueError("Tensors are not of the same size")

    dot_product = torch.dot(tensor_a.flatten(), tensor_b.flatten())
    norm_a = torch.norm(tensor_a)
    norm_b = torch.norm(tensor_b)

    if norm_a.item() == 0 or norm_b.item() == 0:
        raise ValueError("One of the tensors has zero magnitude, cannot compute cosine similarity")

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def get_behavior(sg,
                 behavior_class: type):
    for key, behavior in sg.behavior.items():
        if behavior.__class__.__name__ == behavior_class.__name__:
            return behavior


def add_current_plot(sg,
                     ax,
                     recorder_behavior_class):
    recorder_behavior = sg.get_behavior(recorder_behavior_class)
    # Plot the current
    ax.plot(recorder_behavior.variables["I"][:, :])

    ax.set_xlabel('t')
    ax.set_ylabel('I(t)')
    ax.legend()
    ax.set_title('Synapse Current')


def add_synapses_params_info(sg,
                             ax,
                             synapse_behavior_class: type,
                             text_x=0.0,
                             text_y=0.5):
    synapse_behavior = sg.get_behavior(synapse_behavior_class)
    params_info = f"Synapses parameters:\n"
    params_info += f"Synapse {sg.tag} params:{synapse_behavior.init_kwargs}\n"
    ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5),
            fontsize=8)


def add_weights_plot(sg,
                     ax,
                     recorder_behavior_class: type,
                     neuron_id,
                     **kwargs):
    recorder_behavior = sg.get_behavior(recorder_behavior_class)
    ax.plot(recorder_behavior.variables["weights"][:, :, neuron_id], **kwargs)
    ax.set_xlabel('t')
    ax.set_ylabel('Weights')
    ax.legend()
    ax.set_title(f'Synapse Weights for neuron {neuron_id}')


def add_cosine_similarity_plot(sg,
                               ax,
                               recorder_behavior_class,
                               neuron_1,
                               neuron_2):
    recorder_behavior = sg.get_behavior(recorder_behavior_class)
    cosine_similarity_recorder = []
    for t in range(sg.network.iteration):
        w_neuron_1 = recorder_behavior.variables["weights"][t, :, neuron_1]
        w_neuron_2 = recorder_behavior.variables["weights"][t, :, neuron_2]
        cosine_similarity_recorder.append(cosine_similarity(w_neuron_1, w_neuron_2))
    ax.plot(cosine_similarity_recorder)
    ax.set_xlabel('time')
    ax.set_ylabel('Cosine similarity')
    ax.legend()
    ax.set_title(f'Cosine similarity between neuron {neuron_1} and neuron {neuron_2}')


def add_learning_params_info(sg, ax, learning_behavior_class: type, text_x=0.0, text_y=0.05):
    learning_behavior = sg.get_behavior(learning_behavior_class)

    params_info = f"{learning_behavior.__class__.__name__} params:\n"
    for key, value in learning_behavior.init_kwargs.items():
        params_info += f"{key}: {value}\n"
    ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.4))
