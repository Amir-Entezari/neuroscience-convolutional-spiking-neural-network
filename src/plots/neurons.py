import matplotlib.pyplot as plt
import numpy as np


def get_behavior(ng,
                 behavior_class: type):
    for key, behavior in ng.behavior.items():
        if behavior.__class__.__name__ == behavior_class.__name__:
            return behavior


def add_current_params_info(ng,
                            ax,
                            current_behavior_class: type,
                            text_x=0.0,
                            text_y=0.05):
    current_behavior = get_behavior(ng, current_behavior_class)
    params_info = f"""{current_behavior.__class__.__name__} params: {current_behavior.init_kwargs}"""
    ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))


def add_neuron_model_params_info(ng, ax, model_behavior_class: type, text_x=0.0, text_y=0.05):
    neuron_model_behavior = get_behavior(ng, model_behavior_class)
    params_info = f"{neuron_model_behavior.__class__.__name__} params:\n"
    for key, value in neuron_model_behavior.init_kwargs.items():
        params_info += f"{key}: {value}\n"
    ax.text(text_x, text_y, params_info, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.4))


def add_current_plot(ng,
                     ax,
                     recorder_behavior_class: type):
    recorder_behavior = get_behavior(ng, recorder_behavior_class)
    # Plot the current
    ax.plot(recorder_behavior.variables["I"][:, :])
    ax.plot([], [], label="Other colors: Received I for each neuron")
    # ax.plot(recorder_behavior.variables["inp_I"][:, :1],
    #         label="input current",
    #         color='black')

    ax.set_xlabel('t')
    ax.set_ylabel('I(t)')
    ax.legend()
    ax.set_title(f'Current: {ng.tag}')


def add_raster_plot(ng,
                    ax,
                    event_recorder_class: type,
                    title=None,
                    s=5,
                    **kwargs):
    if not title:
        title = f'Raster Plot: {ng.tag}'
    event_recorder = get_behavior(ng, event_recorder_class)
    # Plot the raster plot
    spike_events = event_recorder.variables["spikes"]
    spike_times = spike_events[:, 0]
    neuron_ids = spike_events[:, 1]
    ax.scatter(spike_times, neuron_ids, s=s, label=f"{ng.tag}", **kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')
    ax.legend(loc='upper right')
    ax.set_title(title)


def add_activity_plot(ng,
                      ax,
                      recorder_behavior_class: type):
    recorder_behavior = get_behavior(ng, recorder_behavior_class)
    # Plot the activity
    activities = recorder_behavior.variables["activity"]
    x_range = np.arange(1, len(activities) + 1)
    ax.plot(x_range, activities, label="activity")
    ax.set_xlabel('Time')
    ax.set_ylabel('activity')
    ax.legend()
    ax.set_title(f'Activity {ng.tag}')


def add_membrane_potential_plot(ng,
                                ax,
                                recorder_behavior_class: type,
                                neuron_model_class: type,
                                ):
    recorder_behavior = get_behavior(ng, recorder_behavior_class)
    neurom_model_behavior = get_behavior(ng, neuron_model_class)
    ax.plot(recorder_behavior.variables["v"][:, :])

    # ax.axhline(y=ng.behavior[model_idx].init_kwargs['threshold'], color='red', linestyle='--',
    #            label=f'{ng.tag} Threshold')
    ax.axhline(y=neurom_model_behavior.init_kwargs['v_reset'], color='black', linestyle='--',
               label=f'{ng.tag} v_reset')

    ax.set_xlabel('Time')
    ax.set_ylabel('v(t)')
    ax.set_title(f'Membrane Potential {ng.tag}')
    ax.legend()


def add_membrane_potential_distribution(ng,
                                        ax,
                                        recorder_behavior_class: type):
    recorder_behavior = get_behavior(ng, recorder_behavior_class)
    rotated_matrix = np.transpose(recorder_behavior.variables["v"])
    # Plotting the rotated heatmap
    ax.imshow(rotated_matrix, aspect='auto', cmap='jet', origin='lower')
    # ax.colorbar(label='Membrane Potential')
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Time (Iterations)')
    ax.set_title('Membrane Potentials Heatmap Distribution Over Time')


def plot_w(ng, title: str,
           recorder_behavior_class: type,
           save: bool = None,
           filename: str = None):
    recorder_behavior = get_behavior(ng, recorder_behavior_class)
    # Generate colors for each neuron
    plt.plot(recorder_behavior.variables["w"][:, :1], label=f'adaptation')

    plt.xlabel('Time')
    plt.ylabel('w')
    plt.legend(loc='upper left', fontsize='small')

    plt.title(title)
    if save:
        plt.savefig(filename or title + '.pdf')
    plt.show()