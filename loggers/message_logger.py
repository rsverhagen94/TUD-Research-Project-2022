from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld
import copy
import json
import numpy as np
import re


class message_logger(GridWorldLogger):
    """ Logs messages send and received by (all) agents """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimiter, log_strategy=1)

    def log(self, grid_world: GridWorld, agent_data: dict):

        log_data = {
            'total_number_messages_human': 0,
            'total_number_messages_agent': 0,
            'total_allocations': 0,
            'total_allocations_human': 0,
            'total_interventions': 0,
            'sensitivity': '',
            'mean_intervention_sensitivity': '',
            'disagreement': 0
        }

        gwmm = grid_world.message_manager
        t = grid_world.current_nr_ticks - 1
        tot_messages_human = 0
        tot_messages_agent = 0
        tot_allocations = 0
        tot_allocations_human = 0
        tot_interventions = 0
        sensitivity = ''
        processed_messages = []
        interventions_sensitivity = []

        for i in range(0, t):
            if i in gwmm.preprocessed_messages.keys():
                for mssg in gwmm.preprocessed_messages[i]:
                    if (i, mssg.content) not in processed_messages and 'Time left: ' not in mssg.content and 'Fire duration: ' not in mssg.content and 'Smoke spreads: ' not in mssg.content \
                        and 'Temperature: ' not in mssg.content and 'Location: ' not in mssg.content and 'Distance: ' not in mssg.content and 'Victims rescued: ' not in mssg.content:
                        processed_messages.append((i, mssg.content))
                        match = re.search(r'<b>(.*?)</b>', mssg.content)
                        if match:
                            sensitivity = match.group(1)
                            tot_allocations += 1
                        else:
                            sensitivity = ''
                        if 'human' in mssg.from_id:
                            tot_messages_human += 1
                        if 'Titus' in mssg.from_id or 'Brutus' in mssg.from_id:
                            tot_messages_agent += 1
                        if 'above my allocation threshold' in mssg.content:
                            tot_allocations_human += 1
                        if 'Allocating this decision with a predicted moral sensitivity of' in mssg.content:
                            tot_interventions += 1
                            interventions_sensitivity.append(float(mssg.content.split()[-6]))

        log_data['total_number_messages_human'] = tot_messages_human
        log_data['total_number_messages_agent'] = tot_messages_agent
        log_data['total_allocations'] = tot_allocations
        log_data['total_allocations_human'] = tot_allocations_human
        log_data['total_interventions'] = tot_interventions
        log_data['sensitivity'] = sensitivity
        log_data['mean_intervention_sensitivity'] = np.mean(interventions_sensitivity)


        return log_data