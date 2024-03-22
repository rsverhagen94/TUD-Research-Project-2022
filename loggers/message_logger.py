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
        self._threshold = None

    def log(self, grid_world: GridWorld, agent_data: dict):

        log_data = {
            'total_number_messages_human': 0,
            'total_number_messages_robot': 0,
            'total_allocations_human': 0,
            'total_allocations_robot': 0,
            'total_allocations': 0,
            'total_interventions': 0,
            'disagreement_rate': 0,
            'correct_behavior_rate': 0,
            'incorrect_behavior_rate': 0,
            'incorrect_intervention_rate': 0,
            'correct_intervention_rate': 0,
            'CRR_ND_self': 0,
            'FR_ND_self': 0,
            'FRR_MD_self': 0,
            'FR_ND_robot': 0,
            'CRR_ND_robot': 0,
            'CR_MD_self': 0,
            'CRR_MD_robot': 0,
            'FR_MD_robot': 0,
        }

        gwmm = grid_world.message_manager
        t = grid_world.current_nr_ticks - 1
        tot_messages_human = 0
        tot_messages_robot = 0
        tot_allocations_human = 0
        tot_allocations_robot = 0
        CRR_ND_self = 0
        FR_ND_self = 0
        FRR_MD_self = 0
        CR_MD_self = 0
        CRR_MD_robot = 0
        FR_MD_robot = 0
        CRR_ND_robot = 0
        FR_ND_robot = 0
        sensitivity = ''
        processed_messages = []
        interventions_sensitivity = []

        for i in range(0, t):
            if i in gwmm.preprocessed_messages.keys():
                for mssg in gwmm.preprocessed_messages[i]:
                    if 'Counterbalancing' in mssg.content:
                        counterbalance_condition = mssg.content.split()[2]
                        robot = mssg.content.split()[4]
                        self._threshold = mssg.content.split()[6]
                   
                    if (i, mssg.content) not in processed_messages and 'Time left: ' not in mssg.content and 'Fire duration: ' not in mssg.content and 'Smoke spreads: ' not in mssg.content \
                        and 'Temperature: ' not in mssg.content and 'Location: ' not in mssg.content and 'Distance: ' not in mssg.content and 'Victims rescued: ' not in mssg.content and 'Counterbalancing' not in mssg.content:
                        processed_messages.append((i, mssg.content))
                        #match = re.search(r'<b>(.*?)</b>', mssg.content)

                        #if match:
                        #    sensitivity = match.group(1)
                        #    tot_allocations += 1
                        #else:
                        #    sensitivity = ''

                        if 'No intervention' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[6]) < 4.2:
                            CRR_ND_self += 1

                        if 'No intervention' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[6]) >= 4.2 and float(mssg.content.split()[6]) <= 5:
                            FRR_MD_self += 1

                        if 'No intervention' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[6]) > 5:
                            CRR_MD_robot += 1

                        if 'No intervention' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[6]) < 3.5:
                            CRR_ND_self += 1

                        if 'No intervention' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[6]) >= 3.5 and float(mssg.content.split()[6]) < 4.2:
                            CRR_ND_robot += 1

                        if 'No intervention' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[6]) >= 4.2:
                            CRR_MD_robot += 1

                        if 'Reallocating' in mssg.content and 'to you' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[9]) < 4.2:
                            FR_ND_self += 1

                        if 'Reallocating' in mssg.content and 'to you' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[9]) >= 4.2 and float(mssg.content.split()[9]) <= 5:
                            CR_MD_self += 1

                        if 'Reallocating' in mssg.content and 'to you' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[9]) < 3.5:
                            FR_ND_self += 1

                        if 'Reallocating' in mssg.content and 'to me' in mssg.content and self._threshold == '5.0' and float(mssg.content.split()[9]) > 5:
                            FR_MD_robot += 1

                        if 'Reallocating' in mssg.content and 'to me' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[9]) >= 3.5 and float(mssg.content.split()[9]) < 4.2:
                            FR_ND_robot += 1

                        if 'Reallocating' in mssg.content and 'to me' in mssg.content and self._threshold == '3.5' and float(mssg.content.split()[9]) >= 4.2:
                            FR_MD_robot += 1
                            
                        if 'human' in mssg.from_id:
                            tot_messages_human += 1

                        if 'Titus' in mssg.from_id and 'No intervention' not in mssg.content or 'Brutus' in mssg.from_id and 'No intervention' not in mssg.content:
                            tot_messages_robot += 1

                        if 'above my allocation self._threshold' in mssg.content:
                            tot_allocations_human += 1

                        if 'below my allocation self._threshold' in mssg.content:
                            tot_allocations_robot += 1

                        #if 'Reallocating' in mssg.content:
                        #    tot_interventions += 1
                        #    interventions_sensitivity.append(float(mssg.content.split()[9]))
                            
        log_data['total_number_messages_human'] = tot_messages_human
        log_data['total_number_messages_robot'] = tot_messages_robot
        log_data['total_allocations_human'] = tot_allocations_human
        log_data['total_allocations_robot'] = tot_allocations_robot

        if self._threshold == '5.0':
            tot_allocations = CRR_ND_self + FR_ND_self + FRR_MD_self + CR_MD_self + CRR_MD_robot + FR_MD_robot
            tot_interventions = FR_ND_self + CR_MD_self + FR_MD_robot
            correct_behavior = CRR_ND_self + CR_MD_self + CRR_MD_robot
            incorrect_behavior = FR_ND_self + FRR_MD_self + FR_MD_robot
            incorrect_interventions = FR_ND_self  + FR_MD_robot
            correct_interventions = CR_MD_self
            log_data['CRR_ND_self'] = CRR_ND_self
            log_data['FR_ND_self'] = FR_ND_self
            log_data['FRR_MD_self'] = FRR_MD_self
            log_data['CR_MD_self'] = CR_MD_self
            log_data['CRR_MD_robot'] = CRR_MD_robot
            log_data['FR_MD_robot'] = FR_MD_robot
            log_data['CRR_ND_robot'] = ''
            log_data['FR_ND_robot'] = ''
            log_data['total_allocations'] = tot_allocations
            log_data['total_interventions'] = tot_interventions
            if tot_allocations > 0:
                log_data['disagreement_rate'] = tot_interventions / tot_allocations
                log_data['correct_behavior_rate'] = correct_behavior / tot_allocations
                log_data['incorrect_behavior_rate'] = incorrect_behavior / tot_allocations
            if tot_interventions > 0:
                log_data['incorrect_intervention_rate'] = incorrect_interventions / tot_interventions
                log_data['correct_intervention_rate'] = correct_interventions / tot_interventions

        if self._threshold == '3.5':
            tot_allocations = CRR_ND_self + FR_ND_self + CRR_ND_robot + FR_ND_robot + CRR_MD_robot + FR_MD_robot
            tot_interventions = FR_ND_self + FR_ND_robot + FR_MD_robot
            correct_behavior = CRR_ND_self + CRR_ND_robot + CRR_MD_robot
            incorrect_behavior = FR_ND_self + FR_ND_robot + FR_MD_robot
            log_data['CRR_ND_self'] = CRR_ND_self
            log_data['FR_ND_self'] = FR_ND_self
            log_data['CRR_ND_robot'] = CRR_ND_robot
            log_data['FR_ND_robot'] = FR_ND_robot
            log_data['CRR_MD_robot'] = CRR_MD_robot
            log_data['FR_MD_robot'] = FR_MD_robot
            log_data['FRR_MD_self'] = ''
            log_data['CR_MD_self'] = ''
            log_data['correct_intervention_rate'] = ''
            log_data['incorrect_intervention_rate'] = ''
            log_data['total_allocations'] = tot_allocations
            log_data['total_interventions'] = tot_interventions
            if tot_allocations > 0:
                log_data['disagreement_rate'] = tot_interventions / tot_allocations
                log_data['correct_behavior_rate'] = correct_behavior / tot_allocations
                log_data['incorrect_behavior_rate'] = incorrect_behavior / tot_allocations

        #log_data['sensitivity'] = sensitivity
        #log_data['mean_intervention_sensitivity'] = np.mean(interventions_sensitivity)

        return log_data