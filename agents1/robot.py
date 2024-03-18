import sys, random, enum, ast, time, threading, os, math
from datetime import datetime
from flask import jsonify
from rpy2 import robjects
from matrx import grid_world
from brains1.custom_agent_brain import custom_agent_brain
from utils1.util_functions import *
from actions1.custom_actions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.custom_actions import Backup, RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop, Injured, AddObject

class Phase(enum.Enum):
    INTRO=0,
    LOCATE=1,
    FIND_NEXT_GOAL=2,
    PICK_UNSEARCHED_ROOM=3,
    PLAN_PATH_TO_ROOM=4,
    FOLLOW_PATH_TO_ROOM=5,
    REMOVE_OBSTACLE_IF_NEEDED=6,
    ENTER_ROOM=7,
    PLAN_ROOM_SEARCH_PATH=8,
    FOLLOW_ROOM_SEARCH_PATH=9,
    PLAN_PATH_TO_VICTIM=10,
    FOLLOW_PATH_TO_VICTIM=11,
    TAKE_VICTIM=12,
    PLAN_PATH_TO_DROPPOINT=13,
    FOLLOW_PATH_TO_DROPPOINT=14,
    DROP_VICTIM=15,
    TACTIC=18,
    PRIORITY=20,
    RESCUE=21,
    EXTINGUISH_CHECK=22

    
class robot(custom_agent_brain):
    def __init__(self, name, condition, resistance, duration, no_fires, victims, task):
        super().__init__(name, condition, resistance, duration, no_fires, victims, task)
        self._phase=Phase.INTRO
        self._name = name
        self._condition = condition
        self._resistance = resistance
        self._duration = duration
        self._time_left = resistance
        self._no_fires = no_fires
        self._victims = victims
        self._task = task
        self._room_victims = []
        self._searched_rooms = []
        self._searched_rooms_defensive = []
        self._searched_rooms_offensive = []
        self._found_victims = []
        self._rescued_victims = []
        self._lost_victims = []
        self._modulos = []
        self._send_messages = []
        self._fire_locations = []
        self._extinguished_fire_locations = []
        self._room_tiles = []
        self._situations = []
        self._plot_times = []
        self._potential_source_offices = []
        self._situation = None
        self._victim_locations = {}
        self._current_door = None
        self._current_room = None
        self._goal_victim = None
        self._goal_location = None
        self._id = None
        self._fire_source_coords = None
        self._current_location = None
        self._plot_generated = False
        self._smoke = '?'
        self._location = '?'
        self._distance = '?'
        self._tactic = 'offensive'
        self._offensive_deployment_time = 0
        self._defensive_deployment_time = 0
        self._offensive_search_rounds = 0

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)  
        load_R_to_Py()

    def filter_bw4t_observations(self, state):
        self._office_doors = {(2, 3): '1', (9, 3): '2', (16, 3): '3', (23, 3): '4', (2, 7): '5', (9, 7): '6', (16, 7): '7',
                              (2, 17): '8', (9, 17): '9', (16, 17): '10', (2, 21): '11', (9, 21): '12', (16, 21): '13', (23, 21): '14'}
        self._second = state['World']['tick_duration'] * state['World']['nr_ticks']
        if int(self._second) % 6 == 0 and int(self._second) not in self._modulos:
            self._modulos.append(int(self._second))
            self._resistance -= 1
            self._duration += 1
            if self._tactic == 'offensive':
                self._offensive_deployment_time += 1
            if self._tactic == 'defensive':
                self._defensive_deployment_time += 1
        self._send_message('Time left: ' + str(self._resistance) + '.', 'RescueBot')
        self._send_message('Fire duration: ' + str(self._duration) + '.', 'RescueBot')
        return state

    def decide_on_bw4t_action(self, state:State):
        print(self._phase)

        self._current_location = state[self.agent_id]['location']

        if self._name == 'Brutus':
            self._threshold = 5.0
        if self._name == 'Titus':
            self._threshold = 4.0

        for info in state.values():
            if 'class_inheritance' in info and 'AreaTile' in info['class_inheritance'] and info['location'] not in self._room_tiles:
                self._room_tiles.append(info['location'])
            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'source' in info['obj_id']:
                if not self._fire_source_coords:
                    self._send_message('Found fire source!', self._name)
                    self._fire_source_coords = info['location']
                    action_kwargs = add_object([info['location']], "/images/fire2.svg", 3, 1, 'fire source')
                    return AddObject.__name__, action_kwargs
                self._location = 'found'
                self._smoke = info['smoke']
                if self._tactic == 'defensive':
                    self._phase = Phase.EXTINGUISH_CHECK
            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                if info['location'] not in self._fire_locations:
                    self._send_message('Found fire in ' + self._current_room + '.', self._name)
                    self._fire_locations.append(info['location'])
                self._smoke = info['smoke']
                if self._tactic == 'defensive':
                    self._phase = Phase.EXTINGUISH_CHECK
            if 'class_inheritance' in info and 'SmokeObject' in info['class_inheritance'] and 'smog' in info['obj_id']:
                if info['location'] in self._office_doors.keys() and info['location'] not in self._potential_source_offices:
                    self._potential_source_offices.append(info['location'])

        if self.received_messages_content and self.received_messages_content[-1] == 'Fire source located and pinned on the map.':
            self._location = 'found' 

        if self._location == 'found':
            for info in state.values():
                if 'class_inheritance' in info and 'EnvObject' in info['class_inheritance'] and 'fire source' in info['name']:
                    self._fire_source_coords = info['location']

        if self._location == '?':
            self._location_cat = 'unknown'
        if self._location == 'found':
            self._location_cat = 'known'

        if self._time_left - self._resistance not in self._plot_times: 
            self._plot_generated = False

        if self._no_fires == 7:
            if len(self._extinguished_fire_locations) / self._no_fires <= 0.45 and self._duration >= 45:
                self._temperature = '>'
                self._temperature_cat = 'higher'
            if len(self._extinguished_fire_locations) / self._no_fires >= 0.8:
                self._temperature = '<'
                self._temperature_cat = 'lower'
            if len(self._extinguished_fire_locations) / self._no_fires < 0.8 and len(self._extinguished_fire_locations) / self._no_fires > 0.45 or \
                len(self._extinguished_fire_locations) / self._no_fires <= 0.45 and self._duration < 45:
                self._temperature = '<≈'
                self._temperature_cat = 'close'
        if self._no_fires == 5:
            if len(self._extinguished_fire_locations) / self._no_fires <= 0.4 and self._duration >= 45:
                self._temperature = '>'
                self._temperature_cat = 'higher'
            if len(self._extinguished_fire_locations) / self._no_fires >= 0.8:
                self._temperature = '<'
                self._temperature_cat = 'lower'
            if len(self._extinguished_fire_locations) / self._no_fires < 0.8 and len(self._extinguished_fire_locations) / self._no_fires > 0.4 or \
                len(self._extinguished_fire_locations) / self._no_fires <= 0.4 and self._duration < 45:
                self._temperature = '<≈'
                self._temperature_cat = 'close'
        if self._no_fires == 3:
            if len(self._extinguished_fire_locations) / self._no_fires == 0 and self._duration >= 45:
                self._temperature = '>'
                self._temperature_cat = 'higher'
            if len(self._extinguished_fire_locations) / self._no_fires >= 0.65:
                self._temperature = '<'
                self._temperature_cat = 'lower'
            if len(self._extinguished_fire_locations) / self._no_fires < 0.65 and len(self._extinguished_fire_locations) / self._no_fires > 0 or \
                len(self._extinguished_fire_locations) / self._no_fires == 0 and self._duration < 45:
                self._temperature = '<≈'
                self._temperature_cat = 'close'

        self._send_message('Smoke spreads: ' + self._smoke + '.', 'RescueBot')
        self._send_message('Temperature: ' + self._temperature + '.', 'RescueBot')
        self._send_message('Location: ' + self._location + '.', 'RescueBot')
        self._send_message('Distance: ' + self._distance + '.', 'RescueBot')

        while True:
            if Phase.EXTINGUISH_CHECK == self._phase:
                for info in state.values():
                    if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id'] and self._tactic == 'defensive' or \
                        'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'source' in info['obj_id'] and self._tactic == 'defensive':
                        #self._fire_locations.append(info['location'])
                        self._send_message('Extinguishing fire in ' + self._current_room + '.', self._name)
                        if info['location'] not in self._extinguished_fire_locations:
                            self._extinguished_fire_locations.append(info['location'])
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 10}
                    if 'class_inheritance' in info and 'EnvObject' in info['class_inheritance'] and 'fire source' in info['name'] and self._tactic == 'defensive' and calculate_distances(self._current_location, info['location']) <= 3:
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 0}
                self._searched_rooms_defensive.append(self._current_room)
                self._phase = Phase.FIND_NEXT_GOAL

            if Phase.INTRO == self._phase:
                self._send_message('If you are ready to begin our mission, press the "Continue" button.', self._name)
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if self._time_left - self._resistance >= 20 and self._time_left - self._resistance <= 25:
                self._situation = 'switch 1'

            if self._time_left - self._resistance >= 35 and self._time_left - self._resistance <= 40:
                self._situation = 'switch 2'

            if self._time_left - self._resistance >= 50 and self._time_left - self._resistance <= 55:
                self._situation = 'switch 3'

            if self._time_left - self._resistance >= 65 and self._time_left - self._resistance <= 70:
                self._situation = 'switch 4'

            if self._time_left - self._resistance >= 80 and self._time_left - self._resistance <= 85:
                self._situation = 'switch 5'

            if self._current_location not in self._room_tiles and not self._plot_generated and self._situation != None and self._situation not in self._situations:
                self._situations.append(self._situation)
                image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_at_time_" + str(self._resistance) + ".svg"
                self._sensitivity = R_to_Py_plot_tactic(self._total_victims_cat, self._location_cat, self._duration, self._resistance, image_name)
                self._plot_generated = True
                
                if self._condition == 'shap':
                    image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                if self._condition == 'util' and self._tactic == 'defensive':
                    image_name = "<img src='/static/images/util_plots/util-continue-defensive-final.svg'/>"
                if self._condition == 'util' and self._tactic == 'offensive':
                    image_name = "<img src='/static/images/util_plots/util-continue-offensive-final.svg'/>"

                if self._sensitivity > self._threshold:
                    if self._tactic == 'offensive':
                        if self._condition == 'shap':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'util':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'baseline':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold.', self._name)
                        self._deploy_time = self._offensive_deployment_time
                    if self._tactic == 'defensive':
                        if self._condition == 'shap':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'util':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'baseline':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is above my allocation threshold.', self._name)
                        self._deploy_time = self._defensive_deployment_time
                    self._decide = 'human'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._time = int(self._second)
                    self._phase = Phase.TACTIC
                    return Idle.__name__, {'duration_in_ticks': 0}
                
                if self._sensitivity <= self._threshold:
                    if self._tactic == 'offensive':
                        if self._condition == 'shap':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'util':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'baseline':
                            self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold.', self._name)

                    if self._tactic == 'defensive':
                        if self._condition == 'shap':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'util':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                + image_name, self._name)
                        if self._condition == 'baseline':
                            self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                                We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                is below my allocation threshold.', self._name)
                    self._decide = self._name
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._time = int(self._second)
                    self._phase = Phase.TACTIC
                    return Idle.__name__, {'duration_in_ticks': 0}

            if Phase.TACTIC == self._phase:
                if self._decide == 'human' and self._tactic == 'offensive' and int(self._second) >= self._time + 15:
                    self._send_message('If you want to continue with the offensive deployment going on for ' + str(self._deploy_time) + ' minutes now, press the "Continue" button. \
                                        If you want to switch to a defensive deployment, press the "Switch" button.', self._name)
                    self._plot_times.append(self._time_left - self._resistance)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Continuing with the offensive deployment that has been going on for ' + str(self._deploy_time) + ' minutes.', self._name)
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Switch':
                        self._send_message('Switching to a defensive deployment after the offensive deployment of ' + str(self._deploy_time) + ' minutes.', self._name)
                        #self._offensive_deployment_time = 0
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        return None, {}

                if self._decide == 'human' and self._tactic == 'defensive' and int(self._second) >= self._time + 15:
                    self._send_message('If you want to continue with the defensive deployment going on for ' + str(self._deploy_time) + ' minutes now, press the "Continue" button. \
                                        If you want to switch to an offensive deployment, press the "Switch" button.', self._name)
                    self._plot_times.append(self._time_left - self._resistance)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Continuing with the defensive deployment that has been going on for ' + str(self._deploy_time) + ' minutes.', self._name)
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Switch':
                        self._send_message('Switching to an offensive deployment after the defensive deployment of ' + str(self._deploy_time) + ' minutes.', self._name)
                        #self._defensive_deployment_time = 0
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        return None, {}

                if self._decide == self._name and self._tactic == 'offensive':
                    if self.received_messages_content and self.received_messages_content[-1] == 'Allocate to me' or self.received_messages_content and 'Allocating' in self.received_messages_content[-1]:
                        self._send_message('Allocating this decision with a predicted moral sensitivity of ' + str(self._sensitivity) + ' to you because you intervened.', self._name)
                        self._decide = 'human'
                    else:
                        if int(self._second) >= self._time + 15:
                            if self._resistance < 15 and self._duration > 60:
                                self._send_message('Switching to a defensive deployment after the offensive deployment of ' + str(self._offensive_deployment_time) + ' minutes, because the fire duration is more than 60 minutes \
                                                   and the estimated fire resistance to collapse is less than 15 minutes, making the chance of saving people and the building too low.', self._name)
                                #self._offensive_deployment_time = 0
                                self._plot_times.append(self._time_left - self._resistance)
                                self._tactic = 'defensive'
                                self._decide = None
                                self._phase = self._last_phase
                            else:
                                self._send_message('Continuing with the offensive deployment going on for ' + str(self._offensive_deployment_time) + ' minutes now, because there is still chance to save people and the building.', self._name)
                                #self._offensive_deployment_time = 0
                                self._plot_times.append(self._time_left - self._resistance)
                                self._tactic = 'offensive'
                                self._decide = None
                                self._phase = self._last_phase
                        else:
                            return None, {}

                if self._decide == self._name and self._tactic == 'defensive':
                    if self.received_messages_content and self.received_messages_content[-1] == 'Allocate to me' or self.received_messages_content and 'Allocating' in self.received_messages_content[-1]:
                        self._send_message('Allocating this decision with a predicted moral sensitivity of ' + str(self._sensitivity) + ' to you because you intervened.', self._name)
                        self._decide = 'human'
                    else:
                        if int(self._second) >= self._time + 15:
                            if self._resistance < 15 and self._duration > 60:
                                self._send_message('Continuing with the defensive deployment going on for ' + str(self._defensive_deployment_time) + ' minutes, because the fire duration is more than 60 minutes \
                                                   and the estimated fire resistance to collapse is less than 15 minutes, making the chance of saving people and the building too low.', self._name)
                                self._tactic = 'defensive'
                                self._decide = None
                                self._phase = self._last_phase
                            else:
                                self._send_message('Switching to an offensive deployment after the defensive deployment of ' + str(self._defensive_deployment_time) + ' minutes, because there is still chance to save people and the building.', self._name)
                                #self._defensive_deployment_time = 0
                                self._plot_times.append(self._time_left - self._resistance)
                                self._tactic = 'offensive'
                                self._decide = None
                                self._phase = self._last_phase
                        else:
                            return None, {}
    
                else:
                    return None, {}

            if self._time_left - self._resistance >= 5 and self._time_left - self._resistance <= 10 and self._location == '?' and not self._plot_generated and \
                self._current_location not in self._room_tiles and 'locate' not in self._situations:
                self._situations.append('locate')
                image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_at_time_" + str(self._resistance) + ".svg"
                self._sensitivity = R_to_Py_plot_locate(self._total_victims_cat, self._duration, self._resistance, self._temperature_cat, image_name)
                #self._sensitivity = 4
                self._plot_generated = True
                if self._condition == 'shap':
                    image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                if self._condition == 'util':
                    if self._temperature_cat != 'higher' and self._resistance > 15:
                        image_name = "<img src='/static/images/util_plots/util-locate-low.svg'/>"
                    else:
                        image_name = "<img src='/static/images/util_plots/util-locate-high.svg'/>"

                if self._sensitivity > self._threshold:
                    if self._condition == 'shap':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is above my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, self._name)
                    if self._condition == 'util':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is above my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                            + image_name, self._name)
                    if self._condition == 'baseline':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is above my allocation threshold.', self._name)
                    self._decide = 'human'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._time = int(self._second)
                    self._phase = Phase.LOCATE
                    return Idle.__name__, {'duration_in_ticks': 0}

                if self._sensitivity <= self._threshold:
                    if self._condition == 'shap':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is below my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, self._name)
                    if self._condition == 'util':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is below my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                            + image_name, self._name)
                    if self._condition == 'baseline':
                        self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                            I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                            is below my allocation threshold.', self._name)
                    self._decide = self._name
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._time = int(self._second)
                    self._phase = Phase.LOCATE
                    return Idle.__name__, {'duration_in_ticks': 0}

            if Phase.LOCATE == self._phase:
                if self._decide == 'human' and int(self._second) >= self._time + 15:
                    self._send_message('If you want to send in fire fighters to help locate the fire source, press the "Fire fighter" button. \
                                      If you do not want to send them in, press the "Continue" button.', self._name)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Not sending in fire fighters to help locate the fire source.', self._name)
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                        self._send_message('Sending in fire fighters to help locate the fire source.', self._name)
                        self._send_message('Target 1 is ' + str(self._potential_source_offices[0][0]) + ' and ' + str(self._potential_source_offices[0][1]) + ' in ' \
                                            + self._office_doors[self._potential_source_offices[0]] + ' target 2 is ' + str(self._potential_source_offices[-1][0]) + ' and ' \
                                            + str(self._potential_source_offices[-1][1]) + ' in ' +  self._office_doors[self._potential_source_offices[-1]], self._name)
                        self._phase = self._last_phase
                    else:
                        return None, {}
                
                if self._decide == self._name:
                    if self.received_messages_content and self.received_messages_content[-1] == 'Allocate to me' or self.received_messages_content and 'Allocating' in self.received_messages_content[-1]:
                        self._send_message('Allocating this decision with a predicted moral sensitivity of ' + str(self._sensitivity) + ' to you because you intervened.', self._name)
                        self._decide = 'human'
                    else:
                        if int(self._second) >= self._time + 15:
                            if self._temperature_cat != 'higher' and self._resistance > 15:
                                self._send_message('Sending in fire fighters to help locate because the estimated fire resistance to collapse (' + str(self._resistance) + ' minutes) is more than 15 minutes \
                                                and the temperate is lower than the auto-ignition temperatures of present substances.', self._name)
                                self._send_message('Target 1 is ' + str(self._potential_source_offices[0][0]) + ' and ' + str(self._potential_source_offices[0][1]) + ' in ' \
                                                    + self._office_doors[self._potential_source_offices[0]] + ' target 2 is ' + str(self._potential_source_offices[-1][0]) + ' and ' \
                                                    + str(self._potential_source_offices[-1][1]) + ' in ' +  self._office_doors[self._potential_source_offices[-1]], self._name)
                                self._phase = self._last_phase
                            else:
                                self._send_message('Not sending in fire fighters because the conditions are not safe enough for fire fighters to enter.', self._name)
                                self._phase = self._last_phase
                        else:
                            return None, {}
                else:
                    return None, {}
                
            if Phase.FIND_NEXT_GOAL == self._phase:
                self._id = None
                self._goal_victim = None
                self._goal_location = None
                zones = self._get_drop_zones(state)
                remaining_zones = []
                remaining_victims = []
                remaining = {}
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._rescued_victims:
                        remaining_zones.append(info)
                        remaining_victims.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remaining_zones = remaining_zones
                    self._remaining = remaining
                if not remaining_zones:
                    return None,{}
                if self._victims == 'known':
                    self._total_victims = len(remaining_victims) + len(self._rescued_victims)
                    if self._total_victims == 0:
                        self._total_victims_cat = 'none'
                    if self._total_victims == 1:
                        self._total_victims_cat = 'one'
                    if self._total_victims > 1:
                        self._total_victims_cat = 'multiple'
                if self._victims == 'unknown':
                    self._total_victims = '?'
                    self._total_victims_cat = 'unclear'
                self._send_message('Victims rescued: ' + str(len(self._rescued_victims)) + '/' + str(self._total_victims) + '.', 'RescueBot')
                for vic in remaining_victims:
                    if vic in self._found_victims and vic not in self._lost_victims:
                        self._goal_victim = vic
                        self._goal_location = remaining[vic]
                        if 'mild' in self._goal_victim:
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                        if 'critical' in self._goal_victim:
                            self._door = state.get_room_doors(self._victim_locations[vic]['room'])[0]
                            self._doormat = state.get_room(self._victim_locations[vic]['room'])[-1]['doormat']
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                        return Idle.__name__, {'duration_in_ticks': 0}              
                self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                if self._tactic == 'offensive':
                    unsearched_rooms = [room['room_name'] for room in state.values()
                    if 'class_inheritance' in room
                    and 'Door' in room['class_inheritance']
                    and room['room_name'] not in self._searched_rooms_offensive]
                if self._tactic == 'defensive':
                    unsearched_rooms = [room['room_name'] for room in state.values()
                    if 'class_inheritance' in room
                    and 'Door' in room['class_inheritance']
                    and room['room_name'] not in self._searched_rooms_defensive]
                if self._remaining_zones and len(unsearched_rooms) == 0:
                    if self._tactic == 'defensive':
                        self._searched_rooms_defensive = []
                        self._searched_rooms_defensive.append(self._door['room_name'])
                        self._send_message('Going to re-explore all offices to extinguish fires.', self._name)
                    if self._tactic == 'offensive':
                        self._searched_rooms_offensive = []
                        self._lost_victims = []
                        self._searched_rooms_offensive.append(self._door['room_name'])
                        self._offensive_search_rounds += 1
                        self._send_message('Going to re-explore all offices to rescue victims.', self._name)
                    self._send_messages = []
                    self._fire_locations = []
                    self.received_messages = []
                    self.received_messages_content = []
                    #self._searched_rooms.append(self._door['room_name'])
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    if self._current_door == None:
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, agent_location))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, agent_location))[-1]['doormat']
                        if self._door['room_name'] == 'office 1':
                            self._doormat = (2,4)
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    if self._current_door != None:
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, self._current_door))[-1]['doormat']
                        if self._door['room_name'] == 'office 1':
                            self._doormat = (2,4)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                if self._door['room_name'] == 'office 1':
                    self._doormat = (2,4)
                doorLoc = self._doormat
                self._current_room = self._door['room_name']
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                if self._tactic == 'offensive' and not self._goal_victim:
                    self._send_message('Moving to ' + str(self._door['room_name']) + ' to search for victims because it is the closest not explored office.', self._name)     
                if self._tactic == 'offensive' and self._goal_victim:
                    self._send_message('Moving to ' + str(self._door['room_name']) + ' to see if ' + self._goal_victim + ' can be rescued now.', self._name)  
                if self._tactic == 'defensive':
                    self._send_message('Moving to ' + str(self._door['room_name']) + ' to search for fire because it is the closest not explored office.', self._name)           
                self._current_door = self._door['location']
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}  
                self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED   

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                for info in state.values():
                    if 'class_inheritance' in info and 'IronObject' in info['class_inheritance'] and 'iron' in info['obj_id'] and info not in objects:
                        objects.append(info)
                        self._send_message('Iron debris is blocking ' + str(self._door['room_name']) + '. Removing iron debris.', self._name)
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'duration_in_ticks': 10}

                if len(objects) == 0:
                    self._phase = Phase.ENTER_ROOM
                    
            if Phase.ENTER_ROOM == self._phase:
                self._state_tracker.update(state)                 
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                room_tiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']]
                self._room_tiles = room_tiles               
                self._navigator.reset_full()
                if self._tactic == 'offensive':
                    self._navigator.add_waypoints(room_tiles)
                if self._tactic == 'defensive':
                    self._navigator.add_waypoints([self._door['location']])
                self._room_victims = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:                   
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._room_victims:
                                self._room_victims.append(vic)

                            if 'healthy' not in vic:# and vic not in self._found_victims:
                                self._recent_victim = vic
                                self._found_victims.append(vic)
                                self._victim_locations[vic] = {'location': info['location'], 'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '.', self._name)

                                if 'critical' in vic and not self._plot_generated:
                                    image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_for_vic_" + vic.replace(' ', '_') + str(self._offensive_search_rounds) + ".svg"
                                    distance = calculate_distances(self._fire_source_coords, self._victim_locations[vic]['location'])
                                    #distance = calculate_distances((2,8), self._victim_locations[vic]['location'])
                                    if distance < 14:
                                        self._distance = 'small'
                                    if distance >= 14:
                                        self._distance = 'large'
                                    if self._temperature_cat == 'close' or self._temperature_cat == 'lower':
                                        temperature = 'lower'
                                    if self._temperature_cat == 'higher':
                                        temperature = 'higher'
                                    self._sensitivity = R_to_Py_plot_rescue(self._duration, self._resistance, temperature, self._distance, image_name)
                                    self._plot_generated = True
                                    if self._condition == 'shap':
                                        image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                                    if self._condition == 'util':
                                        if self._temperature_cat != 'higher' and self._resistance > 15:
                                            if 'elderly man' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-low-granddad.svg'/>"                                            
                                            if 'elderly woman' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-low-grandma.svg'/>" 
                                            if 'injured woman' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-low-woman.svg'/>" 
                                            if 'injured man' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-low-man.svg'/>" 
                                        else:
                                            if 'elderly man' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-high-granddad.svg'/>"                                            
                                            if 'elderly woman' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-high-grandma.svg'/>" 
                                            if 'injured woman' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-high-woman.svg'/>" 
                                            if 'injured man' in vic:
                                                image_name = "<img src='/static/images/util_plots/util-rescue-high-man.svg'/>" 

                                    if self._sensitivity > self._threshold:
                                        if self._condition == 'shap':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is above my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                                + image_name, self._name)
                                        if self._condition == 'util':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is above my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                                + image_name, self._name)
                                        if self._condition == 'baseline':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is above my allocation threshold.', self._name)

                                        self._decide = 'human'
                                        self._time = int(self._second)
                                        self._phase = Phase.RESCUE
                                        return Idle.__name__, {'duration_in_ticks': 0}

                                    if self._sensitivity <= self._threshold:
                                        if self._condition == 'shap':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is below my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                                + image_name, self._name)
                                        if self._condition == 'util':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is below my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                                + image_name, self._name)
                                        if self._condition == 'baseline':
                                            self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                                We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                                I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                                is below my allocation threshold.', self._name)
                                        self._decide = self._name
                                        self._plot_times.append(self._time_left - self._resistance)
                                        self._time = int(self._second)
                                        self._phase = Phase.RESCUE
                                        return Idle.__name__, {'duration_in_ticks': 0}

                    return action, {}

                if self._room_victims:
                    if len(self._room_victims) == 1:
                        self._vic_string = 'victim'
                    if len(self._room_victims) > 1:
                        self._vic_string = 'victims'

                    for vic in self._room_victims:
                        if 'mild' in self._recent_victim and not self._plot_generated:
                            image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_for_vic_" + vic.replace(' ', '_') + ".svg"
                            self._sensitivity = R_to_Py_plot_priority(len(self._room_victims), self._smoke, self._duration, self._location_cat, image_name)
                            self._plot_generated = True
                            if self._condition == 'shap':
                                image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                            if self._condition == 'util':
                                if len(self._room_victims) == 1:
                                    if 'elderly man' in self._recent_victim:
                                        image_name = "<img src='/static/images/util_plots/util-evacuate-granddad.svg'/>"
                                    if 'elderly woman' in self._recent_victim:
                                        image_name = "<img src='/static/images/util_plots/util-evacuate-grandma.svg'/>"
                                    if 'injured man' in self._recent_victim:
                                        image_name = "<img src='/static/images/util_plots/util-evacuate-man.svg'/>"
                                    if 'injured woman' in self._recent_victim:
                                        image_name = "<img src='/static/images/util_plots/util-evacuate-woman.svg'/>"
                                if len(self._room_victims) > 1:
                                    image_name = "<img src='/static/images/util_plots/util-evacuate-multiple.svg'/>"

                            if self._sensitivity > self._threshold:
                                if self._condition == 'shap':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is above my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                        + image_name, self._name)
                                if self._condition == 'util':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is above my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                        + image_name, self._name)
                                if self._condition == 'baseline':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        Please make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is above my allocation threshold.', self._name)
  
                                self._decide = 'human'
                                self._time = int(self._second)
                                self._phase = Phase.PRIORITY
                                return Idle.__name__, {'duration_in_ticks': 0}

                            if self._sensitivity <= self._threshold:
                                if self._condition == 'shap':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is below my allocation threshold. This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                        + image_name, self._name)
                                if self._condition == 'util':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is below my allocation threshold. These are the positive and negative consequences of both decision options: \n \n ' \
                                                        + image_name, self._name)
                                if self._condition == 'baseline':
                                    self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                        We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                        I will make this decision because the predicted moral sensitivity of this situation (<b>' + str(self._sensitivity) + '</b>) \
                                                        is below my allocation threshold.', self._name)
                                self._decide = self._name
                                self._plot_times.append(self._time_left - self._resistance)
                                self._time = int(self._second)
                                self._phase = Phase.PRIORITY
                                return Idle.__name__, {'duration_in_ticks': 0}

                if self._tactic == 'offensive':
                    self._searched_rooms_offensive.append(self._door['room_name'])
                if self._tactic == 'defensive':
                    self._searched_rooms_defensive.append(self._door['room_name'])
                self._phase = Phase.FIND_NEXT_GOAL
                #if self._phase == Phase.FIND_NEXT_GOAL:
                #    return Idle.__name__,{'duration_in_ticks': 20}
                #if self._phase != Phase.FIND_NEXT_GOAL:
                #    return Idle.__name__,{'duration_in_ticks': 50}
            
            if Phase.RESCUE == self._phase:
                if self._decide == 'human' and int(self._second) >= self._time + 15:
                    self._send_message('If you want to send in fire fighters to rescue ' + self._recent_victim + ', press the "Fire fighter" button. \
                                      If you do not want to send them in, press the "Continue" button.', self._name)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                        self._send_message('Sending in fire fighters to rescue ' + self._recent_victim + '.', self._name)
                        vic_x = str(self._victim_locations[self._recent_victim]['location'][0])
                        vic_y = str(self._victim_locations[self._recent_victim]['location'][1])
                        drop_x = str(self._remaining[self._recent_victim][0])
                        drop_y = str(self._remaining[self._recent_victim][1])
                        self._send_message('Coordinates vic ' + vic_x + ' and ' + vic_y + ' coordinates drop ' + drop_x + ' and ' + drop_y, self._name)
                        if self._recent_victim not in self._rescued_victims:
                            self._rescued_victims.append(self._recent_victim)
                        if self._door['room_name'] not in self._searched_rooms_offensive:
                            self._searched_rooms_offensive.append(self._door['room_name'])
                        return None, {}
                    
                    if self.received_messages_content and self._recent_victim in self.received_messages_content[-1] and 'Delivered' in self.received_messages_content[-1]:
                        self._phase = Phase.FIND_NEXT_GOAL

                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Not sending in fire fighters to rescue ' + self._recent_victim + '.', self._name)
                        self._lost_victims.append(self._recent_victim)
                        self._searched_rooms_offensive.append(self._door['room_name'])
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        return None, {}

                if self._decide == self._name:
                    if self.received_messages_content and self.received_messages_content[-1] == 'Allocate to me' or self.received_messages_content and 'Allocating' in self.received_messages_content[-1]:
                        self._send_message('Allocating this decision with a predicted moral sensitivity of ' + str(self._sensitivity) + ' to you because you intervened.', self._name)
                        self._decide = 'human'
                    else:
                        if int(self._second) >= self._time + 15:
                            if self._temperature_cat != 'higher' and self._resistance > 15 and 'Delivered' not in self.received_messages_content[-1]:
                                self._send_message('Sending in fire fighters to rescue ' + self._recent_victim + ' because the temperature is lower than the auto-ignition temperatures of present substances \
                                                    and the estimated fire resistance to collapse is more than 15 minutes.', self._name)
                                vic_x = str(self._victim_locations[self._recent_victim]['location'][0])
                                vic_y = str(self._victim_locations[self._recent_victim]['location'][1])
                                drop_x = str(self._remaining[self._recent_victim][0])
                                drop_y = str(self._remaining[self._recent_victim][1])
                                self._send_message('Coordinates vic ' + vic_x + ' and ' + vic_y + ' coordinates drop ' + drop_x + ' and ' + drop_y, self._name)
                                if self._recent_victim not in self._rescued_victims:
                                    self._rescued_victims.append(self._recent_victim)
                                if self._door['room_name'] not in self._searched_rooms_offensive:
                                    self._searched_rooms_offensive.append(self._door['room_name'])
                                return None, {}
                            
                            if self.received_messages_content and self._recent_victim in self.received_messages_content[-1] and 'Delivered' in self.received_messages_content[-1]:
                                self._phase = Phase.FIND_NEXT_GOAL

                            else:
                                self._send_message('Not sending in fire fighters to rescue ' + self._recent_victim + ' because the conditions are not safe enough for fire fighters to enter.', self._name)
                                self._lost_victims.append(self._recent_victim)
                                self._searched_rooms_offensive.append(self._door['room_name'])
                                self._phase = Phase.FIND_NEXT_GOAL
                        else:
                            return None, {}
                else:
                    return None, {}

            if Phase.PRIORITY == self._phase:
                if self._decide == 'human' and int(self._second) >= self._time + 15:
                    self._send_message('If you want to first extinguish the fire in office ' + self._door['room_name'].split()[-1] + ', press the "Extinguish" button. \
                                      If you want to first evacuate the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ', press the "Evacuate" button.', self._name)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Extinguish' or self.received_messages_content and 'Extinguishing' in self.received_messages_content[-1] :
                        self._send_message('Extinguishing the fire in office ' + self._door['room_name'].split()[-1] + ' first.', self._name)
                        for info in state.values():
                            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                                self._id = info['obj_id']
                                if info['location'] not in self._extinguished_fire_locations:
                                    self._extinguished_fire_locations.append(info['location'])
                                return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 10}
                    if self.received_messages_content and self.received_messages_content[-1] == 'Evacuate':
                        self._send_message('Evacuating the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ' first.', self._name)
                        self._phase = Phase.FIND_NEXT_GOAL
                    if self._id and not state[{'obj_id': self._id}]:
                        self._phase = Phase.FIND_NEXT_GOAL
                        return Idle.__name__, {'duration_in_ticks': 0}
                    else:
                        return None, {}
                
                if self._decide == self._name:
                    if self.received_messages_content and self.received_messages_content[-1] == 'Allocate to me' or self.received_messages_content and 'Allocating' in self.received_messages_content[-1]:
                        self._send_message('Allocating this decision with a predicted moral sensitivity of ' + str(self._sensitivity) + ' to you because you intervened.', self._name)
                        self._decide = 'human'
                    else:
                        if int(self._second) >= self._time + 15:
                            if self._location == '?' and self._smoke == 'fast':
                                self._send_message('Evacuating the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ' first because the fire source is not located yet and the smoke is spreading fast.', self._name)
                                self._phase = Phase.FIND_NEXT_GOAL
                            else:
                                self._send_message('Extinguishing the fire in office ' + self._door['room_name'].split()[-1] + ' first because these are the general guidelines.', self._name)
                                for info in state.values():
                                    if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                                        self._id = info['obj_id']
                                        if info['location'] not in self._extinguished_fire_locations:
                                            self._extinguished_fire_locations.append(info['location'])
                                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 10}
                        else:
                            return None, {}

                    if self._id and not state[{'obj_id': self._id}]:
                        self._phase = Phase.FIND_NEXT_GOAL
                        return Idle.__name__, {'duration_in_ticks': 0}
                    else:
                        return None, {}
                    
                else:
                    return None, {}
                
            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                self._searched_rooms_offensive.append(self._door['room_name'])
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._victim_locations[self._goal_victim]['location']])
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM
                    
            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.TAKE_VICTIM
                    
            if Phase.TAKE_VICTIM == self._phase:
                self._send_message('Evacuating ' + self._goal_victim + ' to safety.', self._name)
                self._rescued_victims.append(self._goal_victim)
                self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                return CarryObject.__name__, {'object_id': self._victim_locations[self._goal_victim]['obj_id']}          

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goal_location])
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_VICTIM 

            if Phase.DROP_VICTIM == self._phase:
                if 'mild' in self._goal_victim:
                    self._send_message('Delivered '+ self._goal_victim + ' at the safe zone.', self._name)
                self._phase = Phase.FIND_NEXT_GOAL
                self._current_door = None
                return Drop.__name__, {}

    def _get_closest_room(self, state, objs, currentDoor):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)
        return min(dists,key=dists.get)
    
    def _send_message(self, mssg, sender):
        msg = Message(content = mssg, from_id = sender)
        if msg.content not in self.received_messages_content:
            self.send_message(msg)
            self._send_messages.append(msg.content)

    def _get_drop_zones(self, state:State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info:info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr']==0:
                zones.append(place)
        return zones