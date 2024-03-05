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

    
class brutus(custom_agent_brain):
    def __init__(self):
        super().__init__()
        self._phase=Phase.INTRO
        self._room_victims = []
        self._searched_rooms = []
        self._found_victims = []
        self._collected_victims = []
        self._modulos = []
        self._send_messages = []
        self._to_do = []
        self._to_search = []
        self._fire_locations = []
        self._area_tiles = []
        self._situations = []
        self._plot_times = []
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
        self._time_left = 31
        self._smoke = '?'
        self._temperature = '<≈'
        self._temperature_cat = 'close'
        self._location = '?'
        self._distance = '?'
        self._tactic = 'offensive'
        self._resistance = 31
        self._duration = 29
        self._offensive_deployment_time = 0
        self._defensive_deployment_time = 0

    #def update_time(self):
        #with self._counter_lock:
        #    self._resistance -= 1
        #    self._duration += 1
        #    if self._resistance < 0:
        #        self._resistance = 90  # Reset the counter after reaching 0
        #        self._duration = 0
        #    if self._tactic == 'offensive':
        #        self._offensive_deployment_time += 1
        #    if self._tactic == 'defensive':
        #        self._defensive_deployment_time += 1

        #self._send_message('Time left: ' + str(self._resistance) + '.', 'RescueBot')
        #self._send_message('Fire duration: ' + str(self._duration) + '.', 'RescueBot')
        # Schedule the next print
        #threading.Timer(6, self.update_time).start()

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)  
        load_R_to_Py()
        #self._counter_lock = threading.Lock()
        #self.update_time()

    def filter_bw4t_observations(self, state):
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
        self._send_message('Smoke spreads: ' + self._smoke + '.', 'RescueBot')
        self._send_message('Temperature: ' + self._temperature + '.', 'RescueBot')
        self._send_message('Location: ' + self._location + '.', 'RescueBot')
        self._send_message('Distance: ' + self._distance + '.', 'RescueBot')
        self._send_message('Our score is ' + str(state['brutus']['score']) +'.', 'Brutus')

        self._current_location = state[self.agent_id]['location']

        for info in state.values():
            if 'class_inheritance' in info and 'AreaTile' in info['class_inheritance'] and info['location'] not in self._area_tiles:
                self._area_tiles.append(info['location'])
            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'source' in info['obj_id']:
                self._send_message('Found fire source!', 'Brutus')
                self._location = 'found'
                self._smoke = info['smoke']
                if self._tactic == 'defensive':
                    self._phase = Phase.EXTINGUISH_CHECK
            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id'] and info['location'] not in self._fire_locations or \
                'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'source' in info['obj_id'] and info['location'] not in self._fire_locations:
                self._send_message('Found fire in ' + self._current_room + '.', 'Brutus')
                self._smoke = info['smoke']
                if self._tactic == 'defensive':
                    self._phase = Phase.EXTINGUISH_CHECK

        if self._location == 'found':
            for info in state.values():
                if 'class_inheritance' in info and 'EnvObject' in info['class_inheritance'] and 'fire source' in info['name']:
                    self._fire_source_coords = info['location']

        if self.received_messages_content and self.received_messages_content[-1] == 'Found fire source!':
            self._send_message('Fire source located and pinned on the map.', 'Brutus')
            # replace with location determined by world builder
            action_kwargs = add_object([(2,8)], "/images/fire2.svg", 3, 1, 'fire source')
            self._location = 'found' 
            return AddObject.__name__, action_kwargs

        if self._location == '?':
            self._location_cat = 'unknown'
        if self._location == 'found':
            self._location_cat = 'known'

        if self._time_left - self._resistance not in self._plot_times: #replace by list keeping track of all times where plots are send
            self._plot_generated = False

        while True:
            if Phase.EXTINGUISH_CHECK == self._phase:
                for info in state.values():
                    if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id'] and self._tactic == 'defensive' or \
                        'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'source' in info['obj_id'] and self._tactic == 'defensive':
                        self._fire_locations.append(info['location'])
                        self._send_message('Extinguishing fire in ' + self._current_room + '...', 'Brutus')
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 500, 'duration_in_ticks': 10}
                    if 'class_inheritance' in info and 'EnvObject' in info['class_inheritance'] and 'fire source' in info['name'] and self._tactic == 'defensive' and calculate_distances(self._current_location, info['location']) <= 3:
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 0}
                self._searched_rooms.append(self._current_room)
                self._phase = Phase.FIND_NEXT_GOAL

            if Phase.INTRO == self._phase:
                self._send_message('If you are ready to begin our mission, press the "Continue" button.', 'Brutus')
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if self._time_left - self._resistance >= 20 and self._time_left - self._resistance <= 25:
                self._situation = 'switch 1'

            if self._time_left - self._resistance >= 40 and self._time_left - self._resistance <= 45:
                self._situation = 'switch 2'

            if self._time_left - self._resistance >= 60 and self._time_left - self._resistance <= 65:
                self._situation = 'switch 3'

            if self._time_left - self._resistance >= 80 and self._time_left - self._resistance <= 85:
                self._situation = 'switch 4'

            if self._current_location not in self._area_tiles and not self._plot_generated and self._situation != None and self._situation not in self._situations:
                self._situations.append(self._situation)
                image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_at_time_" + str(self._resistance) + ".svg"
                sensitivity = R_to_Py_plot_tactic(self._total_victims_cat, self._location_cat, self._duration, self._resistance, image_name)
                sensitivity = 5
                self._plot_generated = True
                image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"

                if sensitivity >= 4.2:
                    if self._tactic == 'offensive':
                        self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                            We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                            Please make this decision because the predicted moral sensitivity of this situation is above my allocation threshold. \
                                            This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, 'Brutus')
                        self._deploy_time = self._offensive_deployment_time
                    if self._tactic == 'defensive':
                        self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                            We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                            Please make this decision because the predicted moral sensitivity of this situation is above my allocation threshold. \
                                            This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, 'Brutus')
                        self._deploy_time = self._defensive_deployment_time
                    self._decide = 'human'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._phase = Phase.TACTIC
                
                if sensitivity < 4.2:
                    if self._tactic == 'offensive':
                        self._send_message('My offensive deployment has been going on for ' + str(self._offensive_deployment_time) + ' minutes now. \
                                            We should decide whether to continue with the current offensive deployment, or switch to a defensive deployment. \
                                            I will make this decision because the predicted moral sensitivity of this situation is below my allocation threshold. \
                                            This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, 'Brutus')
                    if self._tactic == 'defensive':
                        self._send_message('My defensive deployment has been going on for ' + str(self._defensive_deployment_time) + ' minutes now. \
                                            We should decide whether to continue with the current defensive deployment, or switch to an offensive deployment. \
                                            I will make this decision because the predicted moral sensitivity of this situation is below my allocation threshold. \
                                            This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                            + image_name, 'Brutus')
                    self._decide = 'Brutus'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._phase = Phase.TACTIC
                    return Idle.__name__, {'duration_in_ticks':165}

            if Phase.TACTIC == self._phase:
                if self._decide == 'human' and self._tactic == 'offensive':
                    self._send_message('If you want to continue with the offensive deployment going on for ' + str(self._deploy_time) + ' minutes now, press the "Continue" button. \
                                        If you want to switch to a defensive deployment, press the "Switch" button.', 'Brutus')
                    self._plot_times.append(self._time_left - self._resistance)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Continuing with the offensive deployment that has been going on for ' + str(self._deploy_time) + ' minutes, because you decided to.', 'Brutus')
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Switch':
                        self._send_message('Switching to a defensive deployment after the offensive deployment of ' + str(self._deploy_time) + ' minutes, because you decided to.', 'Brutus')
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        return None, {}

                if self._decide == 'human' and self._tactic == 'defensive':
                    self._send_message('If you want to continue with the defensive deployment going on for ' + str(self._deploy_time) + ' minutes now, press the "Continue" button. \
                                        If you want to switch to an offensive deployment, press the "Switch" button.', 'Brutus')
                    self._plot_times.append(self._time_left - self._resistance)
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Continuing with the defensive deployment that has been going on for ' + str(self._deploy_time) + ' minutes, because you decided to.', 'Brutus')
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Switch':
                        self._send_message('Switching to an offensive deployment after the defensive deployment of ' + str(self._deploy_time) + ' minutes, because you decided to.', 'Brutus')
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        return None, {}

                # ADD MORE CONDITIONS FOR BRUTUS TO MAKE DECISION ABOUT SWITCHING TACTICS, FOR EXAMPLE WRT HOW DANGEROUS SITUATION IS (CHECK GUIDELINES)    
                if self._decide == 'Brutus' and self._tactic == 'offensive':
                    if self._resistance > 15 and self._duration < 45:
                        self._send_message('Continuing with the offensive deployment going on for ' + str(self._offensive_deployment_time) + ' minutes now, because the fire duration is less than 45 minutes \
                                            and the estimated fire resistance to collapse is more than 15 minutes.', 'Brutus')
                        self._plot_times.append(self._time_left - self._resistance)
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        self._send_message('Switching to a defensive deployment after the offensive deployment of ' + str(self._offensive_deployment_time) + ' minutes, because the chance of saving people and the building is too low.', 'Brutus')
                        self._plot_times.append(self._time_left - self._resistance)
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase

                if self._decide == 'Brutus' and self._tactic == 'defensive':
                    if self._resistance > 15 and self._duration < 45:
                        self._send_message('Switching to an offensive deployment after the defensive deployment of ' + str(self._defensive_deployment_time) + ' minutes, because the fire duration is less than 45 minutes \
                                            and the estimated fire resistance to collapse is more than 15 minutes.', 'Brutus')
                        self._plot_times.append(self._time_left - self._resistance)
                        self._tactic = 'offensive'
                        self._decide = None
                        self._phase = self._last_phase
                    else:
                        self._send_message('Continuing with the defensive deployment going on for ' + str(self._defensive_deployment_time) + ' minutes, because the chance of saving people and the building is too low.', 'Brutus')
                        self._tactic = 'defensive'
                        self._decide = None
                        self._phase = self._last_phase
    
                else:
                    return None, {}

            if self._time_left - self._resistance >= 5 and self._time_left - self._resistance <= 10 and self._location == '?' and not self._plot_generated and \
                self._current_location not in self._area_tiles and 'locate' not in self._situations:
                self._situations.append('locate')
                image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_at_time_" + str(self._resistance) + ".svg"
                sensitivity = R_to_Py_plot_locate(self._total_victims_cat, self._duration, self._resistance, self._temperature_cat, image_name)
                sensitivity = 5
                self._plot_generated = True
                image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                if sensitivity >= 4.2:
                    self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                      Please make this decision because the predicted moral sensitivity of this situation is above my allocation threshold. \
                                      This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                      + image_name, 'Brutus')
                    self._decide = 'human'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._phase = Phase.LOCATE

                if sensitivity < 4.2:
                    self._send_message('The location of the fire source still has not been found, so we should decide whether to send in fire fighters to help locate the fire source or if sending them in is too dangerous. \
                                      I will make this decision because the predicted moral sensitivity of this situation is below my allocation threshold. \
                                      This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                      + image_name, 'Brutus')
                    self._decide = 'Brutus'
                    self._plot_times.append(self._time_left - self._resistance)
                    self._last_phase = self._phase
                    self._phase = Phase.LOCATE
                    return Idle.__name__, {'duration_in_ticks':165}

            if Phase.LOCATE == self._phase:
                if self._decide == 'human':
                    self._send_message('If you want to send in fire fighters to help locate the fire source, press the "Fire fighter" button. \
                                      If you do not want to send them in, press the "Continue" button.', 'Brutus')
                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Not sending in fire fighters to help locate the fire source because you decided to.', 'Brutus')
                        self._phase = self._last_phase
                    if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                        self._send_message('Sending in fire fighters to help locate the fire source because you decided to.', 'Brutus')
                        # replace by location obtained from world/task configuration
                        self._send_message('Target 1 is 2 and 7 in 5 target 2 is 16 and 21 in 13', 'Brutus')
                        self._phase = self._last_phase
                    else:
                        return None, {}
                
                # ADD MORE CONDITIONS FOR BRUTUS TO MAKE DECISION ABOUT SENDING IN FIREFIGHTERS TO LOCATE FIRE SOURCE, FOR EXAMPLE WRT RESISTENCE TO COLLAPSE
                if self._decide == 'Brutus':
                    if self._temperature_cat != 'higher' and self._resistance > 15:
                        self._send_message('Sending in fire fighters to help locate because the estimated fire resistance to collapse (' + str(self._resistance) + ' minutes) is more than 15 minutes \
                                           and the temperate is lower than the auto-ignition temperatures of present substances.', 'Brutus')
                        # replace by location obtained from world/task configuration
                        self._send_message('Target 1 is 2 and 7 in 5 target 2 is 16 and 21 in 13', 'Brutus')
                        self._phase = self._last_phase
                    else:
                        self._send_message('Not sending in fire fighters because the conditions are not safe enough for fire fighters to enter.', 'Brutus')
                        self._phase = self._last_phase
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
                    if str(info['img_name'])[8:-4] not in self._collected_victims:
                        remaining_zones.append(info)
                        remaining_victims.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remaining_zones = remaining_zones
                    self._remaining = remaining
                if not remaining_zones:
                    return None,{}
                self._total_victims = len(remaining_victims) + len(self._collected_victims)
                if self._total_victims == 0:
                    self._total_victims_cat = 'none'
                if self._total_victims == 1:
                    self._total_victims_cat = 'one'
                if self._total_victims == 'unknown':
                    self._total_victims_cat = 'unclear'
                if self._total_victims > 1:
                    self._total_victims_cat = 'multiple'
                self._send_message('Victims rescued: ' + str(len(self._collected_victims)) + '/' + str(self._total_victims) + '.', 'RescueBot')
                for vic in remaining_victims:
                    if vic in self._found_victims and vic not in self._to_do:
                        self._goal_victim = vic
                        self._goal_location = remaining[vic]
                        self._phase = Phase.PLAN_PATH_TO_VICTIM
                        return Idle.__name__, {'duration_in_ticks':25}              
                self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                unsearched_rooms = [room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searched_rooms
                and room['room_name'] not in self._to_search]
                if self._remaining_zones and len(unsearched_rooms) == 0:
                    self._to_search = []
                    self._to_do = []
                    self._searched_rooms = []
                    self._send_messages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._searched_rooms.append(self._door['room_name'])
                    self._send_message('Going to re-explore all areas.','Brutus')
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    if self._current_door == None:
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, agent_location))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, agent_location))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (2,4)
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    if self._current_door != None:
                        self._door = state.get_room_doors(self._get_closest_room(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = state.get_room(self._get_closest_room(state, unsearched_rooms, self._current_door))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (2,4)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                if self._door['room_name'] == 'area 1':
                    self._doormat = (2,4)
                doorLoc = self._doormat
                self._current_room = self._door['room_name']
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                self._send_message('Moving to ' + str(self._door['room_name']) + ' because it is the closest not explored area.', 'Brutus')                   
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
                        self._send_message('Iron debris is blocking ' + str(self._door['room_name']) + '. Removing iron debris ...', 'Brutus')
                        return RemoveObject.__name__, {'object_id': info['obj_id'], 'duration_in_ticks': 0}

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

                            if vic in self._found_victims and 'location' not in self._victim_locations[vic].keys():
                                self._victim_locations[vic] = {'location': info['location'], 'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                if vic == self._goal_victim:
                                    self._send_message('Found '+ vic + ' in ' + self._door['room_name'] + ' because you told me '+ vic + ' was located here.', 'Brutus')
                                    self._searched_rooms.append(self._door['room_name'])
                                    self._phase = Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_victim = vic
                                self._found_victims.append(vic)
                                self._victim_locations[vic] = {'location': info['location'], 'room': self._door['room_name'], 'obj_id': info['obj_id']}
                                self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '.','Brutus')

                                if 'critical' in vic and not self._plot_generated:
                                    image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_for_vic_" + vic.replace(' ', '_') + ".svg"
                                    distance = calculate_distances(self._fire_source_coords, self._victim_locations[vic]['location'])
                                    #distance = calculate_distances((2,8), self._victim_locations[vic]['location'])
                                    if distance < 16:
                                        self._distance = 'small'
                                    if distance >= 16:
                                        self._distance = 'large'
                                    if self._temperature_cat == 'close' or self._temperature_cat == 'lower':
                                        temperature = 'lower'
                                    if self._temperature_cat == 'higher':
                                        temperature = 'higher'
                                    sensitivity = R_to_Py_plot_rescue(self._duration, self._resistance, temperature, self._distance, image_name)
                                    sensitivity = 5
                                    self._plot_generated = True
                                    image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                                    if sensitivity >= 4.2:
                                        self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                        We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                        Please make this decision because the predicted moral sensitivity of this situation is above my allocation threshold. \
                                                        This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                        + image_name, 'Brutus')
                                        self._decide = 'human'
                                        self._phase = Phase.RESCUE
                                        return Idle.__name__, {'duration_in_ticks': 25}

                                    if sensitivity < 4.2:
                                        self._send_message('I have found ' + vic + ' who I cannot evacuate to safety myself. \
                                                        We should decide whether to send in fire fighters to rescue this victim, or if sending them in is too dangerous. \
                                                        I will make this decision because the predicted moral sensitivity of this situation is below my allocation threshold. \
                                                        This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                        + image_name, 'Brutus')
                                        self._decide = 'Brutus'
                                        self._plot_times.append(self._time_left - self._resistance)
                                        self._phase = Phase.RESCUE
                                        return Idle.__name__, {'duration_in_ticks': 25}

                    return action, {}

                if self._room_victims:
                    if len(self._room_victims) == 1:
                        self._vic_string = 'victim'
                    if len(self._room_victims) > 1:
                        self._vic_string = 'victims'

                    for vic in self._room_victims:
                        if 'mild' in self._recent_victim and not self._plot_generated:
                            image_name = "/home/ruben/xai4mhc/TUD-Research-Project-2022/custom_gui/static/images/sensitivity_plots/plot_for_vic_" + vic.replace(' ', '_') + ".svg"
                            sensitivity = R_to_Py_plot_priority(len(self._room_victims), self._smoke, self._duration, self._location_cat, image_name)
                            sensitivity = 5
                            self._plot_generated = True
                            image_name = "<img src='/static/images" + image_name.split('/static/images')[-1] + "' />"
                            if sensitivity >= 4.2:
                                self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                Please make this decision because the predicted moral sensitivity of this situation is above my allocation threshold. \
                                                This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, 'Brutus')
                                self._decide = 'human'
                                self._phase = Phase.PRIORITY
                                return Idle.__name__, {'duration_in_ticks': 25}

                            if sensitivity < 4.2:
                                 self._send_message('I have found ' + str(len(self._room_victims)) + ' mildly injured ' + self._vic_string + ' in the burning office ' + self._door['room_name'].split()[-1] + '. \
                                                We should decide whether to first extinguish the fire or evacuate the ' + self._vic_string + '. \
                                                I will make this decision because the predicted moral sensitivity of this situation is below my allocation threshold. \
                                                This is how much each feature contributed to the predicted sensitivity: \n \n ' \
                                                + image_name, 'Brutus')
                                 self._decide = 'Brutus'
                                 self._plot_times.append(self._time_left - self._resistance)
                                 self._phase = Phase.PRIORITY
                                 return Idle.__name__, {'duration_in_ticks': 25}

                self._searched_rooms.append(self._door['room_name'])
                self._phase = Phase.FIND_NEXT_GOAL
                if self._phase == Phase.FIND_NEXT_GOAL:
                    return Idle.__name__,{'duration_in_ticks':5}
                if self._phase != Phase.FIND_NEXT_GOAL:
                    return Idle.__name__,{'duration_in_ticks':165}
            
            if Phase.RESCUE == self._phase:
                if self._decide == 'human':
                    self._send_message('If you want to send in fire fighters to rescue ' + self._recent_victim + ', press the "Fire fighter" button. \
                                      If you do not want to send them in, press the "Continue" button.', 'Brutus')
                    if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                        self._send_message('Sending in fire fighters to rescue ' + self._recent_victim + ' because you decided to.', 'Brutus')
                        vic_x = str(self._victim_locations[self._recent_victim]['location'][0])
                        vic_y = str(self._victim_locations[self._recent_victim]['location'][1])
                        drop_x = str(self._remaining[self._recent_victim][0])
                        drop_y = str(self._remaining[self._recent_victim][1])
                        self._send_message('Coordinates vic ' + vic_x + ' and ' + vic_y + ' coordinates drop ' + drop_x + ' and ' + drop_y, 'Brutus')
                        if self._recent_victim not in self._collected_victims:
                            self._collected_victims.append(self._recent_victim)
                        if self._door['room_name'] not in self._searched_rooms:
                            self._searched_rooms.append(self._door['room_name'])
                        return None, {}
                    
                    if self.received_messages_content and self._recent_victim in self.received_messages_content[-1] and 'Delivered' in self.received_messages_content[-1]:
                        self._phase = Phase.FIND_NEXT_GOAL

                    if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                        self._send_message('Not sending in fire fighters to rescue ' + self._recent_victim + ' because you decided to.', 'Brutus')
                        self._collected_victims.append(self._recent_victim)
                        self._searched_rooms.append(self._door['room_name'])
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        return None, {}

                # ADD MORE CONDITIONS
                if self._decide == 'Brutus':
                    if self._temperature_cat != 'higher' and self._resistance > 15 and 'Delivered' not in self.received_messages_content[-1]:
                        self._send_message('Sending in fire fighters to rescue ' + self._recent_victim + ' because the temperature is lower than the auto-ignition temperatures of present substances \
                                            and the estimated fire resistance to collapse is more than 15 minutes.', 'Brutus')
                        vic_x = str(self._victim_locations[self._recent_victim]['location'][0])
                        vic_y = str(self._victim_locations[self._recent_victim]['location'][1])
                        drop_x = str(self._remaining[self._recent_victim][0])
                        drop_y = str(self._remaining[self._recent_victim][1])
                        self._send_message('Coordinates vic ' + vic_x + ' and ' + vic_y + ' coordinates drop ' + drop_x + ' and ' + drop_y, 'Brutus')
                        if self._recent_victim not in self._collected_victims:
                            self._collected_victims.append(self._recent_victim)
                        if self._door['room_name'] not in self._searched_rooms:
                            self._searched_rooms.append(self._door['room_name'])
                        return None, {}
                    
                    if self.received_messages_content and self._recent_victim in self.received_messages_content[-1] and 'Delivered' in self.received_messages_content[-1]:
                        self._phase = Phase.FIND_NEXT_GOAL

                    else:
                        self._send_message('Not sending in fire fighters to rescue ' + self._recent_victim + ' because the conditions are not safe enough for fire fighters to enter.', 'Brutus')
                        self._collected_victims.append(self._recent_victim)
                        self._searched_rooms.append(self._door['room_name'])
                        self._phase = Phase.FIND_NEXT_GOAL
   
                else:
                    return None, {}

            if Phase.PRIORITY == self._phase:
                if self._decide == 'human':
                    self._send_message('If you want to first extinguish the fire in office ' + self._door['room_name'].split()[-1] + ', press the "Extinguish" button. \
                                      If you want to first evacuate the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ', press the "Evacuate" button.', 'Brutus')
                    if self.received_messages_content and self.received_messages_content[-1] == 'Extinguish' or self.received_messages_content and 'Extinguishing' in self.received_messages_content[-1] :
                        self._send_message('Extinguishing the fire in office ' + self._door['room_name'].split()[-1] + ' first because you decided to.', 'Brutus')
                        for info in state.values():
                            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                                self._id = info['obj_id']
                                return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 10}
                    if self.received_messages_content and self.received_messages_content[-1] == 'Evacuate':
                        self._send_message('Evacuating the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ' first because you decided to.', 'Brutus')
                        self._phase = Phase.FIND_NEXT_GOAL
                    if self._id and not state[{'obj_id': self._id}]:
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        return None, {}
                
                # ADD MORE CONDITIONS FOR BRUTUS TO MAKE DECISION
                if self._decide == 'Brutus':
                    if self._location == '?' and self._smoke == 'fast':
                        self._send_message('Evacuating the ' + self._vic_string + ' in office ' + self._door['room_name'].split()[-1] + ' first because the fire source is not located yet and the smoke is spreading fast.', 'Brutus')
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        self._send_message('Extinguishing the fire in office ' + self._door['room_name'].split()[-1] + ' first because these are the general guidelines.', 'Brutus')
                        for info in state.values():
                            if 'class_inheritance' in info and 'FireObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                                self._id = info['obj_id']
                                return RemoveObject.__name__, {'object_id': info['obj_id'], 'remove_range': 5, 'duration_in_ticks': 10}
                if self._id and not state[{'obj_id': self._id}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}
                
            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                self._searched_rooms.append(self._door['room_name'])
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
                self._send_message('Evacuating ' + self._goal_victim + ' to safety.', 'Brutus')
                self._collected_victims.append(self._goal_victim)
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
                    self._send_message('Delivered '+ self._goal_victim + ' at the safe zone.', 'Brutus')
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