import sys, random, enum, ast, time
from matrx import grid_world
from brains1.BW4TBrain import BW4TBrain
from actions1.customActions import *
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
from actions1.customActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop

class Phase(enum.Enum):
    INTRO0 = 0,
    INTRO1 = 1,
    INTRO2 = 2,
    INTRO3 = 3,
    INTRO4 = 4,
    INTRO5 = 5,
    INTRO6 = 6,
    INTRO7 = 7,
    INTRO8 = 8,
    INTRO9 = 9,
    INTRO10 = 10,
    INTRO11 = 11,
    FIND_NEXT_GOAL = 12,
    PICK_UNSEARCHED_ROOM = 13,
    PLAN_PATH_TO_ROOM = 14,
    FOLLOW_PATH_TO_ROOM = 15,
    PLAN_ROOM_SEARCH_PATH = 16,
    FOLLOW_ROOM_SEARCH_PATH = 17,
    PLAN_PATH_TO_VICTIM = 18,
    FOLLOW_PATH_TO_VICTIM = 19,
    TAKE_VICTIM = 20,
    PLAN_PATH_TO_DROPPOINT = 21,
    FOLLOW_PATH_TO_DROPPOINT = 22,
    DROP_VICTIM = 23,
    WAIT_FOR_HUMAN = 24,
    WAIT_AT_ZONE = 25,
    FIX_ORDER_GRAB = 26,
    FIX_ORDER_DROP = 27,
    REMOVE_OBSTACLE_IF_NEEDED = 28,
    ENTER_ROOM = 29,
    PLAN_EXIT = 30,
    FOLLOW_EXIT_PATH = 31


class BaselineAgent(BW4TBrain):
    def __init__(self, slowdown: int):
        super().__init__(slowdown)
        self._slowdown = slowdown
        self._phase = Phase.INTRO0
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 9600
        self._sendMessages = []
        self._currentDoor = None
        self._providedExplanations = []
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._second = None
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []
        self._answered = False
        self._tosearch = []
        self._ignored = 0
        self._followed = 0
        self._noSuggestions = 0
        self._suggestion = []
        self._carrying = False
        self._waiting = False
        self._confidence = True
        self._rescued = []
        self._goalVic = None
        self._location = '?'

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        return state

    def decide_on_bw4t_action(self, state: State):
        #print(self._phase)
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
        # Process messages from team members
        self._processMessages(state, self._teamMembers)
        while True:            
            if Phase.INTRO0 == self._phase:
                if self.received_messages_content and 'Coordinates' in self.received_messages_content[-1] and self._goalVic not in self._rescued:
                    msg = self.received_messages_content[-1]
                    self._dropLoc = tuple((int(msg.split()[-3]), int(msg.split()[-1])))
                    self._goalLoc = tuple((int(msg.split()[2]), int(msg.split()[4])))
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                    return Idle.__name__, {'duration_in_ticks': 0}
                if self.received_messages_content and 'Target' in self.received_messages_content[-1] and self._location == '?' and agent_name!='fire_fighter':
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return Idle.__name__, {'duration_in_ticks': 0}
                else:
                    return None, {}
                
            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                if agent_name and agent_name == 'sebastiaan':
                    self._navigator.add_waypoints([(2, 7)])
                if agent_name and agent_name == 'robbert':
                    self._navigator.add_waypoints([(16, 21)])
                self._phase=Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.PLAN_ROOM_SEARCH_PATH
                    
            if Phase.ENTER_ROOM==self._phase:
                self._state_tracker.update(state)                 
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                if agent_name and agent_name == 'sebastiaan':
                    area = 'area 5'
                if agent_name and agent_name == 'robbert':
                    area = 'area 13'
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == area
                ]
                self._roomtiles=roomTiles               
                self._navigator.reset_full()
                self._navigator.add_waypoints(roomTiles)
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:                   
                    for info in state.values():
                        if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'source' in info['obj_id']:
                            self._sendMessage('Found fire source!', 'Robbert')
                            self._location = '✔'
                    return action,{}
                self._phase = Phase.PLAN_EXIT
                #self._phase=Phase.INTRO0
                #return Idle2.__name__,{'duration_in_ticks':0}

            if Phase.PLAN_EXIT==self._phase:
                self._navigator.reset_full()
                if agent_name and agent_name == 'sebastiaan':
                    loc = (0, 11)
                if agent_name and agent_name == 'robbert':
                    loc = (0, 13)
                self._navigator.add_waypoints([loc])
                self._phase=Phase.FOLLOW_EXIT_PATH

            if Phase.FOLLOW_EXIT_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase=Phase.INTRO0
                return Idle2.__name__,{'duration_in_ticks':0}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalLoc])
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                        self._goalVic = info['img_name'][8:-4]
                        return CarryObject.__name__, {'object_id': info['obj_id']}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._dropLoc])
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                self._rescued.append(self._goalVic)
                self._sendMessage('Delivered ' + self._goalVic + ' at the drop zone.', 'RescueBot')
                self._phase = Phase.INTRO0
                self._currentDoor = None
                self._goalVic = None
                return Drop.__name__, {'duration_in_ticks':0}

    def _getDropZones(self, state: State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    def _processMessages(self, state, teamMembers):
        '''
        process incoming messages.
        Reported blocks are added to self._blocks
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)

        for mssgs in receivedMessages.values():
            for msg in mssgs:
                if msg.startswith("Search:"):
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                if msg.startswith("Found:"):
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room': loc}
                    if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room': loc}
                    if 'mild' in foundVic:
                        self._todo.append(foundVic)
                if msg.startswith('Collect:'):
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if collectVic not in self._foundVictims:
                        self._foundVictims.append(collectVic)
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room': loc}
                    if collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                if msg.startswith('Remove:'):
                    if not self._carrying:
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searchedRooms:
                            self._searchedRooms.remove(area)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._remove = True
                        self._sendMessage(
                            'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                            'RescueBot')
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._sendMessage('Will come to ' + area + ' after dropping ' + self._goalVic + '.',
                                          'RescueBot')
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']:
                self._humanLoc = int(mssgs[-1].split()[-1])


    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
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

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs

    def _dynamicMessage(self, mssg1, mssg2, explanation, sender):
        if explanation not in self._providedExplanations:
            self._sendMessage(mssg1, sender)
            self._providedExplanations.append(explanation)
        if 'Searching' in mssg1:
            if explanation in self._providedExplanations and mssg1 not in self._sendMessages[-5:]:
                self._sendMessage(mssg2, sender)
        if 'Found' in mssg1:
            history = [mssg2[:-1] in mssg for mssg in self._sendMessages]
            if explanation in self._providedExplanations and True not in history:
                self._sendMessage(mssg2, sender)
        if 'Searching' not in mssg1 and 'Found' not in mssg1:
            if explanation in self._providedExplanations and self._sendMessages[-1] != mssg1:
                self._sendMessage(mssg2, sender)