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
from actions1.customActions import Backup, RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop, Injured

class Phase(enum.Enum):
    START=1,
    INJURED=9,
    BACKUP2=10,
    BACKUP=11,
    FIND_NEXT_GOAL=12,
    PICK_UNSEARCHED_ROOM=13,
    PLAN_PATH_TO_ROOM=14,
    FOLLOW_PATH_TO_ROOM=15,
    PLAN_ROOM_SEARCH_PATH=16,
    FOLLOW_ROOM_SEARCH_PATH=17,
    PLAN_PATH_TO_VICTIM=18,
    FOLLOW_PATH_TO_VICTIM=19,
    TAKE_VICTIM=20,
    PLAN_PATH_TO_DROPPOINT=21,
    FOLLOW_PATH_TO_DROPPOINT=22,
    DROP_VICTIM=23,
    WAIT_FOR_HUMAN=24,
    WAIT_AT_ZONE=25,
    FIX_ORDER_GRAB=26,
    FIX_ORDER_DROP=27,
    REMOVE_OBSTACLE_IF_NEEDED=28,
    ENTER_ROOM=29
    
class TutorialAgent2(BW4TBrain):
    def __init__(self, slowdown:int):
        super().__init__(slowdown)
        self._slowdown = slowdown
        self._phase=Phase.START
        self._roomVics = []
        self._searchedRooms = []
        self._foundVictims = []
        self._collectedVictims = []
        self._foundVictimLocs = {}
        self._maxTicks = 9600
        self._sendMessages = []
        self._currentDoor=None 
        #self._condition = condition
        self._providedExplanations = []   
        self._teamMembers = []
        self._carryingTogether = False
        self._remove = False
        self._goalVic = None
        self._goalLoc = None
        self._second = None
        self._criticalRescued = 0
        self._humanLoc = None
        self._distanceHuman = None
        self._distanceDrop = None
        self._agentLoc = None
        self._todo = []
        self._answered = False
        self._tosearch = []
        self._tutorial = True
        self._decided = False
        self._co = 0
        self._hcn = 0

    def initialize(self):
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, 
            action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_bw4t_observations(self, state):
        #self._processMessages(state)
        return state

    def decide_on_bw4t_action(self, state:State):
        for info in state.values():
            if 'class_inheritance' in info and 'SmokeObject' in info['class_inheritance']:
                self._co = info['co_ppm']
                self._hcn = info['hcn_ppm']
        if not state[{'class_inheritance':'SmokeObject'}]:
            self._co = 0
            self._hcn = 0
        self._criticalFound = 0
        for vic in self._foundVictims:
            if 'critical' in vic:
                self._criticalFound+=1
        
        if state[{'is_human_agent':True}]:
            self._distanceHuman = 'close'
        if not state[{'is_human_agent':True}]: 
            if self._agentLoc in [1,2,3,4,5,6,7] and self._humanLoc in [8,9,10,11,12,13,14]:
                self._distanceHuman = 'far'
            if self._agentLoc in [1,2,3,4,5,6,7] and self._humanLoc in [1,2,3,4,5,6,7]:
                self._distanceHuman = 'close'
            if self._agentLoc in [8,9,10,11,12,13,14] and self._humanLoc in [1,2,3,4,5,6,7]:
                self._distanceHuman = 'far'
            if self._agentLoc in [8,9,10,11,12,13,14] and self._humanLoc in [8,9,10,11,12,13,14]:
                self._distanceHuman = 'close'

        if self._agentLoc in [1,2,5,6,8,9,11,12]:
            self._distanceDrop = 'far'
        if self._agentLoc in [3,4,7,10,13,14]:
            self._distanceDrop = 'close'

        self._second = state['World']['tick_duration'] * state['World']['nr_ticks']

        for info in state.values():
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying'])>0 and 'critical' in info['is_carrying'][0]['obj_id']:
                self._collectedVictims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carryingTogether = True
            if 'is_human_agent' in info and 'Human' in info['name'] and len(info['is_carrying'])==0:
                self._carryingTogether = False
        if self._carryingTogether == True:
            return None, {}
        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member!=agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)       
        # Process messages from team members
        self._processMessages(state, self._teamMembers)
        # Update trust beliefs for team members
        #self._trustBlief(self._teamMembers, receivedMessages)

        # CRUCIAL TO NOT REMOVE LINE BELOW!
        #self._sendMessage('Our score is ' + str(state['Brutus']['score']) +'.', 'Brutus')
        while True:          
            if Phase.START==self._phase:
                self._phase=Phase.FIND_NEXT_GOAL
                return Idle.__name__,{'duration_in_ticks':50}          
            if Phase.FIND_NEXT_GOAL==self._phase:
                self._answered = False
                self._advice = False
                self._goalVic = None
                self._goalLoc = None
                zones = self._getDropZones(state)
                remainingZones = []
                remainingVics = []
                remaining = {}
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    #self._goalVic = str(remainingZones[0]['img_name'])[8:-4]
                    #self._goalLoc = remainingZones[0]['location']
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                if not remainingZones:
                    return None,{}

                for vic in remainingVics:
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                        if 'location' in self._foundVictimLocs[vic].keys():
                            self._sendMessage('Please decide whether it is safe enough to call in a "Fire fighter" to rescue ' + self._goalVic +  ', or whether to "Continue" exploring because it is not safe enough to send in fire fighter. \n \n \
                                Important features to consider: \n - Pinned victims located: ' + str(self._collectedVictims) + '\n - toxic concentrations: ' + str(self._hcn) + ' ppm HCN and ' + str(self._co) + ' ppm CO \n \n \
                                I suggest to call in a "Fire fighter" to rescue ' + self._goalVic + ' because: \n - toxic concentrations: ppm HCN < 40 and ppm CO < 500','Brutus')
                            if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                                self._collectedVictims.append(self._goalVic)
                                self._phase=Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                            #self._sendMessage('Extinguishing fire blocking ' + str(self._door['room_name']) + '.','Brutus')
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__,{'duration_in_ticks':25}  
                        else:
                            return None,{}
                            #self._phase=Phase.PLAN_PATH_TO_VICTIM
                            #return Idle.__name__,{'duration_in_ticks':25}  
                        #if 'location' not in self._foundVictimLocs[vic].keys():
                        #    self._phase=Phase.PLAN_PATH_TO_ROOM
                        #    return Idle.__name__,{'duration_in_ticks':25}              
                self._phase=Phase.PICK_UNSEARCHED_ROOM
                #return Idle.__name__,{'duration_in_ticks':25}

            if Phase.PICK_UNSEARCHED_ROOM==self._phase:
                self._advice = False
                agent_location = state[self.agent_id]['location']
                unsearchedRooms=[room['room_name'] for room in state.values()
                if 'class_inheritance' in room
                and 'Door' in room['class_inheritance']
                and room['room_name'] not in self._searchedRooms
                and room['room_name'] not in self._tosearch]
                if self._remainingZones and len(unsearchedRooms) == 0:
                    self._tosearch = []
                    self._todo = []
                    self._searchedRooms = []
                    self._sendMessages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._searchedRooms.append(self._door['room_name'])
                    self._sendMessage('Going to re-explore all areas.','Brutus')
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    if self._currentDoor==None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,agent_location))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state,unsearchedRooms,agent_location))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3,5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    if self._currentDoor!=None:
                        self._door = state.get_room_doors(self._getClosestRoom(state,unsearchedRooms,self._currentDoor))[0]
                        self._doormat = state.get_room(self._getClosestRoom(state, unsearchedRooms,self._currentDoor))[-1]['doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3,5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM==self._phase:
                self._navigator.reset_full()
                if self._goalVic and self._goalVic in self._foundVictims and 'location' not in self._foundVictimLocs[self._goalVic].keys():
                    self._door = state.get_room_doors(self._foundVictimLocs[self._goalVic]['room'])[0]
                    self._doormat = state.get_room(self._foundVictimLocs[self._goalVic]['room'])[-1]['doormat']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    #doorLoc = self._door['location']
                    doorLoc = self._doormat
                else:
                    #doorLoc = self._door['location']
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3,5)
                    doorLoc = self._doormat
                self._navigator.add_waypoints([doorLoc])
                self._tick = state['World']['nr_ticks']
                self._phase=Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM==self._phase:
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._goalVic and self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                # check below
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    if self._goalVic in self._foundVictims and str(self._door['room_name']) == self._foundVictimLocs[self._goalVic]['room'] and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goalVic+'.', 'Brutus')                 
                    if self._goalVic not in self._foundVictims and not self._remove or not self._goalVic and not self._remove:
                        self._sendMessage('Moving to ' + str(self._door['room_name']) + ' because it is the closest not explored area.', 'Brutus')                   
                    self._currentDoor=self._door['location']
                    #self._currentDoor=self._doormat
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        #for info in state.values():
                            #if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9,7),(9,19),(21,19)]:
                            #    self._sendMessage('Found stones blocking my path to ' + str(self._door['room_name']) + '. We can remove them faster if you help me. If you will come here press the "Yes" button, if not press "No".', 'Brutus')
                            #    if self.received_messages_content and self.received_messages_content[-1]=='Yes':
                            #        return None, {}
                            #    if self.received_messages_content and self.received_messages_content[-1]=='No' or state['World']['nr_ticks'] > self._tick + 579:
                            #        self._sendMessage('Removing the stones blocking the path to ' + str(self._door['room_name']) + ' because I want to search this area. We can remove them faster if you help me', 'Brutus')
                                #return RemoveObject.__name__,{'object_id':info['obj_id'],'size':info['visualization']['size']}

                        return action,{}
                    #self._phase=Phase.PLAN_ROOM_SEARCH_PATH
                    self._phase=Phase.REMOVE_OBSTACLE_IF_NEEDED
                    #return Idle.__name__,{'duration_in_ticks':50}         

            if Phase.REMOVE_OBSTACLE_IF_NEEDED==self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                        objects.append(info)
                        if info['percentage_lel'] > 10 and self._hcn > 40 and self._co > 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500'
                        if info['percentage_lel'] > 10 and self._hcn > 40 and self._co <=500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm HCN > 40'
                        if info['percentage_lel'] > 10 and self._hcn <= 40 and self._co > 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm CO > 500'
                        if info['percentage_lel'] > 10 and self._hcn <= 40 and self._co <= 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - explosion danger > 10% LEL'
                        if info['percentage_lel'] <= 10 and self._hcn > 40 and self._co > 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500'
                        if info['percentage_lel'] <= 10 and self._hcn <= 40 and self._co > 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - toxic concentrations: ppm CO > 500'
                        if info['percentage_lel'] <= 10 and self._hcn > 40 and self._co <= 500:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - toxic concentrations: ppm HCN > 40'

                        if info['percentage_lel'] <= 10 and self._hcn <= 40 and self._co <= 500 and int(info['visualization']['size']*7.5)>15:
                            advice = '\n \n I suggest to call in a "Fire fighter" to help extinguishing because: \n - explosion danger < 11% LEL \n - toxic concentrations: ppm HCN < 41 and ppm CO < 501 \n - by myself extinguish time > 15 seconds'
                        if info['percentage_lel'] <= 10 and self._hcn <= 40 and self._co <= 500 and int(info['visualization']['size']*7.5)<=15:
                            advice = '\n \n I suggest to "Extinguish" alone because: \n - explosion danger < 11% LEL \n - toxic concentrations: ppm HCN < 41 and ppm CO < 501 \n - by myself extinguish time < 16 seconds'

                        self._sendMessage('fire is blocking ' + str(self._door['room_name'])+'. \n \n Please decide whether to "Extinguish", "Continue" exploring, or call in a "Fire fighter" to help extinguishing. \n \n \
                            Important features to consider: \n - Pinned victims located: ' + str(self._collectedVictims) + '\n - fire temperature: ' + str(int(info['visualization']['size']*300)) + ' degrees Celcius \
                            \n - explosion danger: ' + str(info['percentage_lel']) + '% LEL \n - by myself extinguish time: ' + str(int(info['visualization']['size']*7.5)) + ' seconds \n - with help extinguish time: \
                            ' + str(int(info['visualization']['size']*3.75)) + ' seconds \n - toxic concentrations: ' + str(self._hcn) + ' ppm HCN and ' + str(self._co) + ' ppm CO' + advice,'Brutus')
                        self._waiting = True
                        if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase=Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Extinguish':
                            self._sendMessage('Extinguishing fire blocking ' + str(self._door['room_name']) + ' alone.','Brutus')
                            self._phase = Phase.ENTER_ROOM
                            return RemoveObject.__name__, {'object_id': info['obj_id'],'size':info['visualization']['size']}
                        if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                            self._sendMessage('Extinguishing fire blocking ' + str(self._door['room_name']) + ' together with fire fighter.','Brutus')
                            self._phase = Phase.BACKUP
                            return Backup.__name__,{'size':info['percentage_lel']}
                        else:
                            return None,{}

                    #if 'class_inheritance' in info and 'Smoke' in info['class_inheritance']:
                    #    self._sendMessage('Smoke detected with ' + info['co_pmm'] + ' ppm CO and ' + info['hcn_ppm'] + ' ppm HCN.','Brutus')
                    #    self._phase=Phase.FIND_NEXT_GOAL

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'iron' in info['obj_id']:
                        objects.append(info)
                        if self._hcn > 40 and self._co > 500 and int(info['weight']/10)>15:
                            advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500 \n - by myself removal time > 15 seconds'
                        if self._hcn > 40 and self._co <= 500 and int(info['weight']/10)>15:
                            advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm HCN > 40 \n - by myself removal time > 15 seconds'
                        if self._hcn <= 40 and self._co > 500 and int(info['weight']/10)>15:
                            advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm CO > 500 \n - by myself removal time > 15 seconds'

                        if self._hcn > 40 and self._co > 500 and int(info['weight']/10)<=15:
                            advice = '\n \n I suggest to "Remove" iron debris alone because: \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500 \n - by myself removal time < 16 seconds'
                        if self._hcn > 40 and self._co <= 500 and int(info['weight']/10)<=15:
                            advice = '\n \n I suggest to "Remove" iron debris alone because: \n - toxic concentrations: ppm HCN > 40 \n - by myself removal time < 16 seconds'
                        if self._hcn <= 40 and self._co > 500 and int(info['weight']/10)<=15:
                            advice = '\n \n I suggest to "Remove" iron debris alone because: \n - toxic concentrations: ppm CO > 500 \n - by myself removal time < 16 seconds'

                        if self._hcn <= 40 and self._co <= 500 and int(info['weight']) > 150:
                            advice='\n \n I suggest to call in a "Fire fighter" to help remove iron debris because: \n - toxic concentrations: ppm CO < 501 and ppm CO < 41 \n - iron debris weight > 150 kilograms'
                        if self._hcn <= 40 and self._co <= 500 and int(info['weight']) <= 150:
                            advice='\n \n I suggest to "Remove" iron debris alone because: \n - toxic concentrations: ppm CO < 501 and ppm CO < 41 \n - iron debris weight < 151 kilograms'
                        self._sendMessage('Iron debris is blocking ' + str(self._door['room_name'])+'. \n \n Please decide whether to "Remove" alone, "Continue" exploring, or call in a "Fire fighter" to help remove. \n \n \
                            Important features to consider: \n - Pinned victims located: ' + str(self._collectedVictims) + ' \n - Iron debris weight: ' + str(int(info['weight'])) + ' kilograms \n - by myself removal time: ' \
                            + str(int(info['weight']/10)) + ' seconds \n - with help removal time: ' + str(int(info['weight']/20)) + ' seconds \n - toxic concentrations: ' + str(self._hcn) + ' ppm HCN and ' + str(self._co) + ' ppm CO ' + 
                            advice,'Brutus')
                        self._waiting = True
                        if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                            self._waiting = False
                            self._tosearch.append(self._door['room_name'])
                            self._phase=Phase.FIND_NEXT_GOAL
                        if self.received_messages_content and self.received_messages_content[-1] == 'Remove':
                            self._sendMessage('Removing iron debris blocking ' + str(self._door['room_name']) + ' alone.','Brutus')
                            self._phase = Phase.ENTER_ROOM
                            return RemoveObject.__name__, {'object_id': info['obj_id'],'size':info['visualization']['size']}
                        if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                            self._sendMessage('Removing iron debris blocking ' + str(self._door['room_name']) + ' together with fire fighter.','Brutus')
                            self._phase = Phase.BACKUP
                            return Backup.__name__,{}
                        else:
                            return None,{}

                if len(objects)==0:                    
                    #self._sendMessage('No need to clear the entrance of ' + str(self._door['room_name']) + ' because it is not blocked by obstacles.','Brutus')
                    self._answered = False
                    self._remove = False
                    self._phase = Phase.ENTER_ROOM


            if Phase.BACKUP==self._phase:
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                        if info['percentage_lel'] > 10 or self._co > 500 or self._hcn > 40:
                            self._sendMessage('fire at ' + str(self._door['room_name']) + ' too dangerous for fire fighter!! \n \n Going to abort extinguishing.','Brutus')
                            self._searchedRooms.append(self._door['room_name'])
                            self._sendMessages = []
                            self.received_messages = []
                            self.received_messages_content = []
                            self._phase = Phase.FIND_NEXT_GOAL
                            return Injured.__name__,{'duration_in_ticks':50}
                        else:
                            self._phase = Phase.ENTER_ROOM
                            return RemoveObjectTogether.__name__, {'object_id': info['obj_id'], 'size':info['visualization']['size']}
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'iron' in info['obj_id']:
                        if self._co > 500 or self._hcn > 40:
                            self._sendMessage('Situation at ' + str(self._door['room_name']) + ' too dangerous for fire fighter!! \n \n Going to abort removing iron debris.','Brutus')
                            self._searchedRooms.append(self._door['room_name'])
                            self._sendMessages = []
                            self.received_messages = []
                            self.received_messages_content = []
                            self._phase = Phase.FIND_NEXT_GOAL
                            return Injured.__name__,{'duration_in_ticks':50}
                        else:
                            self._phase = Phase.ENTER_ROOM
                            return RemoveObjectTogether.__name__, {'object_id': info['obj_id'], 'size':info['visualization']['size']}

            if Phase.BACKUP2==self._phase:
                self._goalVic = None
                self._goalLoc = None
                zones = self._getDropZones(state)
                remainingZones = []
                remainingVics = []
                remaining = {}
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collectedVictims:
                        remainingZones.append(info)
                        remainingVics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remainingZones:
                    #self._goalVic = str(remainingZones[0]['img_name'])[8:-4]
                    #self._goalLoc = remainingZones[0]['location']
                    self._remainingZones = remainingZones
                    self._remaining = remaining
                if not remainingZones:
                    return None,{}

                for vic in remainingVics:
                    if vic in self._foundVictims and vic not in self._todo:
                        self._goalVic = vic
                        self._goalLoc = remaining[vic]
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                        if info['percentage_lel'] > 10 or self._co > 500 or self._hcn > 40:
                            self._sendMessage('fire at ' + str(self._door['room_name']) + ' too dangerous for fire fighter!! \n \n Going to abort extinguishing.','Brutus')
                            self._searchedRooms.append(self._door['room_name'])
                            self._collectedVictims.append(self._goalVic)
                            self._sendMessages = []
                            self.received_messages = []
                            self.received_messages_content = []
                            self._phase = Phase.FIND_NEXT_GOAL
                            return Injured.__name__,{'duration_in_ticks':50}
                        else:
                            self._decided = True
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return RemoveObjectTogether.__name__, {'object_id': info['obj_id'], 'size':info['visualization']['size']}
                    
            if Phase.ENTER_ROOM==self._phase:
                self._answered = False
                if self._goalVic in self._collectedVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._goalVic in self._foundVictims and self._door['room_name']!=self._foundVictimLocs[self._goalVic]['room']:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                if self._door['room_name'] in self._searchedRooms and self._goalVic not in self._foundVictims:
                    self._currentDoor=None
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)                 
                    #self._currentDoor=self._door['location']
                    #self._currentDoor=self._door
                    action = self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    self._phase=Phase.PLAN_ROOM_SEARCH_PATH
                    #self._phase=Phase.REMOVE_OBSTACLE_IF_NEEDED
                    #return Idle.__name__,{'duration_in_ticks':50} 


            if Phase.PLAN_ROOM_SEARCH_PATH==self._phase:
                self._agentLoc = int(self._door['room_name'].split()[-1])
                roomTiles = [info['location'] for info in state.values()
                    if 'class_inheritance' in info 
                    and 'AreaTile' in info['class_inheritance']
                    and 'room_name' in info
                    and info['room_name'] == self._door['room_name']
                ]
                self._roomtiles=roomTiles               
                self._navigator.reset_full()
                self._navigator.add_waypoints(roomTiles)
                #self._sendMessage('Searching through whole ' + str(self._door['room_name']) + ' because my sense range is limited and to find victims.', 'Brutus')
                #self._currentDoor = self._door['location']
                self._roomVics=[]
                self._phase=Phase.FOLLOW_ROOM_SEARCH_PATH
                #return Idle.__name__,{'duration_in_ticks':50}

            if Phase.FOLLOW_ROOM_SEARCH_PATH==self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action!=None:                   
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            if vic not in self._roomVics:
                                self._roomVics.append(vic)

                            if vic in self._foundVictims and 'location' not in self._foundVictimLocs[vic].keys():
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                if vic == self._goalVic:
                                    self._sendMessage('Found '+ vic + ' in ' + self._door['room_name'] + ' because you told me '+vic+ ' was located here.', 'Brutus')
                                    self._searchedRooms.append(self._door['room_name'])
                                    self._phase=Phase.FIND_NEXT_GOAL

                            if 'healthy' not in vic and vic not in self._foundVictims:
                                self._recentVic = vic
                                self._foundVictims.append(vic)
                                self._foundVictimLocs[vic] = {'location':info['location'],'room':self._door['room_name'],'obj_id':info['obj_id']}
                                self._sendMessage('Found ' + vic + ' in ' + self._door['room_name'] + '.','Brutus')

                        if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'fire' in info['obj_id']:
                            if info['percentage_lel'] > 10 and self._hcn > 40 and self._co > 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500'
                            if info['percentage_lel'] > 10 and self._hcn > 40 and self._co <=500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm HCN > 40'
                            if info['percentage_lel'] > 10 and self._hcn <= 40 and self._co > 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - explosion danger > 10% LEL \n - toxic concentrations: ppm CO > 500'
                            if info['percentage_lel'] > 10 and self._hcn <= 40 and self._co <= 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - explosion danger > 10% LEL'
                            if info['percentage_lel'] <= 10 and self._hcn > 40 and self._co > 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm HCN > 40 and ppm CO > 500'
                            if info['percentage_lel'] <= 10 and self._hcn <= 40 and self._co > 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm CO > 500'
                            if info['percentage_lel'] <= 10 and self._hcn > 40 and self._co <= 500:
                                advice = '\n \n I suggest to "Continue" exploring because: \n - toxic concentrations: ppm HCN > 40'

                            if info['percentage_lel'] <= 10 and self._hcn <= 40 and self._co <= 500:
                                advice = '\n \n I suggest to call in a "Fire fighter" to help extinguishing and rescue ' + self._recentVic + ' because: \n - explosion danger < 11% LEL \n - toxic concentrations: ppm HCN < 41 and ppm CO < 501'
                            
                            self._sendMessage('Detected fire in ' + self._door['room_name'] + '. \n \n Please decide whether to call in a "Fire fighter" to help extinguishing and rescue ' + self._recentVic + ', or "Continue" exploring because it is not safe enough to send in fire fighter. \n \n \
                            Important features to consider: \n - Pinned victims located: ' + str(self._collectedVictims) + '\n - fire temperature: ' + str(int(info['visualization']['size']*300)) + ' degrees Celcius \
                            \n - explosion danger: ' + str(info['percentage_lel']) + '% LEL \n - toxic concentrations: ' + str(self._hcn) + ' ppm HCN and ' + str(self._co) + ' ppm CO' + advice,'Brutus')
                            self._waiting = True
                            if self.received_messages_content and self.received_messages_content[-1]=='Continue':
                                self._waiting = False
                                self._tosearch.append(self._door['room_name'])
                                self._phase=Phase.FIND_NEXT_GOAL
                            #if self.received_messages_content and self.received_messages_content[-1] == 'Remove':
                            #    self._sendMessage('Extinguishing fire blocking ' + str(self._door['room_name']) + '.','Brutus')
                            #    return RemoveObject.__name__, {'object_id': info['obj_id']}
                            if self.received_messages_content and self.received_messages_content[-1] == 'Fire fighter':
                                self._sendMessage('Extinguishing fire in ' + str(self._door['room_name']) + ' together with fire fighter.','Brutus')
                                #self._goalVic = self._recentVic
                                #self._goalLoc = self._foundVictimLocs[self._goalVic]['location']
                                self._phase = Phase.BACKUP2
                                return Backup.__name__,{'size':info['percentage_lel']}
                            else:
                                return None, {}

                    return action,{}
                #if self._goalVic not in self._foundVictims:
                #    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic, 'Brutus')
                if self._goalVic in self._foundVictims and self._goalVic not in self._roomVics and self._foundVictimLocs[self._goalVic]['room']==self._door['room_name']:
                    self._sendMessage(self._goalVic + ' not present in ' + str(self._door['room_name']) + ' because I searched the whole area without finding ' + self._goalVic+'.', 'Brutus')
                    self._foundVictimLocs.pop(self._goalVic, None)
                    self._foundVictims.remove(self._goalVic)
                    self._roomVics = []
                    self.received_messages = []
                    self.received_messages_content = []
                self._searchedRooms.append(self._door['room_name'])
                self._phase=Phase.FIND_NEXT_GOAL
                return Idle.__name__,{'duration_in_ticks':25}
                
            if Phase.PLAN_PATH_TO_VICTIM==self._phase:
                if 'mild' in self._goalVic:
                    self._sendMessage('fire fighter will transport ' + self._goalVic + ' in ' + self._foundVictimLocs[self._goalVic]['room'] + ' to safe zone.', 'Brutus')
                self._searchedRooms.append(self._door['room_name'])
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._foundVictimLocs[self._goalVic]['location']])
                self._phase=Phase.FOLLOW_PATH_TO_VICTIM
                #return Idle.__name__,{'duration_in_ticks':50}
                    
            if Phase.FOLLOW_PATH_TO_VICTIM==self._phase:
                if self._goalVic and self._goalVic in self._collectedVictims:
                    self._phase=Phase.FIND_NEXT_GOAL
                else:
                    self._state_tracker.update(state)
                    action=self._navigator.get_move_action(self._state_tracker)
                    if action!=None:
                        return action,{}
                    #if action==None and 'critical' in self._goalVic:
                    #    return MoveNorth.__name__, {}
                    self._phase=Phase.TAKE_VICTIM
                    
            if Phase.TAKE_VICTIM==self._phase:
                objects=[]
                for info in state.values():
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        #self._sendMessage('Please come to ' + str(self._door['room_name']) + ' because we need to carry ' + str(self._goalVic) + ' together.', 'Brutus')
                        self._collectedVictims.append(self._goalVic)
                        self._phase=Phase.INTRO4
                        if not 'Human' in info['name']:
                            return None, {} 
                if len(objects)==0 and 'critical' in self._goalVic:
                    self._criticalRescued+=1
                    self._collectedVictims.append(self._goalVic)
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                if 'mild' in self._goalVic:
                    self._phase=Phase.PLAN_PATH_TO_DROPPOINT
                    self._collectedVictims.append(self._goalVic)
                    return CarryObject.__name__,{'object_id':self._foundVictimLocs[self._goalVic]['obj_id']}                

            if Phase.PLAN_PATH_TO_DROPPOINT==self._phase:
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._goalLoc])
                self._phase=Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT==self._phase:
                #self._sendMessage('Transporting '+ self._goalVic + ' to the drop zone because ' + self._goalVic + ' should be delivered there for further treatment.', 'Brutus')
                self._state_tracker.update(state)
                action=self._navigator.get_move_action(self._state_tracker)
                if action!=None:
                    return action,{}
                self._phase=Phase.DROP_VICTIM
                #return Idle.__name__,{'duration_in_ticks':50}  

            if Phase.DROP_VICTIM == self._phase:
                if 'mild' in self._goalVic:
                    self._sendMessage('fire fighter delivered '+ self._goalVic + ' at the safe zone.', 'Brutus')
                self._phase=Phase.FIND_NEXT_GOAL
                self._currentDoor = None
                self._tick = state['World']['nr_ticks']
                return Drop.__name__,{}

            
    def _getDropZones(self,state:State):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        the place that requires the first drop)
        '''
        places=state[{'is_goal_block':True}]
        places.sort(key=lambda info:info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr']==0:
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
        #areas = ['area A1','area A2','area A3','area A4','area B1','area B2','area C1','area C2','area C3']
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                if msg.startswith("Search:"):
                    area = 'area '+ msg.split()[-1]
                    if area not in self._searchedRooms:
                        self._searchedRooms.append(area)
                if msg.startswith("Found:"):
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5]) 
                    loc = 'area '+ msg.split()[-1]
                    if loc not in self._searchedRooms:
                        self._searchedRooms.append(loc)
                    if foundVic not in self._foundVictims:
                        self._foundVictims.append(foundVic)
                        self._foundVictimLocs[foundVic] = {'room':loc}
                    if foundVic in self._foundVictims and self._foundVictimLocs[foundVic]['room'] != loc:
                        self._foundVictimLocs[foundVic] = {'room':loc}
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
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    if collectVic in self._foundVictims and self._foundVictimLocs[collectVic]['room'] != loc:
                        self._foundVictimLocs[collectVic] = {'room':loc}
                    if collectVic not in self._collectedVictims:
                        self._collectedVictims.append(collectVic)
                if msg.startswith('Remove:'):
                    # add sending messages about it
                    area = 'area ' + msg.split()[-1]
                    self._door = state.get_room_doors(area)[0]
                    self._doormat = state.get_room(area)[-1]['doormat']
                    if area in self._searchedRooms:
                        self._searchedRooms.remove(area)
                    self.received_messages = []
                    self.received_messages_content = []
                    self._remove = True
                    self._sendMessage('Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.', 'Brutus')  
                    self._phase = Phase.PLAN_PATH_TO_ROOM
            if mssgs and mssgs[-1].split()[-1] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']:
                self._humanLoc = int(mssgs[-1].split()[-1])

            #if msg.startswith('Mission'):
            #    self._sendMessage('Unsearched areas: '  + ', '.join([i.split()[1] for i in areas if i not in self._searchedRooms]) + '. Collected victims: ' + ', '.join(self._collectedVictims) +
            #    '. Found victims: ' +  ', '.join([i + ' in ' + self._foundVictimLocs[i]['room'] for i in self._foundVictimLocs]) ,'Brutus')
            #    self.received_messages=[]

    def _trustBlief(self, member, received):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''
        default = 0.5
        trustBeliefs = {}
        for member in received.keys():
            trustBeliefs[member] = default
        for member in received.keys():
            for message in received[member]:
                if 'Found' in message and 'colour' not in message:
                    trustBeliefs[member]-=0.1
                    break
        return trustBeliefs

    def _sendMessage(self, mssg, sender):
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'score' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        if 'score' in msg.content:
            self.send_message(msg)

        #if self.received_messages and self._sendMessages:
        #    self._last_mssg = self._sendMessages[-1]
        #    if self._last_mssg.startswith('Searching') or self._last_mssg.startswith('Moving'):
        #        self.received_messages=[]
        #        self.received_messages.append(self._last_mssg)

    def _getClosestRoom(self, state, objs, currentDoor):
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj]=state.get_room_doors(obj)[0]['location']
        dists = {}
        for room,loc in locs.items():
            if currentDoor!=None:
                dists[room]=utils.get_distance(currentDoor,loc)
            if currentDoor==None:
                dists[room]=utils.get_distance(agent_location,loc)

        return min(dists,key=dists.get)

    def _efficientSearch(self, tiles):
        x=[]
        y=[]
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i%2==0:
                locs.append((x[i],min(y)))
            else:
                locs.append((x[i],max(y)))
        return locs

    def _dynamicMessage(self, mssg1, mssg2, explanation, sender):
        if explanation not in self._providedExplanations:
            self._sendMessage(mssg1,sender)
            self._providedExplanations.append(explanation)
        if 'Searching' in mssg1:
            #history = ['Searching' in mssg for mssg in self._sendMessages]
            if explanation in self._providedExplanations and mssg1 not in self._sendMessages[-5:]:
                self._sendMessage(mssg2,sender)   
        if 'Found' in mssg1:
            history = [mssg2[:-1] in mssg for mssg in self._sendMessages]
            if explanation in self._providedExplanations and True not in history:
                self._sendMessage(mssg2,sender)      
        if 'Searching' not in mssg1 and 'Found' not in mssg1:
            if explanation in self._providedExplanations and self._sendMessages[-1]!=mssg1:
                self._sendMessage(mssg2,sender) 
