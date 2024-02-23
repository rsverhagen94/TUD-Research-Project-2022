import sys, random, enum, ast, time, threading, os, math
import numpy as np
from brains1.BW4TBrain import BW4TBrain
from actions1.customActions import *
from matrx import utils
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.customActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop, AddObject

class TimerAgent(BW4TBrain):
    def __init__(self, slowdown):
        super().__init__(slowdown)
        # Initialization of some relevant variables
        self._slowdown = slowdown
        self._sendMessages = []
        self._score = 0

    def update_time(self):
        with self._counter_lock:
            self._counter_value -= 1
            self._duration += 1
            if self._counter_value < 0:
                self._counter_value = 90  # Reset the counter after reaching 0
                self._duration = 0

        self._sendMessage('Time left: ' + str(self._counter_value) + '.', 'RescueBot')
        self._sendMessage('Fire duration: ' + str(self._duration) + '.', 'RescueBot')

        # Schedule the next print
        threading.Timer(6, self.update_time).start()

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._counter_lock = threading.Lock()
        self._counter_value = 91
        self._duration = 14
        # Start the initial print
        self.update_time()

    def filter_bw4t_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def decide_on_bw4t_action(self, state):
        self._sendMessage('Our score is ' + str(state['brutus']['score']) +'.', 'Brutus')
        while True:
            return None, {}
        
    def _sendMessage(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content and 'Time left:' not in msg.content and 'Fire duration:' not in msg.content:
            self.send_message(msg)
            self._sendMessages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content or 'Time left:' in msg.content or 'Fire duration' in msg.content:
            self.send_message(msg)
            
        
def add_object(locs, image, size, opacity, name):
    action_kwargs = {}
    add_objects = []
    for loc in locs:
        obj_kwargs = {}
        obj_kwargs['location'] = loc
        obj_kwargs['img_name'] = image
        obj_kwargs['visualize_size'] = size
        obj_kwargs['visualize_opacity'] = opacity
        obj_kwargs['name'] = name
        add_objects+=[obj_kwargs]
    action_kwargs['add_objects'] = add_objects
    return action_kwargs