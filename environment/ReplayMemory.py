import numpy as np
import pandas as pd
import tensorflow as tf

import random
import math
import os
class ReplayMemory:


	# Initialize
	def __init__(self, size_board, mem_threshold, gamma):
		self.mem_threshold = mem_threshold
		self.size_board = size_board
		self.num_states = self.size_board * self.size_board
		self.gamma = gamma
		
		self.initial_state = np.empty((self.mem_threshold, self.num_states), dtype = np.uint8)
		self.actions = np.zeros(self.mem_threshold, dtype = np.uint8)
		self.next_state = np.empty((self.mem_threshold, self.num_states), dtype = np.uint8)
		self.gameOver = np.empty(self.mem_threshold, dtype = np.bool)
		self.rewards = np.empty(self.mem_threshold, dtype = np.int8)
		self.count = 0
		self.current = 0




	# Rember the result

	def remember(self, currentState, action, reward, next_state, gameOver):
		self.actions[self.current] = action
		self.rewards[self.current] = reward
		self.initial_state[self.current, ...] = currentState
		self.next_state[self.current, ...] = next_state
		self.gameOver[self.current] = gameOver
		self.count = max(self.count, self.current + 1)
		self.current = (self.current + 1) % self.mem_threshold




	# Get batch

	def getBatch(self, model, batch, num_actions, num_states, sess, X):
		memoryLength = self.count
		chosenbatch = min(batch, memoryLength)
		
		inputs = np.zeros((chosenbatch, num_states))
		targets = np.zeros((chosenbatch, num_actions))

		for i in range(chosenbatch):
			randomIndex = random.randrange(0, memoryLength)
			current_initial_state = np.reshape(self.initial_state[randomIndex], (1, num_states))

			target = sess.run(model, feed_dict = {X: current_initial_state})

			current_next_state = np.reshape(self.next_state[randomIndex], (1, num_states))
			current_outputs = sess.run(model, feed_dict = {X: current_next_state})

			next_stateMaxQ = np.amax(current_outputs)

			if( next_stateMaxQ > 1 ):
				next_stateMaxQ = 1
			
			if( self.gameOver[randomIndex] == True ):
				target[0, [self.actions[randomIndex]]] = self.rewards[randomIndex]
			else:
				target[0, [self.actions[randomIndex]]] = self.rewards[randomIndex] + self.gamma * next_stateMaxQ

			inputs[i] = current_initial_state
			targets[i] = target

		return inputs, targets