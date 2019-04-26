import numpy as np
import pandas as pd
import tensorflow as tf
import math
import os
import keras
import random
size_board = 15
num_actions = 225
num_states = 225
hidden_units = 1000
mem_threshold = 50000
batch = 100
num_epoch = 200
epsilon_disRate = 0.999
min_epsilon = 0.1
gamma = 0.99
learning_rate = 0.15
winning_reward = 1

# Set the model
X = tf.placeholder(tf.float32, [None, num_states])
weight1 = tf.Variable(tf.truncated_normal([num_states, hidden_units], stddev = 1.0 / math.sqrt(float(num_states))))
bias1 = tf.Variable(tf.truncated_normal([hidden_units], stddev = 0.01))
input_layer = tf.nn.relu(tf.matmul(X, weight1) + bias1)

weight2 = tf.Variable(tf.truncated_normal([hidden_units, hidden_units], stddev = 1.0 / math.sqrt(float(hidden_units))))
bias2 = tf.Variable(tf.truncated_normal([hidden_units], stddev = 0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, weight2) + bias2)

weight3 = tf.Variable(tf.truncated_normal([hidden_units, hidden_units], stddev = 1.0 / math.sqrt(float(hidden_units))))
bias3 = tf.Variable(tf.truncated_normal([hidden_units], stddev = 0.01))
hidden_layer2 = tf.nn.relu(tf.matmul(hidden_layer, weight3) + bias3)

weight4 = tf.Variable(tf.truncated_normal([hidden_units, num_actions], stddev = 1.0 / math.sqrt(float(hidden_units))))
bias4 = tf.Variable(tf.truncated_normal([num_actions], stddev = 0.01))
output_layer = tf.matmul(hidden_layer2, weight4) + bias4

Y = tf.placeholder(tf.float32, [None, num_actions])
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batch)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

empty = 0
BlackStone = 1
whiteStone = 2
class GomokuEnvironmentReward():


	# Initialize

	def __init__(self, size_board):
		self.size_board = size_board
		self.num_states = self.size_board * self.size_board
		self.state = np.zeros(self.num_states, dtype = np.uint8)


	# Reset

	def reset(self):
		self.state = np.zeros(self.num_states, dtype = np.uint8)


	# Get the current state

	def getState(self):
		return np.reshape(self.state, (1, self.num_states))


	# Reverse player and get the current state

	def getStateInverse(self):
		tempState = self.state.copy()
		
		for i in range(self.num_states):
			if( tempState[i] == BlackStone ):
				tempState[i] = whiteStone
			elif( tempState[i] == whiteStone ):
				tempState[i] = BlackStone
		
		return np.reshape(tempState, (1, self.num_states))



	# Get the reward
	def GetReward(self, player, action):
	
		# Check left
		if( action % self.size_board > 0 ):
			if( self.state[action - 1] == player ):
				return 0.01

		# Check right
		if( action % self.size_board < self.size_board - 1 ):
			if( self.state[action + 1] == player ):
				return 0.01

		# Check Up
		if( action - self.size_board >= 0 ):
			if( self.state[action - self.size_board] == player ):
				return 0.01

		# Check Down
		if( action + self.size_board < self.num_states ):
			if( self.state[action + self.size_board] == player ):
				return 0.01

		# Check upper left
		if( (action % self.size_board > 0) and (action - self.size_board >= 0) ):
			if( self.state[action - 1 - self.size_board] == player ):
				return 0.01

		# Check upper right
		if( (action % self.size_board < self.size_board - 1) and (action - self.size_board >= 0) ):
			if( self.state[action + 1 - self.size_board] == player ):
				return 0.01

		# Check lower left
		if( (action % self.size_board > 0) and (action + self.size_board < self.num_states) ):
			if( self.state[action - 1 + self.size_board] == player ):
				return 0.01

		# Check lower right
		if( (action % self.size_board < self.size_board - 1) and (action + self.size_board < self.num_states) ):
			if( self.state[action + 1 + self.size_board] == player ):
				return 0.01
		
		return 0

	# Find Consecutive

	def CheckConsecutive(self, player):
		for y in range(self.size_board):
			for x in range(self.size_board):
			
				# Check right
				
				Consecutive = 0
				
				for i in range(5):
					if( x + i >= self.size_board ):
						break
	
					if( self.state[y * self.size_board + x + i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

				
				# Check left
				
				Consecutive = 0
				
				for i in range(5):
					if( x - i >= self.size_board ):
						break
	
					if( self.state[y * self.size_board + x - i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

				
				# check down
				
				Consecutive = 0
				
				for i in range(5):
					if( y + i >= self.size_board ):
						break
	
					if( self.state[(y + i) * self.size_board + x] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

				
				# check up
				
				Consecutive = 0
				
				for i in range(5):
					if( y - i >= self.size_board ):
						break
	
					if( self.state[(y - i) * self.size_board + x] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

				
				# check lower right diagonal
				
				Consecutive = 0
				
				for i in range(5):
					if( (x + i >= self.size_board) or (y + i >= self.size_board) ):
						break
	
					if( self.state[(y + i) * self.size_board + x + i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

				
				# check upper left diagonal
			
				Consecutive = 0
				
				for i in range(5):
					if( (x - i >= self.size_board) or (y - i >= self.size_board) ):
						break
	
					if( self.state[(y - i) * self.size_board + x - i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True

			
				# check lower left diagonal
				Consecutive = 0
				
				for i in range(5):
					if( (x - i < 0) or (y + i >= self.size_board) ):
						break
	
					if( self.state[(y + i) * self.size_board + x - i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True
				
				# check upper right diagonal
			
				Consecutive = 0
				
				for i in range(5):
					if( (x + i < 0) or (y - i >= self.size_board) ):
						break
	
					if( self.state[(y - i) * self.size_board + x + i] == player ):
						Consecutive += 1
					else:
						break

					if( Consecutive >= 5 ):
						return True
	
		return False




	# Check if game is over

	def isGameOver(self, player):
		if( self.CheckConsecutive(BlackStone) == True ):
			if( player == BlackStone ):
				return True, winning_reward
			else:
				return True, 0
		elif( self.CheckConsecutive(whiteStone) == True ):
			if( player == BlackStone ):
				return True, 0
			else:
				return True, winning_reward
		else:
			for i in range(self.num_states):
				if( self.state[i] == empty ):
					return False, 0
			return True, 0
			




	# Make move

	def makeMove(self, player, action):
		self.state[action] = player
		gameOver, reward = self.isGameOver(player)
		
		if( reward == 0 ):
			reward = self.GetReward(player, action)
		
		if( player == BlackStone ):
			next_state = self.getState()
		else:
			next_state = self.getStateInverse()
		
		return next_state, reward, gameOver



	# Getting the action
	def getAction(self, sess, currentState):
		q = sess.run(output_layer, feed_dict = {X: currentState})
		
		while( True ):
			action = q.argmax()

			if( self.state[action] == empty ):
				return action
			else:
				q[0, action] = -99999
	


	# Getting the action Randomly

	def getActionRandom(self):
		while( True ):
			action = random.randrange(0, size_board * size_board)

			if( self.state[action] == empty ):
				return action