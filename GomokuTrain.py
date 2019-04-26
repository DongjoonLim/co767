import numpy as np
import pandas as pd
import tensorflow as tf

from environment.ReplayMemory import ReplayMemory
from environment.GomokuEnvironment import GomokuEnvironment
import random
import math
import os

# set global variables
size_board = 15

num_actions = size_board * size_board
num_states = size_board * size_board
hidden_units = 1000
mem_threshold = 50000
batch = 50
num_epoch = 100
epsilon_disRate = 0.999
min_epsilon = 0.1
gamma = 0.9
learning_rate = 0.2
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

# Play game

def playGame(env, memory, sess, saver, epsilon, iteration):

	# Repeat playing
	winCount = 0

	for i in range(num_epoch):
		env.reset()

		err = 0
		gameOver = False
		currentPlayer = BlackStone
		
		while( gameOver != True ):
			
			# Act
			
			action = - 9999
			
			if( currentPlayer == BlackStone ):
				currentState = env.getState()
			else:
				currentState = env.getStateInverse()

			if( (float(random.randrange(0, 9999)) / 10000) <= epsilon ):
				action = env.getActionRandom()
			else:
				action = env.getAction(sess, currentState)

			if( epsilon > min_epsilon ):
				epsilon = epsilon * epsilon_disRate
			
			next_state, reward, gameOver = env.makeMove(currentPlayer, action)

			if( reward == 1 and currentPlayer == BlackStone ):
				winCount = winCount + 1

			
			# Learning
		
			memory.remember(currentState, action, reward, next_state, gameOver)

			inputs, targets = memory.getBatch(output_layer, batch, num_actions, num_states, sess, X)
			
			_, loss = sess.run([optimizer, cost], feed_dict = {X: inputs, Y: targets})
			err = err + loss
			
			if( currentPlayer == BlackStone ):
				currentPlayer = whiteStone
			else:
				currentPlayer = BlackStone

		print("num_epoch " + str(iteration) + str(i) + ": err = " + str(err) + ": Win count = " + str(winCount) +
				" Win ratio = " + str(float(winCount) / float(i + 1) * 100))

		print(targets)

		if( (i % 10 == 0) and (i != 0) ):
			save_path = saver.save(sess, os.getcwd() + "/GomokuModel.ckpt")
			print("Model saved in file: %s" % save_path)
	return float(winCount) / float(i + 1) * 100





# Main function

def main():

	print("Training new model")

	# Instantiate the environment
	env = GomokuEnvironment(size_board)

	# Instantiate replay memory
	memory = ReplayMemory(size_board, mem_threshold, gamma)

	# Initialize tensorflow
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Saver
	saver = tf.train.Saver()

	# Load model
	if( os.path.isfile(os.getcwd() + "/GomokuModel.ckpt.index") == True ):
		saver.restore(sess, os.getcwd() + "/GomokuModel.ckpt")
		print('Saved model is loaded!')
	
	# Playing the game
	iteration = 0
	winRateList = []
	for x in range(9000000):
		winRate = playGame(env, memory, sess, saver, 1, iteration)
		winRateList.append(winRate)
		print(winRateList)
		iteration += 1
		
		df = pd.DataFrame(winRateList, columns=["colummn"])
		df.to_csv('results/list2.csv', index=False)
	

	# close session
	sess.close()

main()