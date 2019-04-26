# -*- coding: utf-8 -*-



from GomokuTrain import GomokuEnvironment, X, W1, b1, input_layer, W2, b2, hidden_layer, W3, b3, output_layer, Y, cost, optimizer
import tensorflow as tf
import numpy as np
import random
import math
import os
import sys
import time


xrange = range
#------------------------------------------------------------
# Setting Variables
#------------------------------------------------------------
STONE_NONE = 0
STONE_PLAYER1 = 1
STONE_PLAYER2 = 2

gridSize = 10
#------------------------------------------------------------



#------------------------------------------------------------
# Printing the board
#------------------------------------------------------------
def showBoard(env):
	for y in xrange(gridSize):
		for x in xrange(gridSize):
			if( env.state[y * gridSize + x] == STONE_PLAYER1 ):
				sys.stdout.write('O')
			elif( env.state[y * gridSize + x] == STONE_PLAYER2 ):
				sys.stdout.write('X')
			else:
				sys.stdout.write('.')
		sys.stdout.write('\n')
	sys.stdout.write('\n')


#------------------------------------------------------------



#------------------------------------------------------------
# Play game
#------------------------------------------------------------
def playGame(env, sess):

	env.reset()

	gameOver = False
	currentPlayer = STONE_PLAYER1
	
	while( gameOver != True ):
		action = - 9999
		
		if( currentPlayer == STONE_PLAYER1 ):
			currentState = env.getState()
		else:
			currentState = env.getStateInverse()

		action = env.getAction(sess, currentState)
		nextState, reward, gameOver = env.act(currentPlayer, action)
		
		showBoard(env)
		time.sleep(3)
		
		if( currentPlayer == STONE_PLAYER1 ):
			currentPlayer = STONE_PLAYER2
		else:
			currentPlayer = STONE_PLAYER1
#------------------------------------------------------------



#------------------------------------------------------------
# Main function to run
#------------------------------------------------------------
def main(_):

	# Instantiate the environment
	env = GomokuEnvironment(gridSize)

	# Initialize tensorflow
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# Saver
	saver = tf.train.Saver()

	# Load the model
	if( os.path.isfile(os.getcwd() + "/GomokuModel.ckpt.index") == True ):
		saver.restore(sess, os.getcwd() + "/GomokuModel.ckpt")
		print('saved model is loaded!')
	
	# Play the game
	playGame(env, sess)
	
	# close session
	sess.close()
#------------------------------------------------------------



#------------------------------------------------------------
# Run
#------------------------------------------------------------
if __name__ == '__main__':
	tf.app.run()
#------------------------------------------------------------

