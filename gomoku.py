import gym
import logging
import numpy
import random
from gym import spaces
 

class Gomoku(gym.Env):
 
	def __init__(self):
		#size
		self.SIZE = 8
		#-1 is black 1 is white
		self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
		self.viewer = None
		self.step_count = 0
 
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
 
 
	def is_valid_coord(self,x,y):
		return x>=0 and x<self.SIZE and y>=0 and y<self.SIZE
 
	def is_valid_set_coord(self,x,y):
		return self.is_valid_coord(x,y) and self.chessboard[x][y]==0
 
	#get a valid position
	def get_valid_pos_weights(self):
		results = []
		for x in range(self.SIZE):
			for y in range(self.SIZE):
				if self.chessboard[x][y]==0:
					results.append(1)
				else:
					results.append(0)
		return results
 
	#action position and color
	#  [1,3,1] is state [1, 3] and white
	#return next state，reward，whether or not to end，extra info{}
	def step(self, action):
		'''
		if not self.is_valid_set_coord(action[0],action[1]):
			return self.chessboard,-50,False,{}
		'''

		self.chessboard[action[0]][action[1]] = action[2]
 
		self.step_count +=1
 
		#win or not
		color = action[2]
		
		win_reward = 1000
		common_reward = -20
		draw_reward = 0
 
		#1.horizontal dir
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]+i
			y = action[1]
			#left
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#right
			x = action[0]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#win if more than 5 connected
			if count>=5:
				win = True
				break
 
			#stop explorating if not the same 
			if stop0 and stop1:
				break
			i+=1
 
		if win:
			#print('win1')
			return self.chessboard,win_reward,True,{}
		
		#2.vertical
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]
			y = action[1]+i
			#left
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			
			#right
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#win if more than 5
			if count>=5:
				win = True
				break
 
			#stop exploring if not the same for two
			if stop0 and stop1:
				break
			i+=1
		if win:
			#print('win2')
			return self.chessboard,win_reward,True,{}
		
		#3. left diag
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]+i
			y = action[1]+i
			
			#left
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			
			#right
			x = action[0]-i
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#win if more than 5 connected
			if count>=5:
				win = True
				break
 
			#stop exploring if not the same for two
			if stop0 and stop1:
				break
			i+=1
		if win:
			#print('win3')
			return self.chessboard,win_reward,True,{}
 
		#3.right diag
		count = 1
		win = False
 
		i = 1
		stop0 = False
		stop1 = False
 
		while i<self.SIZE:
			x = action[0]-i
			y = action[1]+i
			#left
			if (not stop0) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop0 = True
			#right
			x = action[0]+i
			y = action[1]-i
			if (not stop1) and self.is_valid_coord(x,y) and self.chessboard[x][y] == color:
				count = count+1
			else:
				stop1 = True
 
			#win if more than 5 connected
			if count>=5:
				win = True
				break
 
			#stop exploring if not the same for two
			if stop0 and stop1:
				break
			i+=1
		if win:
			#print('win4')
			return self.chessboard,win_reward,True,{}
 
		if self.step_count == self.SIZE*self.SIZE:
			#print('draw')
			return self.chessboard,draw_reward,True,{}
 
		return self.chessboard,common_reward,False,{}
 
	def reset(self):
		self.chessboard = [ [  0 for v in range(self.SIZE)  ] for v in range(self.SIZE) ]
		self.step_count = 0
		return self.chessboard