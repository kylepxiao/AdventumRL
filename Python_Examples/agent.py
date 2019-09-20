from memory import ExperienceReplay
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import csv
import time
import datetime
import math
import random

def checkForBestMove(x,z,yaw):
	a=-1
	#print(yaw)
	if z<=6:
		if x < 12:
			#print("Segment1")
			if yaw==270:
				a=0
			if yaw==180:
				a=1
			if yaw==90:
				a=3
			if yaw==0:
				a=2
		elif x > 15:
			#print("Segment2")
			if yaw==90:
				a=0
			if yaw==180:
				a=2
			if yaw==0:
				a=1
			if yaw==270:
				a=3
		else:
			#print("Segment3")
			if yaw == 0:
				a=0
			if yaw == 270:
				a=1
			if yaw == 90:
				a=2
			if yaw == 180:
				a=3
	elif ( x>=7 ) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
		#print("Segment4")
		if yaw==90:
			a=0
		if yaw==180:
			a=2
		if yaw==0:
			a=1
		if yaw==270:
			a=3
	elif ((x<7) and (x>3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
		if yaw==0:
			a=0
		if yaw==270:
			a=1
		if yaw==90:
			a=2
		if yaw==180:
			a=3
	elif ((x<3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
		if yaw==0:
			a=2
		if yaw==270:
			a=0
		if yaw==180:
			a=1
		if yaw==90:
			a=3
	elif (z==14) or (z==15):
		if yaw==0:
			a=0
		if yaw==270:
			a=1
		if yaw==90:
			a=2
		if yaw==180:
			a=3
	elif (z==17) or (z==16):
		#print("Segment6")
		if yaw==270:
			a=0
		if yaw==180:
			a=1
		if yaw==0:
			a=2
		if yaw==90:
			a=3
	elif (z>17):
		#print("Segment6")
		if yaw==270:
			a=2
		if yaw==180:
			a=0
		if yaw==0:
			a=3
		if yaw==90:
			a=1
	else:
		a = int(np.random.randint(4))

	if a==-1:
		a = int(np.random.randint(4))
	#print("Success checking for best move")
	return a

class Agent:

	def __init__(self, model, memory=None, memory_size=500, nb_frames=None):
		assert len(model.get_output_shape_at(0)) == 2, "Model's output shape should be (nb_samples, nb_actions)."
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)
		if not nb_frames and not model.get_input_shape_at(0)[1]:
			raise Exception("Missing argument : nb_frames not provided")
		elif not nb_frames:
			nb_frames = model.get_input_shape_at(0)[1]
		elif model.get_input_shape_at(0)[1] and nb_frames and model.get_input_shape_at(0)[1] != nb_frames:
			raise Exception("Dimension mismatch : time dimension of model should be equal to nb_frames.")
		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		self.exp_replay.reset_memory()

	def check_game_compatibility(self, game):
		#if len(self.model.input_layers_node_indices) != 1:
			#raise Exception('Multi node input is not supported.')
		game_output_shape = (1, None) + game.get_frame().shape
		if len(game_output_shape) != len(self.model.get_input_shape_at(0)):
			raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		else:
			for i in range(len(self.model.get_input_shape_at(0))):
				if self.model.get_input_shape_at(0)[i] and game_output_shape[i] and self.model.get_input_shape_at(0)[i] != game_output_shape[i]:
					raise Exception('Dimension mismatch. Input shape of the model should be compatible with the game.')
		if len(self.model.get_output_shape_at(0)) != 2 or self.model.get_output_shape_at(0)[1] != game.nb_actions:
			raise Exception('Output shape of model should be (nb_samples, nb_actions).')

	def get_game_data(self, game):
		frame = game.get_frame()
		if self.frames is None:
			self.frames = [frame] * self.nb_frames
		else:
			self.frames.append(frame)
			self.frames.pop(0)
		return np.expand_dims(self.frames, 0)

	def clear_frames(self):
		self.frames = None

	def train(self, game, nb_epoch=1000, batch_size=50, gamma=0.9, epsilon=[1., .1], epsilon_rate=0.5, reset_memory=False, observe=0, checkpoint=None, total_sessions=0, session_id=1):
		self.check_game_compatibility(game)

		ts = int(time.time())
		#fn = "gold-{}.csv".format(ts)

		#fn = "9nyc-250-1000-epr8-heat-adam.csv"
		#fn = "400-rl-nopool.csv"
		fn = "3-normal.csv"
		fn2 = "heat.csv"
		#advice_type = "OA"
		advice_type = "OA"
		meta_advice_type = "HFHA"
		#meta_feedback_frequency = 0.1
		#meta_feedback_frequency = 0.5 #HF!!!
		meta_feedback_frequency = 0.1 #LF!!!

		heatmap = [ [0]*20 for i in range(20)]

		if session_id == 1:
			advice_type = "OA"
		if session_id == 2:
			advice_type = "NA"
		if session_id == 3:
			advice_type = "RL"
		# print(heatmap)
		# with open("dummyheat.csv",'a') as f2:
		# 	csvWriter = csv.writer(f2,delimiter=',')
		# 	csvWriter.writerows(heatmap)
		# if ( session_id >= 3 and session_id < 5 ):
		# 	print("Switching to HFLA")
		# 	meta_advice_type = "HFLA"
		# 	#meta_feedback_frequency = 0.1
		# elif ( session_id >= 5 and session_id < 7 ):
		# 	print("Switching to LFHA")
		# 	meta_feedback_frequency = 0.1
		# 	meta_advice_type = "LFHA"
		# elif ( session_id >= 7 and session_id < 9 ):
		# 	print("Switching to LFLA")
		# 	meta_advice_type = "LFLA"
		# elif ( session_id >= 9 and session_id < 11 ):
		# 	advice_type = "OA"
		# 	print("Switching to NA HFLA")
		# 	meta_advice_type = "HFLA"
		# 	meta_feedback_frequency = 0.5
		# elif ( session_id >= 11 and session_id < 13 ):
		# 	print("Switching to NA HFLA")
		# 	meta_advice_type = "HFLA"
		# 	#meta_feedback_frequency = 0.1
		# elif ( session_id >= 13 and session_id < 15 ):
		# 	print("Switching to NA LFHA")
		# 	meta_feedback_frequency = 0.1
		# 	meta_advice_type = "LFHA"
		# elif ( session_id >= 15 and session_id < 17 ):
		# 	print("Switching to NA LFLA")
		# 	meta_advice_type = "LFLA"

		# if ( session_id >= 2 and session_id < 3 ):
		# 	meta_feedback_frequency = 0.1
		# 	print("Switching to LFHA")
		# 	advice_type = "OA"
		# 	meta_advice_type = "LFHA"
		# 	meta_feedback_frequency = 0.1
		# elif ( session_id >= 3 and session_id < 4 ):
		# 	advice_type = "NA"
		# 	print("Switching to NA LFHA")
		# 	meta_feedback_frequency = 0.1
		# 	meta_advice_type = "LFHA"
		# elif ( session_id >= 4 and session_id < 5 ):
		# 	print("Switching to NA LFLA")
		# 	meta_feedback_frequency = 0.1
		# 	advice_type = "NA"
		# 	meta_advice_type = "LFLA"
		# elif ( session_id >= 5 and session_id < 6 ):
		# 	advice_type = "OA"
		# 	print("Switching to OA HFHA")
		# 	meta_advice_type = "HFHA"
		# 	meta_feedback_frequency = 0.5
		# elif ( session_id >= 6 and session_id < 7 ):
		# 	advice_type = "NA"
		# 	meta_feedback_frequency = 0.5
		# 	print("Switching to NA HFHA")
		# 	meta_advice_type = "HFHA"
		# 	meta_feedback_frequency = 0.5
		# elif ( session_id >= 7 and session_id < 8 ):
		# 	advice_type = "NA"
		# 	print("Switching to NA HFLA")
		# 	meta_feedback_frequency = 0.5
		# 	meta_advice_type = "HFLA"
		# elif ( session_id >= 8 and session_id < 9 ):
		# 	advice_type = "OA"
		# 	meta_feedback_frequency = 0.5
		# 	print("Switching to OA HFLA")
		# 	meta_advice_type = "HFLA"

		# if ( session_id >= 4 and session_id < 7 ):
		# 	#print("Switching to LFLA")
		# 	advice_type = "RL"
		# 	#meta_advice_type = "LFLA"
		# elif ( session_id >= 7 and session_id < 10 ):
		# 	# with open("1RLheat.csv",'a') as f2:
		# 	# 	csvWriter = csv.writer(f2,delimiter=',')
		# 	# 	csvWriter.writerows(heatmap)
		# 	# 	heatmap = [ [0]*20 for i in range(20)]
		# 	advice_type = "NA"
		# 	#print("Switching to LFHA")
		# 	#meta_feedback_frequency = 0.1
		# 	#meta_advice_type = "LFHA"
		# elif ( session_id >= 10 ):
		# 	# with open("1NAheat.csv",'a') as f2:
		# 	# 	csvWriter = csv.writer(f2,delimiter=',')
		# 	# 	csvWriter.writerows(heatmap)
		# 	# 	heatmap = [ [0]*20 for i in range(20)]
		# 	#print("Switching to LFLA")
			
		# 	#meta_advice_type = "LFLA"
		# 	advice_type = "NA"

		# with open(fn,'w') as f:
		# 	f.write('session_id,advice_type,time,epoch,frames,score,win_perc,loss'+'\n')
		# 	f.flush()
		# 	f.close()            with open(fn,'a') as f:
		with open(fn,'a') as f:
			total_frames = 0
			#f.write('session_id,advice_type,time,epoch,frames,score,win_perc,loss'+'\n')
			#f.flush()
			if type(epsilon)  in {tuple, list}:
				delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
				final_epsilon = epsilon[1]
				epsilon = epsilon[0]
			else:
				final_epsilon = epsilon
			model = self.model
			nb_actions = model.get_output_shape_at(0)[-1]
			win_count = 0
			rolling_win_window = []
			max_obs_loss = -99999999999999999
			m_loss = -99999999
			for epoch in range(nb_epoch):
				lastAdviceStep = 0
				adviceGiven = 0
				adviceAttempts = 0
				modelActions = 0
				print(heatmap)
				loss = 0.
				game.reset()
				self.clear_frames()
				if reset_memory:
					self.reset_memory()
				game_over = False
				S = self.get_game_data(game)
				savedModel = False
				while not game_over:
					a = 0
					if advice_type == "RL":
						if np.random.random() < epsilon or epoch < observe:
							a = int(np.random.randint(game.nb_actions))
							#print("Random Action")
						else:
							q = model.predict(S) #use the prediction confidence to determine whether to ask the player for help
							qs = model.predict_classes(S)
							#a = int(np.argmax(qs[0]))
							#highest_conf = np.amax(q)
							#print("Game Grid: {}".format(game.get_grid()))
							#print("Highest MSE Confidence = {}".format(highest_conf))
							#a = int(np.argmax(q[0]))
							a = int(np.argmax(qs[0]))
					if advice_type == "OA":
						if np.random.random() < epsilon or epoch < observe:
							a = int(np.random.randint(game.nb_actions))
							#print("Random Action")
						else:
							q = model.predict(S) #use the prediction confidence to determine whether to ask the player for help
							qs = model.predict_classes(S)
							#print(qs)
							#print(q)
							highest_loss = abs(np.amax(q)) #added ABS
							lowest_loss = abs(np.amin(q))
							#print(highest_loss)
							#print("HighestLoss:{}".format(highest_loss))
							if highest_loss > max_obs_loss and highest_loss != 0:
								max_obs_loss = highest_loss
								#print("MaxLoss:{}".format(highest_loss))
							#inn = highest_loss / max_obs_loss
							relative_cost = np.power(lowest_loss / max_obs_loss,0.5)
							#print("RelCostA:{}".format(relative_cost))
							if relative_cost < 1e-20:
								relative_cost = 1e-20
							relative_cost = -1/(np.log(relative_cost)-1)
							#print("RelCostB:{}".format(relative_cost))
							confidence_score_max = 1
							confidence_score_min = 0.01
							feedback_chance = confidence_score_min + (confidence_score_max-confidence_score_min) * relative_cost
							
							if feedback_chance < 0.01:
								feedback_chance = 0.01
							#if feedback_chance < 0.1:
							giveAdvice = False
							if (random.random() < meta_feedback_frequency):
								giveAdvice = True
								adviceAttempts = adviceAttempts + 1
							if (relative_cost <= 0.25 and game.stepsTaken >= (lastAdviceStep+10)) or giveAdvice == False:
								#print("HC: {}".format(max_obs_loss))
								modelActions = modelActions + 1
								#print("Highest Loss: {} RC: {} POS: Q0:{}".format(highest_loss, relative_cost, q[0]))
								a = int(np.argmax(qs[0]))
							else:
								if random.random() < .5 and (meta_advice_type == "HFLA" or meta_advice_type == "LFLA"):
									lastAdviceStep = game.stepsTaken
									a = int(np.random.randint(game.nb_actions))
									adviceGiven = adviceGiven + 1
									#print("Taking BAD Player Action")
								else:
									lastAdviceStep = game.stepsTaken
									adviceGiven = adviceGiven + 1
									x = game.location[0]
									z = game.location[1]
									yaw = game.location[2]
									a=-1
									#print(yaw)
									if z<=6:
										if x < 12:
											#print("Segment1")
											if yaw==270:
												a=0
											if yaw==180:
												a=1
											if yaw==90:
												a=3
											if yaw==0:
												a=2
										elif x > 15:
											#print("Segment2")
											if yaw==90:
												a=0
											if yaw==180:
												a=2
											if yaw==0:
												a=1
											if yaw==270:
												a=3
										else:
											#print("Segment3")
											if yaw == 0:
												a=0
											if yaw == 270:
												a=1
											if yaw == 90:
												a=2
											if yaw == 180:
												a=3
									elif ( x>=7 ) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										#print("Segment4")
										if yaw==90:
											a=0
										if yaw==180:
											a=2
										if yaw==0:
											a=1
										if yaw==270:
											a=3
									elif ((x<7) and (x>3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										if yaw==0:
											a=0
										if yaw==270:
											a=1
										if yaw==90:
											a=2
										if yaw==180:
											a=3
									elif ((x<3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										if yaw==0:
											a=2
										if yaw==270:
											a=0
										if yaw==180:
											a=1
										if yaw==90:
											a=3
									elif (z==14) or (z==15):
										if yaw==0:
											a=0
										if yaw==270:
											a=1
										if yaw==90:
											a=2
										if yaw==180:
											a=3
									elif (z==17) or (z==16):
										#print("Segment6")
										if yaw==270:
											a=0
										if yaw==180:
											a=1
										if yaw==0:
											a=2
										if yaw==90:
											a=3
									elif (z>17):
										#print("Segment6")
										if yaw==270:
											a=2
										if yaw==180:
											a=0
										if yaw==0:
											a=3
										if yaw==90:
											a=1
									else:
										a = int(np.random.randint(game.nb_actions))

									if a==-1:
										a = int(np.random.randint(game.nb_actions))
									# if z < 6 and x < 13:
									# 	print("Segment1")
									# 	if yaw == 270:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z < 8 and x >= 13:
									# 	print("Segment2")
									# 	if yaw == 0:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z >= 8 and x == 13:
									# 	print("Segment3")
									# 	if yaw == 90:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z >= 8 and z<= 17 and x < 6:
									# 	print("Segment4")
									# 	if yaw == 0:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z > 18 and x < 18:
									# 	print("Segment5")
									# 	if yaw == 270:
									# 		a = 0
									# 	else:
									# 		a = 1
									# else:
									# 	a = int(np.argmax(q[0]))

								#print("Game Grid: {}".format(game.get_grid()))
								#print("Highest MSE Confidence = {}".format(highest_conf))
							
					if advice_type == "NA":
						if np.random.random() < epsilon or epoch < observe:
							a = int(np.random.randint(game.nb_actions))
							game.play(a)
							heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
							#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
							#f2.flush()
							r = game.get_score()
							S_prime = self.get_game_data(game)
							game_over = game.is_over()
							transition = [S, a, r, S_prime, game_over]
							self.memory.remember(*transition)
							S = S_prime
							#print("Random Action")
						else:
							q = model.predict(S) #use the prediction confidence to determine whether to ask the player for help
							qs = model.predict_classes(S)
							highest_loss = abs(np.amax(q)) #added ABS
							lowest_loss = abs(np.amin(q))
							#print("HighestLoss:{}".format(highest_loss))
							if highest_loss > max_obs_loss and highest_loss != 0:
								max_obs_loss = highest_loss
								#print("MaxLoss:{}".format(highest_loss))
							#inn = highest_loss / max_obs_loss
							relative_cost = np.power(lowest_loss / max_obs_loss,0.5)
							#print("RelCostA:{}".format(relative_cost))
							if relative_cost < 1e-20:
								relative_cost = 1e-20
							relative_cost = -1/(np.log(relative_cost)-1)
							#print("RelCostB:{}".format(relative_cost))
							confidence_score_max = 1
							confidence_score_min = 0.01
							feedback_chance = confidence_score_min + (confidence_score_max-confidence_score_min) * relative_cost
							#feedback_chance = random.random()
							#print("Feedback Chance: {}".format(feedback_chance))
							if feedback_chance < 0.01:
								feedback_chance = 0.01
							#if feedback_chance > meta_feedback_frequency:
							#if feedback_chance < 0.1:
							#print(relative_cost)
							giveAdvice = False
							if (random.random() < meta_feedback_frequency):
								giveAdvice = True
								adviceAttempts = adviceAttempts + 1
							if (relative_cost <= 0.25 and game.stepsTaken >= (lastAdviceStep+10)) or giveAdvice == False:
								#print("Taking Model Action")
								#print("HC: {}".format(max_obs_loss))
								#print("Confidence: {} RC: {}".format(feedback_chance, relative_cost))
								modelActions = modelActions + 1
								#a = int(np.argmin(q[0]))
								a = int(np.argmax(qs[0]))
								game.play(a)
								heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
								#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
								#f2.flush()
								r = game.get_score()
								S_prime = self.get_game_data(game)
								game_over = game.is_over()
								transition = [S, a, r, S_prime, game_over]
								self.memory.remember(*transition)
								S = S_prime
							else:
								#print("Taking Player Action")
								if random.random() < .5 and (meta_advice_type == "HFLA" or meta_advice_type == "LFLA"):
									a = int(np.random.randint(game.nb_actions))
									adviceGiven = adviceGiven + 1
									game.play(a)
									heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
									lastAdviceStep = game.stepsTaken
									#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
									#f2.flush()
									r = game.get_score()
									S_prime = self.get_game_data(game)
									game_over = game.is_over()
									transition = [S, a, r, S_prime, game_over]
									self.memory.remember(*transition)
									S = S_prime
									if game_over == False:
										#game.play(checkForBestMove(game.location[0],game.location[1],game.location[2]))
										a = int(np.random.randint(game.nb_actions))
										game.play(a)
										heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
										#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
										#f2.flush()
										r = game.get_score()
										S_prime = self.get_game_data(game)
										game_over = game.is_over()
										transition = [S, a, r, S_prime, game_over]
										self.memory.remember(*transition)
										S = S_prime
										# if game_over == False:
										# 	game.play(checkForBestMove(game.location[0],game.location[1],game.location[2]))
										# 	heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
										# 	#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
										# 	#f2.flush()
										# 	r = game.get_score()
										# 	S_prime = self.get_game_data(game)
										# 	game_over = game.is_over()
										# 	transition = [S, a, r, S_prime, game_over]
										# 	self.memory.remember(*transition)
										# 	S = S_prime
									#print("Taking BAD Player Action")
								else:
									adviceGiven = adviceGiven + 1
									lastAdviceStep = game.stepsTaken
									x = game.location[0]
									z = game.location[1]
									yaw = game.location[2]
									#print(x)
									#print(z)
									a=-1
									#print(yaw)
									if z<=6:
										if x < 12:
											#print("Segment1")
											if yaw==270:
												a=0
											if yaw==180:
												a=1
											if yaw==90:
												a=3
											if yaw==0:
												a=2
										elif x > 15:
											#print("Segment2")
											if yaw==90:
												a=0
											if yaw==180:
												a=2
											if yaw==0:
												a=1
											if yaw==270:
												a=3
										else:
											#print("Segment3")
											if yaw == 0:
												a=0
											if yaw == 270:
												a=1
											if yaw == 90:
												a=2
											if yaw == 180:
												a=3
									elif ( x>=7 ) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										#print("Segment4")
										if yaw==90:
											a=0
										if yaw==180:
											a=2
										if yaw==0:
											a=1
										if yaw==270:
											a=3
									elif ((x<7) and (x>3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										if yaw==0:
											a=0
										if yaw==270:
											a=1
										if yaw==90:
											a=2
										if yaw==180:
											a=3
									elif ((x<3)) and ( (z == 7) or (z == 8) or (z == 9) or (z == 10) or (z == 11) or (z == 12) ):
										if yaw==0:
											a=2
										if yaw==270:
											a=0
										if yaw==180:
											a=1
										if yaw==90:
											a=3
									elif (z==14) or (z==15):
										if yaw==0:
											a=0
										if yaw==270:
											a=1
										if yaw==90:
											a=2
										if yaw==180:
											a=3
									elif (z==17) or (z==16):
										#print("Segment6")
										if yaw==270:
											a=0
										if yaw==180:
											a=1
										if yaw==0:
											a=2
										if yaw==90:
											a=3
									elif (z>17):
										#print("Segment6")
										if yaw==270:
											a=2
										if yaw==180:
											a=0
										if yaw==0:
											a=3
										if yaw==90:
											a=1
									else:
										a = int(np.random.randint(game.nb_actions))

									if a==-1:
										a = int(np.random.randint(game.nb_actions))
									# #print(yaw)
									# if z < 6 and x < 13:
									# 	#print("Segment1")
									# 	if yaw == 270:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z < 8 and x >= 13:
									# 	#print("Segment2")
									# 	if yaw == 0:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z >= 8 and x == 13:
									# 	#print("Segment3")
									# 	if yaw == 90:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z >= 8 and z<= 17 and x < 6:
									# 	#print("Segment4")
									# 	if yaw == 0:
									# 		a = 0
									# 	else:
									# 		a = 1
									# elif z > 18 and x < 18:
									# 	#print("Segment5")
									# 	if yaw == 270:
									# 		a = 0
									# 	else:
									# 		a = 1
									# else:
									# 	a = int(np.argmax(q[0]))

								#Play an extra 2 times (for NA friction)
								game.play(a)
								heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
								#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
								#f2.flush()
								r = game.get_score()
								S_prime = self.get_game_data(game)
								game_over = game.is_over()
								transition = [S, a, r, S_prime, game_over]
								self.memory.remember(*transition)
								S = S_prime
								if game_over == False:
									game.play(checkForBestMove(game.location[0],game.location[1],game.location[2]))
									heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
									#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
									#f2.flush()
									r = game.get_score()
									S_prime = self.get_game_data(game)
									game_over = game.is_over()
									transition = [S, a, r, S_prime, game_over]
									self.memory.remember(*transition)
									S = S_prime
									# if game_over == False:
									# 	game.play(checkForBestMove(game.location[0],game.location[1],game.location[2]))
									# 	heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
									# 	#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
									# 	#f2.flush()
									# 	r = game.get_score()
									# 	S_prime = self.get_game_data(game)
									# 	game_over = game.is_over()
									# 	transition = [S, a, r, S_prime, game_over]
									# 	self.memory.remember(*transition)
									# 	S = S_prime
					if game_over == False:
						if advice_type != "NA":
							game.play(a)
							heatmap[game.location[0]][game.location[1]] = heatmap[game.location[0]][game.location[1]] + 1
							#f2.write('{},{},{},{}\n'.format(advice_type,game.location[0],game.location[1],1 ))
							#f2.flush()
							r = game.get_score()
							S_prime = self.get_game_data(game)
							game_over = game.is_over()
							transition = [S, a, r, S_prime, game_over]
							self.memory.remember(*transition)
							S = S_prime
					if epoch >= observe:
						batch = self.memory.get_batch(model=model, batch_size=batch_size, gamma=gamma)
						if batch:
							inputs, targets = batch
							mtob = model.train_on_batch(inputs, targets)
							if mtob > m_loss:
								m_loss = mtob
							loss += float(mtob)
							#print( "LOSS: {} CULM_LOSS: {}".format(mtob,loss))
					if checkpoint and (savedModel == False) and ((epoch + 1 - observe) % checkpoint == 0 or epoch + 1 == nb_epoch):
						#model.save_weights('weights.dat')
						print("Checkpoint... saving model..")
						if advice_type == "OA":
							model.save('oa_model.h5')
						if advice_type == "NA":
							model.save('na_model.h5')
						if advice_type == "RL":
							model.save('rl_model.h5')
						# model_json = model.to_json()
						# with open("model.json", "w") as json_file:
						#    json_file.write(model_json)
						# #serialize weights to HDF5
						# model.save_weights("model.h5")
						savedModel = True
				if game.is_won():
					win_count += 1
					rolling_win_window.insert(0,1)
				else:
					rolling_win_window.insert(0,0)
				if epsilon > final_epsilon and epoch >= observe:
					epsilon -= delta
					percent_win = 0
					cdt = datetime.datetime.now()
					if sum(rolling_win_window) != 0:
						percent_win = sum(rolling_win_window)/4
					total_frames = total_frames + game.stepsTaken
					f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(session_id,advice_type,meta_advice_type,str(cdt), (epoch + 1), total_frames, game.score, percent_win, epsilon, loss, game.stepsTaken, adviceGiven, adviceAttempts, modelActions ))
					f.flush()
					print("Session: {} | Time: {} | Epoch {:03d}/{:03d} | Steps {:.4f} | Epsilon {:.2f} | Score {} | Loss {}".format(session_id,str(cdt),epoch + 1, nb_epoch, game.stepsTaken, epsilon, game.score, loss ))
					if len(rolling_win_window) > 4:
						rolling_win_window.pop()
					time.sleep(1.0)

			if advice_type == "OA":
				with open("{}OAheatxtues.csv".format(session_id),'w+') as f2:
					csvWriter = csv.writer(f2,delimiter=',')
					csvWriter.writerows(heatmap)
				#heatmap = [ [0]*20 for i in range(20)]
			if advice_type == "RL":
				with open("{}RLheatxtues.csv".format(session_id),'w+') as f2:
					csvWriter = csv.writer(f2,delimiter=',')
					csvWriter.writerows(heatmap)
				#heatmap = [ [0]*20 for i in range(20)]
			if advice_type == "NA":
				with open("{}NAheatxtues.csv".format(session_id),'w+') as f2:
					csvWriter = csv.writer(f2,delimiter=',')
					csvWriter.writerows(heatmap)
				#heatmap = [ [0]*20 for i in range(20)]

	def play(self, game, nb_epoch=10, epsilon=0., visualize=False):
		self.check_game_compatibility(game)
		model = self.model
		win_count = 0
		frames = []
		for epoch in range(nb_epoch):
			print("Playing")
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			if visualize:
				frames.append(game.draw())
			game_over = False
			while not game_over:
				if np.random.rand() < epsilon:
					print("random")
					action = int(np.random.randint(0, game.nb_actions))
				else:
					q = model.predict(S)[0]
					possible_actions = game.get_possible_actions()
					q = [q[i] for i in possible_actions]
					action = possible_actions[np.argmax(q)]
				print(action)
				game.play(action)
				S = self.get_game_data(game)
				if visualize:
					frames.append(game.draw())
				game_over = game.is_over()
			if game.is_won():
				win_count += 1
		print("Accuracy {} %".format(100. * win_count / nb_epoch))
		#Visualizing/printing images is currently super slow
		if visualize:
			if 'images' not in os.listdir('.'):
				os.mkdir('images')
			for i in range(len(frames)):
				plt.imshow(frames[i], interpolation='none')
				plt.savefig("images/" + game.name + str(i) + ".png")
