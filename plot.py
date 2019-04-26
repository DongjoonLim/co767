import matplotlib.pyplot as plt
import numpy as np
#DQN plot
dqn_episode = np.load('epi_dqn.npy')
dqn_test_reward = np.load('reward_epi_dqn.npy')
plt.plot(dqn_episode, dqn_test_reward)
plt.ylabel('Test Reward')
plt.xlabel('Episode')
plt.ylim(-1300, 1000)
plt.title('DQN')
plt.savefig('DQN.png')
plt.show()

#Nature DQN
nature_dqn_episode = np.load('epi_nature_dqn.npy')
nature_dqn_test_reward = np.load('reward_epi_nature_dqn.npy')
plt.plot(nature_dqn_episode, nature_dqn_test_reward)
plt.ylabel('Test Reward')
plt.xlabel('Episode')
plt.ylim(-1300, 1000)
plt.title('Nature DQN')
plt.savefig('Nature_DQN.png')
plt.show()

#Double DQN
ddqn_episode = np.load('epi_ddqn.npy')
ddqn_test_reward = np.load('reward_epi_ddqn.npy')
plt.plot(ddqn_episode, ddqn_test_reward)
plt.ylabel('Test Reward')
plt.xlabel('Episode')
plt.ylim(-1300, 1000)
plt.title('Double DQN')
plt.savefig('DDQN.png')
plt.show()

#Dueling DQN
duel_dqn_episode = np.load('epi_duel_dqn.npy')
duel_dqn_test_reward = np.load('reward_epi_duel_dqn.npy')
plt.plot(duel_dqn_episode, duel_dqn_test_reward)
plt.ylabel('Test Reward')
plt.xlabel('Episode')
plt.ylim(-1300, 1000)
plt.title('Dueling DQN')
plt.savefig('Dueling_DQN.png')
plt.show()

#Combine 2 plots
plt.plot(dqn_episode, dqn_test_reward, label='DQN')
plt.plot(nature_dqn_episode, nature_dqn_test_reward, label='Nature DQN')
#plt.plot(ddqn_episode, ddqn_test_reward, 'ro', label = 'Double DQN')
#plt.plot(duel_dqn_episode, duel_dqn_test_reward, 'yo', label='Dueling DQN')
plt.ylim(-1300, 1000)
plt.title('Comparison of DQN and Nature DQN')
plt.legend()
plt.savefig('Combine_1.png')
plt.show()

#Combine 2 plots
#plt.plot(dqn_episode, dqn_test_reward, 'go', label='DQN')
plt.plot(nature_dqn_episode, nature_dqn_test_reward, label='Nature DQN')
plt.plot(ddqn_episode, ddqn_test_reward, label = 'Double DQN')
#plt.plot(duel_dqn_episode, duel_dqn_test_reward, 'yo', label='Dueling DQN')
plt.ylim(-1300, 1000)
plt.title('Comparison of Nature DQN and Double DQN')
plt.legend()
plt.savefig('Combine_2.png')
plt.show()

#Combine 2 plots
#plt.plot(dqn_episode, dqn_test_reward, 'go', label='DQN')
#plt.plot(nature_dqn_episode, nature_dqn_test_reward, 'bo', label='Nature DQN')
plt.plot(ddqn_episode, ddqn_test_reward, label = 'Double DQN')
plt.plot(duel_dqn_episode, duel_dqn_test_reward, label='Dueling DQN')
plt.ylim(-1300, 1000)
plt.title('Comparison of Double DQN and Dueling DQN')
plt.legend()
plt.savefig('Combine_3.png')
plt.show()






