from snake_main_game import BLOCK_WIDTH, SnakeGame, Direction, Point
import torch
import numpy as np
from collections import deque
import matplotlib as mp
import random
from rl_model import Q_training, Qnet
MEMORY_SIZE = 10000 # FOR limtiing memory emphasizing newer experiences
BATCH_SIZE = 1000 # for SGD
ALPHA = 0.001 # learning rate for NN

class Agent:
    
    def __init__(self):
        self.mem = deque(maxlen= MEMORY_SIZE) # for replay
        self.e = 0 # epsilon for randomness, will be assigned later on
        self.number_of_games = 0 # total games played
        self.gamma = 0.9 # discount rate
        self.model = Qnet(11,256,3) # 11 in, 256, hidden, 3:out
        self.trainer = Q_training(self.model, ALPHA, self.gamma)
        
    
    def get_state(self, game:SnakeGame):
        
        # state: danger straight0, right1, left2, dir left3, right4, up 5, down6, food left7, right8, up9 down10
        state = [0] * 11
        
        dir = game.direction
                
        # position of the head of the snake
        x= game.snake_head.x 
        y= game.snake_head.y
        
        if dir == Direction.RIGHT:
            state[4] = 1 # dir right
            
            if game.collision(Point(x+BLOCK_WIDTH,y)): # danger straight
                state[0] = 1
            if game.collision(Point(x,y+BLOCK_WIDTH)): # danger right
                state[1] = 1
            if game.collision(Point(x,y-BLOCK_WIDTH)): # danger left
                state[2] = 1          
            
            
        elif dir == Direction.DOWN:
            state[6] = 1 # dir down
            
            if game.collision(Point(x+BLOCK_WIDTH,y)): # danger left
                state[2] = 1
            if game.collision(Point(x,y+BLOCK_WIDTH)): # danger straight
                state[0] = 1
            if game.collision(Point(x-BLOCK_WIDTH,y)): # danger right
                state[1] = 1
                
                
        elif dir == Direction.LEFT:
            state[3] = 1 # dir left
            
            if game.collision(Point(x,y-BLOCK_WIDTH)): # danger right
                state[1] = 1
            if game.collision(Point(x,y+BLOCK_WIDTH)): # danger left
                state[2] = 1
            if game.collision(Point(x-BLOCK_WIDTH,y)): # danger straight
                state[0] = 1
                
        else: # up
            state[5] = 1 # dir up
            
            if game.collision(Point(x+BLOCK_WIDTH,y)): # danger right
                state[1] = 1
            if game.collision(Point(x,y-BLOCK_WIDTH)): # danger straight
                state[0] = 1
            if game.collision(Point(x-BLOCK_WIDTH,y)): # danger left
                state[2] = 1
                
                
        # food location
        
        # left or right
        if x - game.food.x < 0 : # right
            state[8] = 1
        if x - game.food.x > 0: # left
            state[7] = 1
            
        # up or down
        if y - game.food.y < 0 : # down
            state[10] = 1
        if y - game.food.y > 0: # up
            state[9] = 1
        
        
        return np.array(state, dtype=int)
    
    
    
        
        
    
    def store_mem(self, state, action, reward, new_state, isOver):
        self.mem.append((state,action, reward, new_state, isOver))
        
    
    # for target network: Experience replay
    def train_1(self):
        
        if len(self.mem) > BATCH_SIZE: # if more than needed, select a sample
            sample = random.sample(self.mem, BATCH_SIZE) # take a sample of batchsize
        else:
            sample = self.mem # whole memory

        
        states,actions, rewards, new_states, isOvers = zip(*sample) # unpack and group
        self.trainer.train_step(states,actions, rewards, new_states, isOvers) # train on all samples elements
    
    #for short-term predictor
    def traint_2(self,state,action, reward, new_state, isOver):
        self.trainer.train_step(state,action, reward, new_state, isOver) # train with each step taken
    
    
    # convert raw output of Qnet(model) to a 0,1 vector for action determination
    def get_action(self, state):
        action = [0,0,0]
        self.e = 90 - self.number_of_games # as games progress and learns, less random action
        
        # select randomly
        if random.randint(0, 240) < self.e:
            index = random.randint(0,2)
            action[index] = 1
            
        # use prediction
        else:
            state_input = torch.tensor(state, dtype=torch.float) # convert state to tensor to input to model
            high_index = torch.argmax( self.model(state_input)).item() # index of highest  in action vector //uses Qnet model for a forward iteration
            action[high_index] = 1 # set which action to take
            
        return action
            

# global
def train():
    record = 0 # highest score
    agent = Agent()
    game = SnakeGame()
    
    while True:
        
        state = agent.get_state(game) # current state
        action = agent.get_action(state)
        score, isOver, reward = game.play(action) # goes to next state/ Adds to game.iteration
        new_state = agent.get_state(game) # retain state_new
        agent.store_mem(state,action, reward, new_state, isOver)
        agent.traint_2(state,action, reward, new_state, isOver)
        
        if isOver: # after each episode is over, target network is trained again
            agent.number_of_games += 1
            
            agent.train_1() # train by experiecne replay of the whole episode
            #plot
            game.reset_game()
            if score > record:
                record = score
                agent.model.save()
            print(f"Game Number : {agent.number_of_games}, Score: {score}, Current Highest Record: {record}")
        
        

if __name__ == "__main__":
    train()
    
    
    
    
    
