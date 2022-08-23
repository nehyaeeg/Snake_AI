from collections import namedtuple
from enum import Enum
import pygame
import random

#GLOBALS
BLOCK_WIDTH = 20
FRAME_RATE = 20

#COLORS
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE = (0, 0, 255)
RED = (200,0,0)

font  = pygame.font.SysFont("Aria", 23)

#coordiantes
Point = namedtuple("Point",["x","y"])

#Directions available
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
pygame.init()
#class implementing the game environemt
class SnakeGame:
    
    def __init__(self,width=600,height=400):
        self.w = width #width of window
        self.h = height # height of window
        self.window = pygame.display.set_mode((self.w, self.h)) #create main window
        pygame.display.set_caption("Navid's Snake Game AI") #give title
        self.clock = pygame.time.Clock() # for keeping time
        self.reset_game()
        
    # For reseting the game after each episode
    def reset_game(self):
        self.direction = Direction.RIGHT # Defualt initila direction
        self.snake_head = Point(((self.w) //(2 * BLOCK_WIDTH)) * BLOCK_WIDTH, ((self.h) //(2 * BLOCK_WIDTH)) * BLOCK_WIDTH) # initial coordinates of head of snake
        # whole body of snake (top left corner of each block)/ initially contians only 3 blocks
        self.snake_body = [self.snake_head, #head
                           Point(self.snake_head.x - BLOCK_WIDTH,self.snake_head.y), # first left square
                           Point(self.snake_head.x - 2*BLOCK_WIDTH,self.snake_head.y)] # second left square
        self.score = 0
        self.isOver = False # termianl case reached or not
        self.food = self.__random_food()
        self.game_iter = 0 # number of iteration(moves) in the current episode
        
        
    #randomly place food, excludes edges
    def __random_food(self):
        
        food = Point(random.randint(0, ((self.w - BLOCK_WIDTH) //BLOCK_WIDTH ))* BLOCK_WIDTH, #x
                     random.randint(0, ((self.h - BLOCK_WIDTH) //BLOCK_WIDTH ))* BLOCK_WIDTH) #y
        
        #if collides with body try again
        if food in self.snake_body:
            return self.__random_food()
        
        else:
            return food
        
    #main play function causing snake to move
    # action comes from agent
    def play(self,action):
        self.game_iter += 1
        
        #to quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #if red x is clicked for closing
                pygame.quit()
                quit()
                        
        # move snake based on action given anc collect reward
        reward = self.__move_head(action)         
        self.__update_screen()
        self.clock.tick(FRAME_RATE)
        return self.score, self.isOver, reward
            
            
    # move to new position based on input
    def __move_head(self,action):
        
        # order matters 
        direction_list = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        
        # Action is an array: straitgh:[1,0,0] or right:[0,1,0], or Left:[0,0,1] 
        # wrt current direction as seen by snake
        if action[0] == 1:
            pass
        elif action[1] == 1:
            self.direction = direction_list [(direction_list.index(self.direction) + 1) % 4]
        elif action[2] == 1:
            self.direction = direction_list [(direction_list.index(self.direction) - 1) % 4]
            
        x = self.snake_head.x
        y = self.snake_head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_WIDTH
        elif self.direction == Direction.LEFT:
            x -= BLOCK_WIDTH
        elif self.direction == Direction.UP:
            y -= BLOCK_WIDTH
        elif self.direction == Direction.DOWN:
            y += BLOCK_WIDTH
            
        p = Point(x,y) # new head
            
        #termination case borders or collide with self or lingering without accomplishemnt
        if self.collision(p):
            self.isOver = True
            return -10 # reward
            
        self.snake_body.insert(0,p) #insert at front
        self.snake_head = p # update head           
        #Food is eaten
        if self.food == self.snake_head:
            self.score += 1
            self.food = self.__random_food() # renew food
            self.__update_screen()
            return 10 # reward
            
        else:
            self.snake_body.pop() # remove last one, indicating movement
            return 0
            
    # check for termination
    def collision(self, point:Point):
        if point.x<0 or point.x> self.w - BLOCK_WIDTH or point.y<0 or point.y> self.h - BLOCK_WIDTH or point in self.snake_body or self.game_iter > 100 * len(self.snake_body): # last element is to prevent lingering without achievement
            return True
        
        return False
            
        
        
        
    #draws snake, food, and score
    def __update_screen(self):
        self.window.fill(BLACK)
        for item in self.snake_body:
            pygame.draw.rect(self.window, WHITE, pygame.Rect(item.x, item.y, BLOCK_WIDTH, BLOCK_WIDTH)) # draw each block
            pygame.draw.rect(self.window, BLACK, pygame.Rect(item.x, item.y, 1, BLOCK_WIDTH)) # draw lines separating squares vertical
            pygame.draw.rect(self.window, BLACK, pygame.Rect(item.x, item.y, BLOCK_WIDTH, 1)) # draw lines separating squares horizontal
        pygame.draw.rect(self.window, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_WIDTH, BLOCK_WIDTH)) # draw food
        text = font.render(f"Score: {self.score}",True,WHITE) #show text score
        self.window.blit(text,(self.w //2 -18, 0))
        pygame.display.flip()