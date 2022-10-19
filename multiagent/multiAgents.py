# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food = successorGameState.getFood().asList()
        distMin = float("inf")
        # close food ++
        for f in food:
            distMin = min(distMin, manhattanDistance(newPos, f))
        # close ghost --
        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(newPos, ghost) < 5 :
                return -float('inf')

        return successorGameState.getScore() + 1.0/distMin

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        successor , value = self.max_value(gameState,0,0)
        return successor
    
    def max_value(self, gameState, agentIndex,depth):
        val = (None,-float("inf"))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextVal = (successor,self.value(nextState, (depth+1)%gameState.getNumAgents(), depth+1))
            val = max(val,nextVal,key=lambda x:x[1])       
        return val

    def min_value(self, gameState, agentIndex,depth):
        val = (None,float("inf"))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextVal = (successor,self.value(nextState, (depth+1)%gameState.getNumAgents(), depth+1))
            val = min(val,nextVal,key=lambda x:x[1])
        return val
    
    def value(self, gameState, agentIndex,depth):
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex==0 :
            return self.max_value(gameState, agentIndex, depth)[1]
        else :
            return self.min_value(gameState, agentIndex, depth)[1]

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        successor , value = self.max_value(gameState,0,0,-float("inf"),float("inf"))
        return successor

    def max_value(self, gameState, agentIndex,depth, alpha, beta):
        val = (None,-float("inf"))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextVal = (successor,self.alphabet(nextState, (depth+1)%gameState.getNumAgents(), depth+1,alpha, beta))
            val = max(val,nextVal,key=lambda x:x[1])     
            if val[1] > beta:
                return val 
            else :
                alpha = max(val[1] , alpha)
        return val

    def min_value(self, gameState, agentIndex,depth, alpha, beta):
        val = (None,float("inf"))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextVal = (successor,self.alphabet(nextState, (depth+1)%gameState.getNumAgents(), depth+1, alpha, beta))
            val = min(val,nextVal,key=lambda x:x[1])
            if val[1] < alpha:
                return val 
            else :
                beta = min(val[1] , beta)        
        return val
    
    def alphabet(self, gameState, agentIndex,depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex==0 :
            return self.max_value(gameState, agentIndex, depth, alpha, beta)[1]
        else :
            return self.min_value(gameState, agentIndex, depth, alpha, beta)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"   
        maxDepth = self.depth * gameState.getNumAgents()
        successor , value =  self.expectimax(gameState, maxDepth, 0,None)
        return successor

    def max_val(self,gameState,depth,agentIndex, move):
        val = (None, -(float('inf')))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextMove = None
            if depth != self.depth * gameState.getNumAgents():
                nextMove = move
            else:
                nextMove = successor
            nextVal = self.expectimax(nextState,depth - 1,(agentIndex + 1) % gameState.getNumAgents(),nextMove)
            val = max(val,nextVal,key = lambda x:x[1])
        return val

    def exp_val(self,gameState,depth,agentIndex, move):
        score = 0
        propability = 1.0/len(gameState.getLegalActions(agentIndex))
        for successor in gameState.getLegalActions(agentIndex):
            nextState = gameState.generateSuccessor(agentIndex, successor)
            nextVal = self.expectimax(nextState, depth - 1, (agentIndex + 1) % gameState.getNumAgents(), move)
            score += nextVal[1] * propability
            val = (move,score)
        return val
    
    def expectimax(self, gameState, depth, agentIndex, move):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (move, self.evaluationFunction(gameState))
        if agentIndex == 0:
            return self.max_val(gameState,depth,agentIndex, move)
        else:
            return self.exp_val(gameState,depth,agentIndex,move)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I considered closest food, remaining food, remaining capsules and distance to ghosts
    Avg score was close to but less than 100, on changing the weights for each metric, avg score crossed 1000.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    remFood = currentGameState.getNumFood() + 1 
    remCaps = len(currentGameState.getCapsules()) + 1
    ghostDist = float('inf')
    for ghost in currentGameState.getGhostPositions():
        if (manhattanDistance(pos, ghost) < 2):
            return -float('inf')
        ghostDist = min(ghostDist,manhattanDistance(pos, ghost))
    closestFood = float('inf')
    for food in  currentGameState.getFood().asList():
        closestFood = min(closestFood, manhattanDistance(pos, food))
    eval = 1.0/remFood * 10000 + ghostDist + 1.0/(closestFood + 1) * 40 + 1.0/remCaps * 300
    return eval

# Abbreviation
better = betterEvaluationFunction
