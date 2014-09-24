# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        newGhostDistances = [manhattanDistance(newPos, ghost) for ghost in newGhostPositions]
        newScaredTimes = [ghost.scaredTimer for ghost in newGhostStates]
        food = min([util.manhattanDistance(food, newPos) for food in newFood.asList()]) if newFood.asList() else 0
        eatFood = len(newFood.asList()) < len(oldFood.asList())

        if max(newScaredTimes) <= 0:
            if min(newGhostDistances) >= 3:
                return -20 * food + eatFood * 1000
            if min(newGhostDistances) <= 0:
                return -float('inf')
            return sum([-13/dist for dist in newGhostDistances]) + eatFood * 50


        for state in newGhostStates:
            if state.getPosition() == newPos and not state.scaredTimer > 0:
                return -float('inf')
            if state.getPosition() == newPos and state.scaredTimer > 0:
                return float('inf')
        return -sum(newGhostDistances)/len(newGhostDistances)


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
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

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
      Your minimax agent (question 7)
    """

    def getAction(self, gameState, agent = 0, depth = None):
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
        """
        num_agents = gameState.getNumAgents()
        depth = depth if depth != None else self.depth * num_agents
        eval_fn = lambda action: self.evaluationFunction(gameState.generateSuccessor(agent, action))
        next_agent = (agent + 1)%num_agents
        min_or_max = max if agent == 0 else min

        if not gameState.getLegalActions(agent) or depth == 0:
            return self.evaluationFunction(gameState)
        if not gameState.getLegalActions(agent):
          return min_or_max([eval_fn(action) for action in gameState.getLegalActions(agent)])

        actions = gameState.getLegalActions(agent)
        successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]
        mini_max_val = min_or_max([(self.getAction(s[1], next_agent, depth - 1), s[1], s[0]) for s in successors],\
         key = lambda s: s[0])
        return mini_max_val[2] if depth == (self.depth*num_agents) and agent == 0 else mini_max_val[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState, agent = 0, depth = None):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        eval_fn = lambda action: self.evaluationFunction(gameState.generateSuccessor(agent, action))
        num_agents = gameState.getNumAgents()
        depth = depth if depth != None else self.depth * num_agents
        next_agent = (agent + 1)%num_agents
        actions = gameState.getLegalActions(agent)

        if not actions or depth == 0:
            return float(self.evaluationFunction(gameState))

        successors = [(action, gameState.generateSuccessor(agent, action)) for action in actions]

        if agent == 0:
            expectimax = max([(self.getAction(s[1], next_agent, depth-1), s[1], s[0]) for s in successors],\
             key = lambda s: s[0])
            return expectimax[2] if depth == (self.depth * num_agents) else expectimax[0]

        return sum([self.getAction(s[1], next_agent, depth - 1) for s in successors])/float(len(actions))


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).

      DESCRIPTION: <write something here so we know what you did>
    """
    r = random.Random()

    def eat_food(action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newFood = successorGameState.getFood().asList()
        oldFood = currentGameState.getFood().asList()
        return (len(oldFood) > len(newFood)) * 10000 * scoreEvaluationFunction(successorGameState)

    def closest_food(action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newFood = successorGameState.getFood()
        newPos = successorGameState.getPacmanPosition()
        food = max([util.manhattanDistance(food, newPos) for food in newFood.asList()]) if newFood.asList() else 0
        return -200*food* scoreEvaluationFunction(successorGameState)

    def finna_die(action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        for state in newGhostStates:
            if state.getPosition() == newPos and not state.scaredTimer > 0:
                return -float("inf")
        return False

    def ghost_buster(action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()
        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        for state in newGhostStates:
            if state.getPosition() == newPos and state.scaredTimer > 0:
                return scoreEvaluationFunction(successorGameState)
        return False

    actions = currentGameState.getLegalPacmanActions()
    eating = [eat_food(a) + finna_die(a) + closest_food(a) for a in actions]
    # eat_bust_die = [(eat_food(a),finna_die(a),ghost_buster(a)) for a in actions]
    # ebd = [e*1000-d*100000000+100000*g for (e,d,g) in eat_bust_die]
    averager = len(actions) if actions else 1
    # val = sum(ebd)/averager
    # print sum(eating)
    return sum(eating)/averager

# Abbreviation
better = betterEvaluationFunction
