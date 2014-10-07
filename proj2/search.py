# search.py
# ---------
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


# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import logic
import game

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostSearchProblem)
        """
        util.raiseNotDefined()

    def terminalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionSearchProblem
        """
        util.raiseNotDefined()

    def result(self, state, action):
        """
        Given a state and an action, returns resulting state and step cost, which is
        the incremental cost of moving to that successor.
        Returns (next_state, cost)
        """
        util.raiseNotDefined()

    def actions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        util.raiseNotDefined()

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        util.raiseNotDefined()

    def isWall(self, position):
        """
        Return true if position (x,y) is a wall. Returns false otherwise.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def atLeastOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at least one of the expressions in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> print logic.pl_true(atleast1,model2)
    True
    """
    return logic.Expr("|", *expressions)


def atMostOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at most one of the expressions in the list is true.
    """
    exact = exactlyOne(expressions)
    none = logic.Expr("&",logic.Expr("~", *expressions))
    return logic.Expr("|", none, exact)


def exactlyOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that exactly one of the expressions in the list is true.
    """
    return logic.Expr("|", *[logic.Expr("&", expr_i, \
        *[logic.Expr("~", expr_j) for expr_j in expressions if expr_i != expr_j]) for expr_i in expressions])


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print plan
    ['West', 'South', 'North']
    """
    plan = []
    i = 0
    while True:
        true_actions = [a for a in actions if logic.PropSymbolExpr(a, i) in model and model[logic.PropSymbolExpr(a, i)]]
        if not true_actions:
            break
        plan.append(true_actions[0])
        i += 1
    return plan


def positionLogicPlan(problem):
    """
    Given an instance of a PositionSearchProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    possible_actions = [game.Directions.NORTH, game.Directions.SOUTH, game.Directions.EAST, game.Directions.WEST]
    time = 0
    next_action_states = getActionsAndState(problem, problem.getStartState(), [], 0)
    terminal_state = containsGoalState(problem, [s[0] for s in next_action_states])
    expanded = set([problem.getStartState()])

    while not terminal_state:
        time += 1
        next_action_states = reduce(lambda x,y: x + y, [getActionsAndState(problem,s[0],s[1],time) for s in next_action_states])
        next_action_states = filter(lambda s: s[0] not in expanded, next_action_states)
        for s in next_action_states:
            expanded.add(s[0])
        terminal_state = containsGoalState(problem, [s[0] for s in next_action_states])
    sol_state = [problem.terminalTest(s[0]) for s in next_action_states].index(True)
    print "trying to find model"
    print "sol_state:", next_action_states[sol_state][1]
    model = logic.pycoSAT([logic.Expr("&", *next_action_states[sol_state][1])])
    print model
    if model:
        return extractActionSequence(model, possible_actions)
    return []

def getActionsAndState(problem, state, actions=[], time=0):
    expr_and_a = [(a, logic.PropSymbolExpr(a, time)) for a in problem.actions(state)]
    actions_and_state = []
    for i in xrange(len(expr_and_a)):
        expr = logic.Expr("&", expr_and_a[i][1], \
            *[logic.Expr("~", a[1]) for a in expr_and_a if a[0] != expr_and_a[i][0]])
        actions_and_state += [(problem.result(state, expr_and_a[i][0])[0], actions + [expr])]
    return actions_and_state

def containsGoalState(problem, states):
    return sum([s == problem.getGoalState() for s in states]) > 0

def foodLogicPlan(problem):
    """
    Given an instance of a FoodSearchProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    def foodHeuristic(state, problem, time):
        food_l = [util.manhattanDistance(food, state[0]) for food in state[1].asList()]
        if not food_l:
            return 0
        return max(food_l)

    return aStarSearch(problem, foodHeuristic)

def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostSearchProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    an eastern wall.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    def foodGhostHeuristic(problem):
        # print "finding pos's"
        ghostPositions = [getGhostPositionArray(problem, pos.getPosition()) for pos in problem.getGhostStartStates()]
        print ghostPositions

        def h(state, problem, time):
            # print ghostPositions
            # print state[0]
            for ghostPos in ghostPositions:
                if state[0] == ghostPos[time%len(ghostPos)] or state[0] == ghostPos[(time-1)%len(ghostPos)]:
                    # print "about to hit a ghost"
                    return 99999999999
            food_l = [util.manhattanDistance(food, state[0]) for food in state[1].asList()]
            if not food_l:
                return 0
            return max(food_l)
        return h

    return aStarSearch(problem, foodGhostHeuristic(problem))

def getGhostPositionArray(foodGhostProblem, startPos):
    x,y = startPos
    pos_arr = [(x, y)]
    while not foodGhostProblem.isWall((x+1, y)):
        x += 1
        pos_arr.append((x,y))
        # print "loooooooop1"
    while not foodGhostProblem.isWall((x-1, y)):
        x -= 1
        pos_arr.append((x,y))
        # print "loooooooop2"
    while x + 1 < startPos[0]:
        x += 1
        pos_arr.append((x,y))
        # print x, startPos[0]
    pos_arr = [pos for pos in pos_arr] if pos_arr[0] != pos_arr[-1] and len(pos_arr) > 1 else pos_arr[:len(pos_arr)-1]
    return pos_arr



def aStarSearch(problem, heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    possible_actions = [game.Directions.NORTH, game.Directions.SOUTH, game.Directions.EAST, game.Directions.WEST]
    frontier = util.PriorityQueue()
    print problem.getStartState(), problem.getStartState()[1].asList()
    initial_score = heuristic(problem.getStartState(), problem = problem, time = 0)
    frontier.push((problem.getStartState(), [], initial_score, 0), priority = initial_score)
    expanded = set([(problem.getStartState(), 0)])
    while not frontier.isEmpty():
        #items in queue have form (state, expressions_to_state, step_cost, time)
        (state, exps, dist, time) = frontier.pop()
        # print state
        print dist, time, state
        if state in expanded or dist >= 99999999999:
            continue
        # print frontier.heap

        if problem.terminalTest(state):
            # print "score: ", score
            model = logic.pycoSAT([logic.Expr("&", *exps)])
            if model:
                return extractActionSequence(model, possible_actions)
            return []
        expanded.add((state, time))
        successors = getActionsAndState(problem, state, exps, time)
        for successor in successors:
            s_state, s_exps = successor
            s_h = dist + 1 + heuristic(s_state, problem = problem, time = time+1)
            # print s_h
            if s_h < 99999999999:
                frontier.push((s_state, s_exps, dist, time+1), priority = s_h)
    return []

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
