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
    return logic.Expr("|", *[logic.Expr("&", expressions[i], \
    *[logic.Expr("~", expr) for expr in expressions[:i]+expressions[i+1:]]) for i in xrange(len(expressions))])


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
    def foodGhostHeuristic(state, problem, time):
        ghostPositions = [getGhostPosition(time, problem, pos.getPosition()) for pos in problem.getGhostStartStates()]
        print ghostPositions
        print state[0]
        for ghostPos in ghostPositions:
            if state[0] == ghostPos:
                return 999999
        food_l = [util.manhattanDistance(food, state[0]) for food in state[1].asList()]
        if not food_l:
            return 0
        return max(food_l)

    return aStarSearch(problem, foodGhostHeuristic)

def getGhostPosition(time, foodGhostProblem, startPos):
    leftWall, row = startPos
    rightWall = startPos[0]
    while not foodGhostProblem.isWall((leftWall, row)):
        leftWall -= 1
    while not foodGhostProblem.isWall((rightWall, row)):
        rightWall += 1
    width = rightWall-leftWall-1
    start = startPos[0] - leftWall - 1
    curr_loop = time % (2 * (width - 1))
    if curr_loop <= (start * 2) and curr_loop >= start:
        return (curr_loop - start + leftWall, row)
    if curr_loop < start:
        return (leftWall + (start - curr_loop), row)
    curr_loop -= (start * 2)
    start = startPos[0]
    to_right = rightWall - startPos[0] - 1
    if curr_loop <= to_right:
        return (start + curr_loop, row)
    return (rightWall - (curr_loop - to_right + 1), row)



def aStarSearch(problem, heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    possible_actions = [game.Directions.NORTH, game.Directions.SOUTH, game.Directions.EAST, game.Directions.WEST]
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0, -1), priority = heuristic(problem.getStartState(), problem = problem, time = 0))
    expanded = set()
    while not frontier.isEmpty():
        #items in queue have form (state, expressions_to_state, step_cost, time)
        (state, exps, score, time) = frontier.pop()
        if state in expanded:
            continue
        if problem.terminalTest(state):
            model = logic.pycoSAT([logic.Expr("&", *exps)])
            if model:
                return extractActionSequence(model, possible_actions)
            return []
        expanded.add(state)
        successors = getActionsAndState(problem, state, exps, time+1)
        for successor in successors:
            s_state, s_exps = successor
            s_h = score + 1 + heuristic(s_state, problem = problem, time = time)
            frontier.push((s_state, s_exps, s_h, time+1), priority = s_h)
    return []

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
