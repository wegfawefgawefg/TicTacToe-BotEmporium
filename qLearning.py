import tictactoe as tt
import random
from tqdm import tqdm
import math
from pprint import pprint
import pickle

'''
assert that tallying end game states is correct
to increase performance
    play with alpha
    play with epsilon growth function
    rotational and reflectional symmetry hash equivalency, store rotation modifier
'''

def scoreEndBoard(board, winner, myPlayer):
    if not winner:
        return 1
    elif winner == tt.togglePlayer(myPlayer):
        return -10
    elif winner == myPlayer:
        return 10

def genQTable():
    qTable = [  [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]
    return qTable

def pickBestNextMove(qTables, keysSoFar, board, player, myPlayer, tryHard, verbose=False):
    boardKey = (tt.hash(board), player)
    empties = tt.listEmpties(board)

    if boardKey in qTables:
        keysSoFar.append(boardKey)
        qTable = qTables[boardKey]

        if verbose:
            pprint(qTable)

        tryHardOrNot = random.random()
        if tryHardOrNot > tryHard:
            return random.choice(empties)
        else:
            greatest = -math.inf
            bestEmpty = None
            for empty in empties:
                x = empty[0]
                y = empty[1]
                val = qTable[y][x]
                if val > greatest:
                    greatest = val
                    bestEmpty = empty
            return bestEmpty        

        #   FUTURE FEATURE:
        #   #   take into account the tryhard number
    else:
        qTables[boardKey] = genQTable()
        keysSoFar.append(boardKey)
        return random.choice(empties)

def updateQTable(score, qTables, keysSoFar, movesSoFar, alpha):
    #   FUTURE FEATURE:
    #   #   alpha, future vs current knkowledge

    #   propogate learned value back through the q tables of each game board
    for key, move in reversed(list(zip(keysSoFar, movesSoFar))):
        qTable = qTables[key]
        x, y = move[0], move[1]
        current = qTable[y][x]
        
        deltaNew = score * alpha
        deltaOld = (1-alpha) * current
        newVal = deltaOld + deltaNew
        qTable[y][x] = newVal

        score = newVal

def train(qTables, numGames, alpha, tryHard):
    tryHardGrowth = (1 - tryHard) / numGames

    for i in tqdm(range(numGames)):
        board = tt.genBoard()
        movesLeft = True
        winner = False
        player = 2
        keysSoFar = []
        movesSoFar = []

        computersPlayer = random.randint(1,2)
        while(movesLeft and not winner):
            if player == computersPlayer:
                bestMove = pickBestNextMove(qTables, keysSoFar, board, player, computersPlayer, tryHard)
                movesSoFar.append(bestMove)
                tt.applyMove(player, bestMove, board)
            else:
                moves = tt.listEmpties(board)
                randomMove = random.choice(moves)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)

            winner = tt.getWinner(board)
            movesLeft = not tt.noMoreMoves(board)

        score = scoreEndBoard(board, winner, computersPlayer)
        updateQTable(score, qTables, keysSoFar, movesSoFar, alpha)
        tryHard = tryHard + tryHardGrowth
        # print(tryHard)

def test(qTables, numGames, tryHard=1.0):
    numWins = 0
    numTies = 0
    numLosses = 0

    for i in tqdm(range(numGames)):
        board = tt.genBoard()
        movesLeft = True
        winner = False
        player = 2
        keysSoFar = []
        movesSoFar = []

        computersPlayer = random.randint(1,2)
        while(movesLeft and not winner):
            if player == computersPlayer:
                bestMove = pickBestNextMove(qTables, keysSoFar, board, player, computersPlayer, tryHard)
                movesSoFar.append(bestMove)
                tt.applyMove(player, bestMove, board)
            else:
                moves = tt.listEmpties(board)
                randomMove = random.choice(moves)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)

            winner = tt.getWinner(board)
            movesLeft = not tt.noMoreMoves(board)

        if winner == computersPlayer:
            numWins += 1
        elif winner == tt.togglePlayer(computersPlayer):
            numLosses += 1
        else:   #   tie
            numTies += 1

    return numWins, numLosses, numTies


def playGame():
    saveQTables = False
    fileName = 'qTables.pickle'

    qTables = {}


    keysSoFar = []
    movesSoFar = []
    tryHard = 0
    alpha = 0.9
    numTrials = 1000000

    if saveQTables:
        train(qTables, numTrials, alpha, tryHard)
        f = open(fileName, 'wb')
        pickle.dump(qTables, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        f = open(fileName, 'rb')
        qTables = pickle.load(f)
        f.close()
    
    numWins, numLosses, numTies = test(qTables, 1000)
    print("VS RANDOM OPPONENT...")
    print("numWins:"  + str(numWins))
    print("numLosses:"  + str(numLosses))
    print("numTies:"  + str(numTies))
    quit()



    board = tt.genBoard()
    movesLeft = True
    winner = False
    player = 2
    computersPlayer = random.randint(1,2)



    print("NEW GAME")
    if computersPlayer == 2:
        print("COMPUTER GOES FIRST...")
    while(movesLeft and not winner):
        if player == 2:
            print("X's Turn")
        else: # player == 1
            print("O's Turn")
        tt.printBoard(board)

        if player == computersPlayer:
            bestMove = pickBestNextMove(qTables, keysSoFar, board, player, computersPlayer, tryHard=1.0, verbose=True)
            movesSoFar.append(bestMove)
            tt.applyMove(player, bestMove, board)
            player = tt.togglePlayer(player)
        elif player == tt.togglePlayer(computersPlayer):
            validMove = False
            while validMove == False:
                move = input("input move of form 'y x' ")
                y = int(move[0])
                x = int(move[2])
                #   validate move
                if board[y][x] is not 0:
                    print("!!!INVALID MOVE!!!")
                    continue
                else:
                    validMove = True
                board[y][x] = tt.togglePlayer(computersPlayer)
                player = tt.togglePlayer(player)
        
        winner = tt.getWinner(board)
        movesLeft = not tt.noMoreMoves(board)

    tt.printBoard(board)
    
    score = scoreEndBoard(board, winner, computersPlayer)
    updateQTable(score, qTables, keysSoFar, movesSoFar, alpha)
    for key in keysSoFar:
        pprint(key)
        pprint(qTables[key])

    if winner:
        if winner == 2:
            print("WINNER: X")
        else: # winner == 1
            print("WINNER: O")
    else:
        print("TIE")

def main():
    playGame()

if __name__ == '__main__':
    main()