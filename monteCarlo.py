import tictactoe as tt
import random
from tqdm import tqdm
import math
from pprint import pprint
import pickle
import copy

'''
selection
expansion
simulation
backup
'''

'''
BUGS
not simulating vs human player
100% winrate vs random. seems impossible...
'''

def scoreEndBoard(board, winner, myPlayer):
    if not winner:
        return 0
    elif winner == tt.togglePlayer(myPlayer):
        return -1
    elif winner == myPlayer:
        return 1

def genFullKey(board, player):
    return (tt.hash(board), player)

def simulate(numSimulations, board, player, myPlayer):
    originBoard = copy.deepcopy(board)
    originPlayer = player
    totalScore = 0
    for i in range(numSimulations):
        simBoard = copy.deepcopy(originBoard)
        simPlayer = originPlayer

        winner = tt.getWinner(simBoard)
        movesLeft = not tt.noMoreMoves(simBoard)
        while(movesLeft and not winner):
            moves = tt.listEmpties(simBoard)
            randomMove = random.choice(moves)
            tt.applyMove(simPlayer, randomMove, simBoard)
            simPlayer = tt.togglePlayer(simPlayer)

            winner = tt.getWinner(simBoard)
            movesLeft = not tt.noMoreMoves(simBoard)
        
        score = scoreEndBoard(simBoard, winner, myPlayer)
        totalScore += score

    return totalScore


def expand(mct, board, player):
    nextBoards = tt.listNextBoards(board, player)
    for nextBoard in nextBoards:
        nextBoardKey = genFullKey(nextBoard, tt.togglePlayer(player))
        if nextBoardKey not in mct: #   there might be convergent branches
            vals = {'n':0, 'v':0 }
            mct[nextBoardKey] = vals

def simChildrenInner(mct, nextBoard, player, myPlayer, numSims):
    nextBoardKey = genFullKey(nextBoard, tt.togglePlayer(player))
    if nextBoardKey not in mct:
        mct[nextBoardKey] = 0

    #   give each one a bunch of game sims
    simScoreTotal = simulate(numSims, nextBoard, tt.togglePlayer(player), myPlayer)
    mct[nextBoardKey] += simScoreTotal

def simulateChildren(mct, board, player, myPlayer, numSims, verbose=False):
    key = genFullKey(board, player)
    if key not in mct:
        mct[key] = 0

    nextBoards = tt.listNextBoards(board, player)

    winner = tt.getWinner(board)
    if not winner and nextBoards:
        #   add next boards to mct
        if verbose:
            for nextBoard in tqdm(nextBoards):
                simChildrenInner(mct, nextBoard, player, myPlayer, numSims)
        else:
            for nextBoard in nextBoards:
                simChildrenInner(mct, nextBoard, player, myPlayer, numSims)

        #   pick the highest score and return that
        scores = [mct[genFullKey(nextBoard, tt.togglePlayer(player))] for nextBoard in nextBoards]
        highestScore = sorted(scores)[-1]
        return highestScore
    else:
        score = scoreEndBoard(board, tt.togglePlayer(player), myPlayer)
        score = score * math.pow(numSims, 2 )
        mct[key] = score 
        return score

def pickBestNextMove(mct, board, player):
    nextBoards = tt.listNextBoards(board, player)
    bestBoard = None
    highest = -math.inf
    for nextBoard in nextBoards:
        nextBoardKey = genFullKey(nextBoard, tt.togglePlayer(player))
        score = mct[nextBoardKey]
        if score > highest:
            highest = score
            bestBoard = nextBoard
    return bestBoard        

def test(mct, numGames, numSims):
    numWins = 0
    numTies = 0
    numLosses = 0

    for i in tqdm(range(numGames)):
        board = tt.genBoard()
        movesLeft = True
        winner = False
        player = 2

        computersPlayer = random.randint(1,2)
        while(movesLeft and not winner):
            if player == computersPlayer:
                simulateChildren(mct, board, player, computersPlayer, numSims)
                bestBoard = pickBestNextMove(mct, board, player)
                # print("################")
                # tt.printBoard(board)
                # tt.printBoard(bestBoard)
                # print("BESTMOVE")
                board = bestBoard
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
    mct = {}
    board = tt.genBoard()
    player = 2
    computersPlayer = 2
    numSimsPreGame = 100000
    numSimsOnline = 100

    saveMCTree = False
    fileName = 'mct.pickle'

    if saveMCTree:
        simulateChildren(mct, board, player, computersPlayer, numSimsPreGame, verbose=True)
        f = open(fileName, 'wb')
        pickle.dump(mct, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        quit()
    else:
        f = open(fileName, 'rb')
        mct = pickle.load(f)
        f.close()
    
    # mct = {}
    # numTrials = 100
    # numWins, numLosses, numTies = test(mct, numTrials, numSimsOnline)
    # print("VS RANDOM OPPONENT...")
    # print("numWins:"  + str(numWins))
    # print("numLosses:"  + str(numLosses))
    # print("numTies:"  + str(numTies))
    # quit()

    #   w 0.6, t 0.11, l 0.3


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
            simulateChildren(mct, board, player, computersPlayer, numSimsOnline, verbose=True)
            bestBoard = pickBestNextMove(mct, board, player)
            board = bestBoard
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