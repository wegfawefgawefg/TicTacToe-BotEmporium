import tictactoe as tt
import random
from tqdm import tqdm
import math
from pprint import pprint
import pickle

'''
selection
expansion
simulation
backup
'''

def genFullKey(board, player):
    return (tt.hash(board), player)

def simulate(board, player):
    #   while game is not done, pick random moves until game done
    #   return end result of board, or result if evaluation ended early
    pass

def train(mct, numTrials):
    board = tt.genBoard()
    player = 2

    key = genFullKey(board, player)

    for i in range(0, numTrials):
        if key in mct:  #   ~SELECTION~
            nextBoards = tt.listNextBoards(board, player)
            #   do we pick the highest? or lowest...
            lowest = -math.inf
            for nextBoard in nextBoards:
                nextBoardKey = genFullKey(nextBoard, player)
                vals = mct[nextBoardKey]
                n, v = vals['n'], vals['v']
                #   total the number of times its children were visited
                

        #   #   compute the val for each child
        #   #   print those values and bail out
        else:           #   ~EXPANSION~
            #   generate children if and init vals
            nextBoards = tt.listNextBoards(board, player)
            for nextBoard in nextBoards:
                nextBoardKey = genFullKey(nextBoard, player)
                if nextBoardKey not in mct: #   there might be convergent branches
                    vals = {'n':0, 'v':0}
                    mct[nextBoardKey] = vals
            #   #   simulate one at random  ~SIMULATION~
            randChild = random.choice(nextBoards)
            score = simulate(randChild, tt.togglePlayer(player))
            vals = mct[nextBoardKey]
            vals['n'] += 1
            vals['v'] += score

def playGame():
    saveMCTree = False
    fileName = 'mct.pickle'

    mct = {}
    numTrials = 5

    if saveMCTree:
        train(mct, numTrials)
        f = open(fileName, 'wb')
        pickle.dump(mcts, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        f = open(fileName, 'rb')
        qTables = pickle.load(f)
        f.close()
    
    # numWins, numLosses, numTies = test(mcts, numTrials)
    # print("VS RANDOM OPPONENT...")
    # print("numWins:"  + str(numWins))
    # print("numLosses:"  + str(numLosses))
    # print("numTies:"  + str(numTies))
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