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

def sortBoardsByVisits(mct, nextBoards, player):
    #   might need to replace this with tuples, 
    #   #   lookups can step on eachother

    visitsToBoard = {}
    visits = []
    for nextBoard in nextBoards:
        nextBoardKey = genFullKey(nextBoard, player)
        visitCount = mct[nextBoardKey]['n']
        visitsToBoard[visitCount] = tt.hash(nextBoard)
        visits.append(visitCount)

    sortedVisitCounts = sorted(visits)
    sortedBoards = [visitsToBoard[visitCount] for visitCount in sortedVisitCounts]
    unhashed = [tt.unHash(board) for board in sortedBoards]
    return unhashed

def explore(numExplores, count, mct, board, player, myPlayer, depth, numSimsAtDepth, numSimsAtLeaf):
    count[0] += 1
    if count[0] % 10000 == 0:
        print("%: " + str(count[0] / numExplores * 10.0))

    #   gen board entries and children
    key = genFullKey(board, player)
    if key not in mct:
        mct[key] = {'n':0, 'v':0 }
    expand(mct, board, player)

    if depth >= 0:
        nextBoards = tt.listNextBoards(board, player)
        winner = tt.getWinner(board)
        if not winner and nextBoards:
            #   keep expanding
            childScores = []
            for nextBoard in nextBoards:
                highestScore = explore(numExplores, count, mct, nextBoard, tt.togglePlayer(player), myPlayer, depth-1, numSimsAtDepth, numSimsAtLeaf)
                childScores.append(highestScore)
            
            if nextBoards:
                highestScore = sorted(childScores)[-1]
                mct[key]['v'] = highestScore
                return highestScore
        else:
            score = scoreEndBoard(board, tt.togglePlayer(player), myPlayer)
            score = score * math.pow(numSimsAtLeaf * numSimsAtDepth, 2 )
            mct[key]['v'] = score 
            return score
    else:
        nextBoards = tt.listNextBoards(board, player)
        winner = tt.getWinner(board)
        if not winner and nextBoards:
            #   sort the next boards by score
            #   simulate randomly
            nextBoards = tt.listNextBoards(board, player)
            for i in range(numSimsAtDepth):
                boardsLToGByScore = sortBoardsByVisits(mct, nextBoards, tt.togglePlayer(player))
                leastVisited = boardsLToGByScore[0]
                simScoreTotal = simulate(numSimsAtLeaf, leastVisited, tt.togglePlayer(player), myPlayer)
                lvbkey = genFullKey(leastVisited, tt.togglePlayer(player))
                mct[lvbkey]['v'] += simScoreTotal
                mct[lvbkey]['n'] += numSimsAtLeaf

            #   pick the highest score and return that
            scores = [mct[genFullKey(nextBoard, tt.togglePlayer(player))]['v'] for nextBoard in nextBoards]
            highestScore = sorted(scores)[-1]
            return highestScore
        else:
            print("checking end boards")
            score = scoreEndBoard(board, tt.togglePlayer(player), myPlayer)
            score = score * math.pow(numSimsAtLeaf * numSimsAtDepth, 2 )
            mct[key]['v'] = score 
            return score

def pickBestNextMove(mct, board, player):
    nextBoards = tt.listNextBoards(board, player)
    bestBoard = None
    greatestVal = -math.inf
    for nextBoard in nextBoards:
        nextBoardKey = genFullKey(nextBoard, tt.togglePlayer(player))
        vals = mct[nextBoardKey]
        v = vals['v']
        if v > greatestVal:
            greatestVal = v
            bestBoard = nextBoard
    return bestBoard        


def test(mct, numGames):
    numWins = 0
    numTies = 0
    numLosses = 0

    for i in tqdm(range(numGames)):
        board = tt.genBoard()
        movesLeft = True
        winner = False
        player = 2

        depth = 1
        numExplores = 9 ** depth 
        count = [0]

        computersPlayer = random.randint(1,2)
        while(movesLeft and not winner):
            if player == computersPlayer:
                explore(numExplores,
                    count=count, 
                    mct=mct, 
                    board=board, 
                    player=player, 
                    myPlayer=computersPlayer, 
                    depth=depth, 
                    numSimsAtDepth=9, 
                    numSimsAtLeaf=1)
                bestBoard = pickBestNextMove(mct, board, player)
                board = bestBoard
                player = tt.togglePlayer(player)
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

    depth = 9
    numExplores = 9 ** depth 
    count = [0]

    saveMCTree = False
    fileName = 'mct.pickle'

    if saveMCTree:
        explore(numExplores,
        count=count, 
        mct=mct, 
        board=board, 
        player=player, 
        myPlayer=computersPlayer, 
        depth=depth, 
        numSimsAtDepth=10, 
        numSimsAtLeaf=10)
        f = open(fileName, 'wb')
        pickle.dump(mct, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        f = open(fileName, 'rb')
        mct = pickle.load(f)
        f.close()
    

    # numTrials = 100
    # numWins, numLosses, numTies = test(mct, numTrials)
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


    playDepth = 2

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
            explore(numExplores,
            count=count, 
            mct=mct, 
            board=board, 
            player=player, 
            myPlayer=computersPlayer, 
            depth=playDepth, 
            numSimsAtDepth=10, 
            numSimsAtLeaf=10)

            bestBoard = pickBestNextMove(mct, board, player)
            # tt.applyMove(player, bestMove, board)
            board = bestBoard
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