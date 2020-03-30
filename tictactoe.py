from copy import copy, deepcopy
from pprint import pprint
import random
import math

def applyMove(player, move, board):
    board[move[1]][move[0]] = player

def genBoard():
    # board = [[0, 2, 0],
    #          [0, 2, 0],
    #          [0, 0, 0]]
    board = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]
    return board

def listEmpties(board):
    empties = []
    for y, row in enumerate(board):
        for x, col in enumerate(row):
            if col == 0:
                empties.append((x, y))
    return empties
    
def listNextBoards(board, player):
    empties = listEmpties(board)
    nextBoards = []
    for empty in empties:
        newBoard = deepcopy(board)
        x = empty[0]
        y = empty[1]
        newBoard[y][x] = player
        nextBoards.append(newBoard)
    return nextBoards

#   check if winner
def getWinner(board):
    diagPositions = [
        [   [0, 0],
            [1, 1],
            [2, 2]],
        [   [0, 2],
            [1, 1],
            [2, 0]]]

    for player in [1, 2]:
        #   check horizontal
        for row in board:
            count = 0
            for col in row:
                if col == player:
                    count += 1
            if count == 3:
                return player
        #   check vertical
        for c in range(0, 3):
            count = 0
            for r in range(0, 3):
                piece = board[r][c]
                if piece == player:
                        count += 1
            if count == 3:
                return player


        for direction in diagPositions:
            count = 0
            for pos in direction:
                x = pos[0]
                y = pos[1]
                piece = board[y][x]
                if piece == player:
                        count += 1
            if count == 3:
                return player
    return False

def printBoard(board):
    print("= = =")
    for i, row in enumerate(board):
        if not i == 0:
            print()
        for col in row:
            piece = '-'
            if col == 1:
                piece = 'O'
            elif col == 2:
                piece = 'X'
            print(piece + " ",end='')
    print()
    print("= = =")

def noMoreMoves(board):
    empties = listEmpties(board)
    if len(empties) == 0:
        return True
    else:
        return False

def scoreEndBoard(board, depth, myPlayer):
    winner = getWinner(board)
    if not winner:
        return -100 * math.pow(2, 9 - depth)    #   tie is always depth 9..
    elif winner == togglePlayer(myPlayer):
        return -1000 * math.pow(2, 9 - depth)
    elif winner == myPlayer:
        return 100000000 / depth #/ math.pow(2, depth)

def hash(board):
    hashKey = []
    for row in board:
        for col in row:
            hashKey.append(str(col))
    hashKey = ''.join(hashKey)
    return hashKey

def unHash(hash):
    boardConcat = []
    for c in hash:
        boardConcat.append(int(c))
    board = [boardConcat[0:3], boardConcat[3:6], boardConcat[6:9]]
    return board

def togglePlayer(player):
    if player == 2:
        return 1
    else:
        return 2

def getBoardScores(inBoard, myPlayer, inPlayer, depth=0):
    stack = []
    boardScores = {}

    stack.append(
        {'board':inBoard,
         'player':inPlayer,
         'depth':depth}
    )

    while len(stack) > 0:
        args = stack[-1]
        board = args['board']
        player = args['player']
        depth = args['depth']

        #   base case, end board
        winner = getWinner(board)
        if winner:
            score = scoreEndBoard(board, depth, myPlayer)
            boardScores[hash(board)] = score
            stack.pop()
        elif noMoreMoves(board):
            boardScores[hash(board)] = scoreEndBoard(board, depth, myPlayer)
            stack.pop()

        else:   #   nobody won yet, and there are move moves
            nextBoards = listNextBoards(board, player)
            allPresent = True
            score = 0
            for nextBoard in nextBoards:
                if hash(nextBoard) in boardScores:
                    score += boardScores[hash(nextBoard)]
                else:
                    allPresent = False
                    newArgs = { 'board':nextBoard, 
                                'player':togglePlayer(player),
                                'depth':depth + 1}
                    stack.append(newArgs)
            if allPresent:
                boardScores[hash(board)] = score
                stack.pop()

    return boardScores

def genBoardScores(myPlayer, currentTurnPlayer):
    emptyBoard = genBoard()
    boardScores = getBoardScores(emptyBoard, myPlayer=myPlayer, inPlayer=currentTurnPlayer)
    return boardScores

def pickBestNextBoard(boardScores, inBoard, myPlayer):
    nextBoards = listNextBoards(inBoard, myPlayer)
    nextBoardScores = {}
    for nextBoard in nextBoards:
        nextBoardScores[hash(nextBoard)] = boardScores[hash(nextBoard)]
    
    bestBoard = None
    bestScore = -1000000
    for board, score in nextBoardScores.items():
        if score > bestScore:
            bestBoard = board
            bestScore = score

    return unHash(bestBoard)

#####################################################
#                   main stoff                      #
#####################################################
def playGame():
    #   generate board
    board = genBoard()
    #   while game not won and moves left
            #   make move or ask me to move
    movesLeft = True
    winner = False
    player = 2
    computersPlayer = random.randint(1,2)

    boardScores = genBoardScores(myPlayer=computersPlayer, currentTurnPlayer=player)

    print("NEW GAME")
    if computersPlayer == 2:
        print("COMPUTER GOES FIRST...")
    while(movesLeft and not winner):
        if player == 2:
            print("X's Turn")
        else: # player == 1
            print("O's Turn")
        printBoard(board)

        if player == computersPlayer:
            bestNextBoard = pickBestNextBoard(boardScores, board, computersPlayer)
            board = bestNextBoard
            player = togglePlayer(player)
        elif player == togglePlayer(computersPlayer):
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
                board[y][x] = togglePlayer(computersPlayer)
                player = togglePlayer(player)

        
        winner = getWinner(board)
        movesLeft = not noMoreMoves(board)

    printBoard(board)

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