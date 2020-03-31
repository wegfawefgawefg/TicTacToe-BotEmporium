import tictactoe as tt
import random
from tqdm import tqdm
import math

'''
KNOWN BUG:
ONLY WORKS IF COMPUTER IS X, MOTHER FUCKER:
    STRATEGY:
        REPLACE PLAYER==COMPUTERSpLAYER WITH MAXIMIZING SWAP DEPENDING ON TURN
'''



def scoreEndBoard(board, winner, myPlayer):
    if not winner:
        return 0
    elif winner == tt.togglePlayer(myPlayer):
        return -1
    elif winner == myPlayer:
        return 1
    
def minimax(totalCount, board, player, myPlayer):
    prevBoards = {}
    count = [0]
    
    alpha = -math.inf
    beta = math.inf
    score = minimax_inner(count, board, player, myPlayer, prevBoards, alpha, beta)
    totalCount[0] += count[0]
    print("\nboards evaluated: " + str(count[0]))
    return score

def minimax_inner(count, board, player, myPlayer, prevBoards, alpha, beta):
    hashKey = tt.hash(board)
    if hashKey in prevBoards:
        return prevBoards[hashKey]

    count[0] += 1

    winner = tt.getWinner(board)
    if winner:
        score = scoreEndBoard(board, winner, myPlayer)
        # prevBoards[hashKey] = score
        return score
    elif tt.noMoreMoves(board):
        score = scoreEndBoard(board, winner, myPlayer)
        # prevBoards[hashKey] = score
        return score    
    else:
        nextBoards = tt.listNextBoards(board, tt.togglePlayer(player))
        if player == myPlayer:  #   maximizing next moves
            bestScore = -math.inf
            for nextBoard in nextBoards:
                if beta <= alpha:
                    break
                score = minimax_inner(count,
                    nextBoard, 
                    player=tt.togglePlayer(player),
                    myPlayer=myPlayer,
                    prevBoards=prevBoards,
                    alpha=alpha, beta=beta)
                prevBoards[tt.hash(nextBoard)] = score
                if score > bestScore:
                    bestScore = score
                    alpha = bestScore
            return bestScore
        else:   #   minimizing next moves
            bestScore = math.inf
            for nextBoard in nextBoards:
                if beta <= alpha:
                    break              
                score = minimax_inner(count,
                    nextBoard, 
                    player=tt.togglePlayer(player),
                    myPlayer=myPlayer,
                    prevBoards=prevBoards,
                    alpha=alpha, beta=beta)
                prevBoards[tt.hash(nextBoard)] = score
                if score < bestScore:
                    bestScore = score
                    beta = bestScore
            return bestScore

def pickBestNextBoard(board, player, myPlayer):
    totalCount = [0]

    nextBoards = tt.listNextBoards(board, myPlayer)
    bestBoard = None
    bestScore = -10000
    for nextBoard in tqdm(nextBoards):
        score = minimax(totalCount,
                        nextBoard, 
                        player=tt.togglePlayer(player),
                        myPlayer=myPlayer)
        if score > bestScore:
            bestScore = score
            bestBoard = nextBoard
    print("\ntotal boards evaluated: " + str(totalCount[0]))
    return bestBoard

def playGame():
    board = tt.genBoard()
    movesLeft = True
    winner = False
    player = 2
    computersPlayer = 2 #random.randint(1,2)
    turn = 0

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
            board = pickBestNextBoard(board, player, computersPlayer)
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
        turn += 1
        
        winner = tt.getWinner(board)
        movesLeft = not tt.noMoreMoves(board)

    tt.printBoard(board)

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