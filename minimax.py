import tictactoe as tt
import random
from tqdm import tqdm

def scoreEndBoard(board, winner, myPlayer):
    if not winner:
        return 0
    elif winner == tt.togglePlayer(myPlayer):
        return -1
    elif winner == myPlayer:
        return 1
    
def minimax(board, depth, maximizing, player, myPlayer):
    winner = tt.getWinner(board)
    if winner:
        return scoreEndBoard(board, winner, myPlayer)
    elif tt.noMoreMoves(board):
        return scoreEndBoard(board, winner, myPlayer)
    else:
        nextBoards = tt.listNextBoards(board, tt.togglePlayer(player))
        if maximizing:
            bestScore = -10000
            for nextBoard in nextBoards:
                score = minimax(nextBoard, 
                                depth=depth+1, 
                                maximizing=not maximizing,
                                player=tt.togglePlayer(player),
                                myPlayer=myPlayer)
                if score > bestScore:
                    bestScore = score
            return bestScore
        else:   #   not maximizing
            bestScore = 10000
            for nextBoard in nextBoards:
                score = minimax(nextBoard, 
                                depth=depth+1, 
                                maximizing=maximizing,
                                player=tt.togglePlayer(player),
                                myPlayer=myPlayer)
                if score < bestScore:
                    bestScore = score
            return bestScore


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
            nextBoards = tt.listNextBoards(board, computersPlayer)
            bestBoard = None
            bestScore = -10000
            for nextBoard in tqdm(nextBoards):
                score = minimax(nextBoard, 
                                depth=turn, 
                                maximizing=True,
                                player=player,
                                myPlayer=computersPlayer)
                if score > bestScore:
                    bestScore = score
                    bestBoard = nextBoard
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