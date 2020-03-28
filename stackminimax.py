import tictactoe as tt
import random

def minimaxGenBoardScores(inBoard, myPlayer, inPlayer):
    stack = []
    boardScores = {}

    stack.append(
        {'board':inBoard,
         'player':inPlayer}
    )

    while len(stack) > 0:
        args = stack[-1]
        board = args['board']
        player = args['player']

        maximizing = False
        if player == myPlayer:
            maximizing = True

        #   base case, end board
        winner = tt.getWinner(board)
        if winner:
            boardScores[tt.hash(board)] = tt.scoreEndBoard(board, winner, myPlayer)
            print("###########")
            tt.printBoard(board)
            print(boardScores[tt.hash(board)])

            stack.pop()
        elif tt.noMoreMoves(board):
            boardScores[tt.hash(board)] = tt.scoreEndBoard(board, winner, myPlayer)
            stack.pop()

        else:   #   nobody won yet, and there are move moves
            nextBoards = tt.listNextBoards(board, player)
            allPresent = True
            for nextBoard in nextBoards:
                if not (tt.hash(nextBoard) in boardScores):
                    allPresent = False
                    newArgs = { 'board':nextBoard, 
                                'player':tt.togglePlayer(player),}
                    stack.append(newArgs)
            if allPresent:
                scores = [boardScores[tt.hash(nextBoard)] for board in nextBoards]
                if maximizing:
                    boardScores[tt.hash(board)] = max(scores)
                else:
                    boardScores[tt.hash(board)] = min(scores)    
                stack.pop()

    return boardScores