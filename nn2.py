import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import tictactoe as tt
from torch.autograd import Variable
import random

'''
VERSION 2.0:
add valid move masking in both move selection and error propogation
This should substantially clean up invalid loops and whipping, also will allow for easy dropout and a real deployable offline mode 
    -unlike v1, which required whipping so it needed to always be online
    -also offline mode means we can use dropout in training, and different architectures, such as some resnet
    -try adding residuals to the forward prop to make a resnet
'''

'''
add gpu training
add some graphing

train
test
play against it code

add some dropout
play with architecture


TODO:
finish the training function
    specifically:
        valid move punishing
        then list moves made
        based on end game result, punish each move relative to whether it resulted in a win or not

try making it so that the error is 0 for valid moves, but error of max for invalid moves
so probably this means the target is equal to the network output for valid moves
    wont this make a pressure towards 0 in almost every output?
    should we aim for 0.5 instead? 0.5 would be better maybe
        could try both i guess


maybe method of improvement
instead of teaching it valid moves,
only sample from the valid moves, so just set all unvalid moves in the output to 0 when you do the max
this should work. 
still send 0 through invalid moves though


'''

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu( self.l1(x) )
        x = F.relu( self.l2(x) )
        x = self.l3(x)
        return x
        # return F.softmax(x, dim=2)

def genBoard():
    board =  np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]], dtype=float)
    return board

def scoreEndBoard(board, winner, myPlayer):
    if not winner:
        return 0.5
    elif winner == tt.togglePlayer(myPlayer):
        return 0
    elif winner == myPlayer:
        return 1

'''
should it know whos turn it is?
'''
def oneHotTicTacToe(board, computersPlayer):
    me = np.where(board == computersPlayer, 1, 0)
    notMe = np.where(board == tt.togglePlayer(computersPlayer), 1, 0)
    me = me.flatten()
    notMe = notMe.flatten()
    oneHot = np.append(me, notMe)
    oneHot = torch.tensor(oneHot, dtype=torch.float32)
    return oneHot

#   def playGame()


#   def test()

'''
dont forget to let the computer be any player
'''
def trainToMakeValidMoves(net, criterion, optimizer, epochs):
    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = 2 #random.randint(1,2)

        optimizer.zero_grad()

        #   generate a random board
        board = np.random.randint(low = 0, high = 3, size = (3, 3))
        oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)

        validMoves = np.where(board == 0, 1, 0)
        target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)

        output = net(oneHot)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def trainAgainstSelf(net, criterion, optimizer, epochs):
    numInvalidMoves = 0
    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        optimizer.zero_grad()

        board = np.zeros(shape = (3, 3))
        # board = np.random.randint(low = 0, high = 3, size = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        movesA = []
        outputsA = []
        movesB = []
        outputsB = []
        while(not winner and movesLeft):
            move = None
            moveValid = False
            while not moveValid:
                #   generate a move
                if player == computersPlayer:
                    oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                else:
                    oneHot = oneHotTicTacToe(board, tt.togglePlayer(computersPlayer)).view(1, 1, 18)
                output = net(oneHot)
                values, index = output.view(9).max(0)
                if board.flatten()[index] == 0: #   if move is valid
                    moveValid = True

                    #   apply the move
                    move = index
                    board = board.flatten()
                    if player == computersPlayer:
                        board[move] = computersPlayer
                    else:
                        board[move] = tt.togglePlayer(computersPlayer)
                    board = board.reshape(3, 3)
                    
                    #   store for later
                    if player == computersPlayer:
                        movesA.append(move)
                        outputsA.append(output)
                    else:
                        movesB.append(move)
                        outputsB.append(output)
                else:   #   invalid move, prime the whip
                    # print("invalid move")
                    numInvalidMoves += 1
                    optimizer.zero_grad()
                    validMoves = np.where(board == 0, 1, 0)
                    target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            player = tt.togglePlayer(player)

            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)
        
        #   get end score of game

        score = scoreEndBoard(board, winner, computersPlayer)
        for i, move in enumerate(movesA):
            output = outputsA[i]
            target = output.clone().view(9)
            target[move] = score
            target = target.view(1, 1, 9)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        score = scoreEndBoard(board, winner, tt.togglePlayer(computersPlayer))
        for i, move in enumerate(movesB):
            output = outputsB[i]
            target = output.clone().view(9)
            target[move] = score
            target = target.view(1, 1, 9)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def train(net, criterion, optimizer, epochs):
    numInvalidMoves = 0
    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        optimizer.zero_grad()

        board = np.zeros(shape = (3, 3))
        # board = np.random.randint(low = 0, high = 3, size = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        moves = []
        outputs = []
        while(not winner and movesLeft):
            if player == computersPlayer:
                move = None
                moveValid = False
                while not moveValid:
                    #   generate a move
                    oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                    output = net(oneHot)
                    values, index = output.view(9).max(0)
                    if board.flatten()[index] == 0: #   if move is valid
                        moveValid = True

                        #   apply the move
                        move = index
                        board = board.flatten()
                        board[move] = computersPlayer
                        board = board.reshape(3, 3)
                        
                        #   store for later
                        moves.append(move)
                        outputs.append(output)
                    else:   #   invalid move, prime the whip
                        # print("invalid move")
                        numInvalidMoves += 1
                        optimizer.zero_grad()
                        validMoves = np.where(board == 0, 1, 0)
                        target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
            else:   #   opponents turn
                empties = tt.listEmpties(board)
                randomMove = random.choice(empties)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)

            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)
        
        #   get end score of game

        score = scoreEndBoard(board, winner, computersPlayer)
        for i, move in enumerate(moves):
            output = outputs[i]
            target = output.clone().view(9)
            target[move] = score
            target = target.view(1, 1, 9)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test(net, criterion, optimizer, epochs):
    numInvalidMoves = 0

    numWins = 0
    numLosses = 0
    numTies = 0

    optimizer.zero_grad()

    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        board = np.zeros(shape = (3, 3))
        # board = np.random.randint(low = 0, high = 3, size = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        while(not winner and movesLeft):
            if player == computersPlayer:
                move = None
                moveValid = False
                while not moveValid:
                    #   generate a move
                    oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                    output = net(oneHot)
                    values, index = output.view(9).max(0)
                    if board.flatten()[index] == 0: #   if move is valid
                        moveValid = True

                        #   apply the move
                        move = index
                        board = board.flatten()
                        board[move] = computersPlayer
                        board = board.reshape(3, 3)
                        
                    else:   #   invalid move, prime the whip
                        # print("invalid move")
                        numInvalidMoves += 1
                        optimizer.zero_grad()
                        validMoves = np.where(board == 0, 1, 0)
                        target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
            else:   #   opponents turn
                empties = tt.listEmpties(board)
                randomMove = random.choice(empties)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)

            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)
        
        if winner == computersPlayer:
            numWins += 1
        elif winner == tt.togglePlayer(computersPlayer):
            numLosses += 1
        else:   #   winner == False
            numTies += 1

    return numWins, numLosses, numTies

def playWithUser(net, criterion, optimizer, online=True):
    while(True):
        board = genBoard()
        movesLeft = True
        winner = False
        player = 2
        computersPlayer = 2 #random.randint(1,2)

        print()
        print("NEW GAME")
        if computersPlayer == 2:
            print("COMPUTER GOES FIRST...")

        moves = []
        outputs = []
        while(movesLeft and not winner):
            if player == 2:
                print("X's Turn")
            else: # player == 1
                print("O's Turn")
            tt.printBoard(board)

            if player == computersPlayer:
                move = None
                moveValid = False
                while not moveValid:
                    #   generate a move
                    oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                    output = net(oneHot)
                    values, index = output.view(9).max(0)
                    if board.flatten()[index] == 0: #   if move is valid
                        moveValid = True

                        #   apply the move
                        move = index
                        board = board.flatten()
                        board[move] = computersPlayer
                        board = board.reshape(3, 3)
                        
                        #   store for later
                        moves.append(move)
                        outputs.append(output)
                    else:   #   invalid move, prime the whip
                        print("invalid move")
                        optimizer.zero_grad()
                        validMoves = np.where(board == 0, 1, 0)
                        target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
            elif player == tt.togglePlayer(computersPlayer):
                validMove = False
                while validMove == False:
                    move = input("input move of form 'y x' ")
                    y = int(move[0])
                    x = int(move[2])
                    #   validate move
                    if not board[y][x] == 0:
                        print("!!!INVALID MOVE!!!")
                        continue
                    else:
                        validMove = True
                    board[y][x] = tt.togglePlayer(computersPlayer)
            player = tt.togglePlayer(player)
            
            winner = tt.getWinner(board)
            movesLeft = not tt.noMoreMoves(board)

        tt.printBoard(board)

        if online:
            score = scoreEndBoard(board, winner, computersPlayer)
            for i, move in enumerate(moves):
                output = outputs[i]
                target = output.clone().view(9)
                target[move] = score
                target = target.view(1, 1, 9)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        if winner:
            if winner == 2:
                print("WINNER: X")
            else: # winner == 1
                print("WINNER: O")
        else:
            print("TIE")


def main():
    net = Net(18, 36, 9)
    # net = Net(18, 256, 9)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # print(net)

    # train(net=net, epochs=100)

    # board = genBoard()
    # oneHot = oneHotTicTacToe(board=board, computersPlayer=2)
    # oneHot = oneHot.reshape(1, 22)  #   presumably we could batch the inputs for rapid training

    # output = net(oneHot)

    trainToMakeValidMoves(net=net, criterion=criterion, optimizer=optimizer, epochs=1000)
    train(net=net, criterion=criterion, optimizer=optimizer, epochs=10000)
    numWins, numLosses, numTies = test(net=net, criterion=criterion, optimizer=optimizer, epochs=1000)
    print("wins, losses, ties:")
    print(numWins)
    print(numLosses)
    print(numTies)

    # trainAgainstSelf(net=net, criterion=criterion, optimizer=optimizer, epochs=1000)
    # numWins, numLosses, numTies = test(net=net, criterion=criterion, optimizer=optimizer, epochs=1000)

    # print("wins, losses, ties:")
    # print(numWins)
    # print(numLosses)
    # print(numTies)

    # playWithUser(net=net, criterion=criterion, optimizer=optimizer)

if __name__ == '__main__':
    main()