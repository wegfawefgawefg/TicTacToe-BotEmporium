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
///LOOKS ABOUT DONE///
///MOVING ON TO NEXT PROJECT///

wins, losses, ties:
899
87
14
'''

'''
TODO:
-make offline mode
-clean up the rest of the function, play against user, train against self
-assert exploration is happening in play against self
-why are moves worse without unenforcing invalid moves?, does it understand the game less? or is it just more training sets
-expand the newtork, add dropout, try resnet
-output precent chance of winning?
-experiment with more inputs? such as whos turn it is?
-investigate network dying: (SOLVED) (IF THE WEIGHTS ARE NEGATIVE, THE MAX WILL BE AN INVALID MOVE) SOLUTION: set masked vals to -inf instead of 0
    -sometimes it learns a really poor meta and loses most of its games
    -not sure whats going on here, i think some moves get so dissincentivised that they die
-make it play against all random moves instead of just some random moves
-make it occasionally make random moves to be more explorative or something


TODO MAYBE:
-gpu? batch training many games at once? (is this worth the effort)
-tournament with other algorithms?
-refactor all the tic tac toe game code to shrink it down?
'''

'''
METRICS REQUEST LIST
-tally invalid moves attempted?
-games as x vs o
-moves as x vs o
-game winning blocks missed / numGames over time?
'''

class Net(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, outputSize)

        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        # x = self.dropout1(x)
        x = F.leaky_relu( self.l2(x) )
        # x = self.dropout2(x)
        x = self.l8(x)
        return x
        # return F.softmax(x, dim=2)

class BigNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.l2 = nn.Linear(hiddenSize, hiddenSize)
        self.l3 = nn.Linear(hiddenSize, hiddenSize)
        self.l4 = nn.Linear(hiddenSize, hiddenSize)
        self.l5 = nn.Linear(hiddenSize, hiddenSize)
        self.l6 = nn.Linear(hiddenSize, hiddenSize)
        self.l7 = nn.Linear(hiddenSize, hiddenSize)
        self.l8 = nn.Linear(hiddenSize, outputSize)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout6 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.5)


    def forward(self, x):
        x = F.leaky_relu( self.l1(x) )
        x = self.dropout1(x)
        x = F.leaky_relu( self.l2(x) )
        x = self.dropout2(x)
        x = F.leaky_relu( self.l3(x) )
        x = self.dropout3(x)
        x = F.leaky_relu( self.l4(x) )
        x = self.dropout4(x)
        x = F.leaky_relu( self.l5(x) )
        x = self.dropout5(x)
        x = F.leaky_relu( self.l6(x) )
        x = self.dropout6(x)
        x = F.leaky_relu( self.l7(x) )
        x = self.dropout7(x)
        x = self.l8(x)
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
        # return -1
        return 1
    elif winner == tt.togglePlayer(myPlayer):
        return -1
        # return -2
    elif winner == myPlayer:
        return 5
        # return 1

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

def trainAgainstSelf(net, criterion, optimizer, epochs):
    net.train()

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
            #   generate a move
            if player == computersPlayer:
                oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
            else:
                oneHot = oneHotTicTacToe(board, tt.togglePlayer(computersPlayer)).view(1, 1, 18)
            output = net(oneHot)

            #   mask out invalid moves
            invalidMoves = np.where( board.flatten() > 0, True, False)
            maskedOutput = output.clone().view(9)
            maskedOutput[invalidMoves] = -10
            values, index = maskedOutput.max(0)

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
    net.train()

    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        optimizer.zero_grad()

        board = np.zeros(shape = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        gameDuration = 0

        moves = []
        outputs = []
        while(not winner and movesLeft):
            if player == computersPlayer:
                #   generate a move
                oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                output = net(oneHot)

                #   mask out invalid moves
                invalidMoves = np.where( board.flatten() > 0, True, False)
                maskedOutput = output.clone().view(9)
                maskedOutput[invalidMoves] = -10
                values, index = maskedOutput.max(0)

                #   apply the move
                move = index
                board = board.flatten()
                board[move] = computersPlayer
                board = board.reshape(3, 3)
                        
                #   store for later
                moves.append(move)
                outputs.append(output)

            else:   #   opponents turn
                empties = tt.listEmpties(board)
                randomMove = random.choice(empties)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)
            gameDuration += 1

            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)
        
        #   get end score of game
        score = scoreEndBoard(board, winner, computersPlayer)
        # gameDurationMultiplier = 1.0 - gameDuration / 10
        # gameDurationMultiplier = gameDurationMultiplier * 0.9
        dilutionFactor = 0.9
        totalDilutant = 1.0
        for i, move in reversed(list(enumerate(moves))):
            totalDilutant *= dilutionFactor
            output = outputs[i]
            target = output.clone().view(9)
            target[move] = score * totalDilutant
            target = target.view(1, 1, 9)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test(net, epochs):
    net.eval()

    numWins = 0
    numLosses = 0
    numTies = 0

    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = random.randint(1,2)

        board = np.zeros(shape = (3, 3))
        # board = np.random.randint(low = 0, high = 3, size = (3, 3))

        movesLeft = np.any(np.where(board == 0, 1, 0))
        winner = tt.getWinner(board)

        while(not winner and movesLeft):
            if player == computersPlayer:
                #   generate a move
                oneHot = oneHotTicTacToe(board, computersPlayer).view(1, 1, 18)
                output = net(oneHot)

                #   mask out invalid moves
                invalidMoves = np.where( board.flatten() > 0, True, False)
                maskedOutput = output.clone().view(9)
                maskedOutput[invalidMoves] = -10
                values, index = maskedOutput.max(0)

                #   apply the move
                move = index
                board = board.flatten()
                board[move] = computersPlayer
                board = board.reshape(3, 3)
                        
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

def playWithUser(net, online=True):
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
    net = Net(18, 64, 9)
    # net = Net(18, 256, 9)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(net=net, criterion=criterion, optimizer=optimizer, epochs=10000)
    numWins, numLosses, numTies = test(net=net, epochs=1000)
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