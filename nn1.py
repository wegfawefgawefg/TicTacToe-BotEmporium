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


def train(net, criterion, optimizer, epochs):
    for i in tqdm(range(epochs)):
        player = 2
        computersPlayer = 2 #random.randint(1,2)

        optimizer.zero_grad()

        # board = np.zeros(shape = (3, 3))
        board = np.random.randint(low = 0, high = 3, size = (3, 3))
        print(board)

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
                        move = index
                    else:   #   invalid move, prime the whip
                        print("invalid move")
                        optimizer.zero_grad()
                        validMoves = np.where(board == 0, 1, 0)
                        target = torch.tensor(validMoves, dtype=torch.float).view(1, 1, 9)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                #   apply the move
                board = board.flatten()
                board[move] = computersPlayer
                board = board.reshape(3, 3)
                print(board)
                quit()
            else:   #   opponents turn
                moves = tt.listEmpties(board)
                randomMove = random.choice(moves)
                tt.applyMove(player, randomMove, board)
            player = tt.togglePlayer(player)


            movesLeft = np.any(np.where(board == 0, 1, 0))
            winner = tt.getWinner(board)

        #   evaluate game
        #   determine target
        #   go through all moves chosen
        #   punish each one by a depth factor times the target
        #   update network


def main():
    net = Net(18, 36, 9)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # print(net)

    # train(net=net, epochs=100)

    # board = genBoard()
    # oneHot = oneHotTicTacToe(board=board, computersPlayer=2)
    # oneHot = oneHot.reshape(1, 22)  #   presumably we could batch the inputs for rapid training

    # output = net(oneHot)

    trainToMakeValidMoves(net=net, criterion=criterion, optimizer=optimizer, epochs=10000)
    # train(net=net, criterion=criterion, optimizer=optimizer, epochs=10000)

    # for i in range(0, 10):
    #     board = np.random.randint(low = 0, high = 3, size = (3, 3))
    #     oneHot = oneHotTicTacToe(board, 2).view(1, 1, 18)
    #     output = net(oneHot)
    #     print(board)
    #     print(output)

if __name__ == '__main__':
    main()