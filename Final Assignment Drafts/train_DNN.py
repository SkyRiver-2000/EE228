from game2048.game import Game
# from game2048.displays import Display
from game2048.agents import Agent, ExpectiMaxAgent

import numpy as np
import pandas as pd
import random as rd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt

from MyAgent import *
from dnn import Deep_NN

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10
    TRAIN_EPOCHES = 100
    TRAIN_GAMES = 5
    NUM_EPOCHS = 100
    MAX_ITER = 2e3
    BATCH_SIZE = 16
    MEMORY_SIZE = int(1e5)
    
    # game = Game(GAME_SIZE, SCORE_TO_WIN)
    # print(type(game.board))
    
    scores = [ ]
    
    model = Conv_Net().cuda()
    model.load_state_dict(torch.load("./mdl.pkl"))
    optim = opt.Adam(model.parameters(), lr = 5e-4)
    criterion = nn.MSELoss()
    
    Board_Memory = torch.zeros(size = (MEMORY_SIZE, 12, GAME_SIZE, GAME_SIZE)).type(torch.FloatTensor)
    Move_Memory = torch.zeros(size = (MEMORY_SIZE, 4)).type(torch.FloatTensor)
    Memory_count = 0
    
    for ep in range(TRAIN_EPOCHES):
        for i in range(TRAIN_GAMES):
            game, n_iter = Game(GAME_SIZE), 0
            target_Agent = ExpectiMaxAgent(game)
            while n_iter < MAX_ITER:
                n_iter += 1
                org_board = deepcopy(target_Agent.game.board)
                target_dir = target_Agent.step()
                target_Agent.game.move(target_dir)
                if np.max(np.max(org_board)) == 2048:
                    break
                if Memory_count < MEMORY_SIZE:
                    pos = Memory_count
                    Memory_count += 1
                else:
                    pos = rd.choice(range(MEMORY_SIZE))
                Board_Memory[pos] = one_hot_encode(org_board)
                Move_Memory[pos, target_dir] = n_iter
            print(target_Agent.game.score)
        optim.zero_grad()
        samples = rd.sample(range(Memory_count), BATCH_SIZE)
        X_train, y_train = Board_Memory[samples], Move_Memory[samples]
        X_train, y_train = Variable(X_train).type(torch.FloatTensor).cuda(), Variable(y_train).type(torch.FloatTensor).cuda()
        outputs = model(X_train)
        _, y_pred = torch.max(outputs, 1)
        _, train = torch.max(y_train, 1)
        train_correct = torch.sum(y_pred.data == train.data)
        loss = criterion(outputs, y_train)
        print("Epoch: {}, Loss: {}".format(ep, loss.data.cpu().numpy()))
        loss.backward()
        optim.step()
        
        print("Training Accuracy: %.2f%%" % (train_correct.cpu().numpy() / BATCH_SIZE * 100))
    
    print('Training Done!')
    model = model.cpu()
    model.eval()
    
    score_list = [ ]
    for idx in range(N_TESTS):
        score_list.append(single_run(GAME_SIZE, SCORE_TO_WIN, MyOwnAgent, model))
    print("Scores:", score_list) 
    print("Average scores: @%s times" % N_TESTS, np.mean(score_list))
    
    torch.save(model.state_dict(), "./dnn_mdl.pkl")
