import pickle
from typing import Tuple, Annotated, Literal

import numpy as np
import time

import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch import manual_seed
import logging
import gurobipy as gp
from gurobipy import GRB
from utils import correction_single_obj

# torch.autograd.set_detect_anomaly(True)

manual_seed(1234)

# capacity = 100
capacity = np.load(
    'data/synthetic/capacity_kp_50.npy')
purchase_fee = 0.2
compensation_fee = 0.21

itemNum = 50
featureNum = 5
trainSize = 1000
targetNum = 2

def get_xTrue(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        cnt = num * itemNum
        for i in range(itemNum):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        sol = np.zeros(itemNum)
        for i in range(itemNum):
            sol[i] = x[i].x
            
        objective = m.objVal
#        print("TOV: ", sol, objective)
        
    return sol

def get_Xs1Xs2(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice)
    realWeight = np.zeros(itemNum)
    predWeight = np.zeros(itemNum)
    realPriceNumpy = np.zeros(itemNum)
    for i in range(itemNum):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = np.zeros(itemNum,dtype='i')
        for i in range(itemNum):
            predSol[i] = x[i].x
            
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

        # Stage 2:
        realWeight = realWeight.tolist()
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x = m2.addVars(itemNum, vtype=GRB.BINARY, name='x')
        sigma = m2.addVars(itemNum, vtype=GRB.BINARY, name='sigma')

        OBJ = purchase_fee * x.prod(realPrice)
        for i in range(itemNum):
            OBJ = OBJ - compensation_fee * realPrice[i] * sigma[i]
        m2.setObjective(OBJ, GRB.MAXIMIZE)

        m2.addConstr((x.prod(realWeight) - sigma.prod(realWeight)) <= cap)
        for i in range(itemNum):
            m2.addConstr(x[i] == predSol[i])
            m2.addConstr(x[i] >= sigma[i])
        try:
            m2.optimize()
            objective = m2.objVal
            sol = np.zeros(itemNum)
            for i in range(itemNum):
                sol[i] = x[i].x - sigma[i].x
        except:
            print(predPrice, predWeight, realPrice, realWeight, predSol)
#        print("Stage 2: ", sol, objective)

    return predSol,sol
    

def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(itemNum)
        value = np.zeros(itemNum)
        # cnt = num * itemNum
        # TODO: check
        for i in range(itemNum):
            weight[i] = weightTemp[num][i]
            value[i] = valueTemp[num][i]
            # cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(itemNum, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(itemNum):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        sol = []
        selectedItemNum = 0
        for i in range(itemNum):
            sol.append(x[i].x)
            if x[i].x == 1:
              selectedItemNum = selectedItemNum + 1
        objective = m.objVal
        obj_list.append(objective)
        selectedNum_list.append(selectedItemNum)
        # print(selectedItemNum)
#        print("TOV: ", sol, objective)
        
    return np.array(obj_list)

    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def make_fc(num_layers, num_features,
            # num_target must be 1 or 2. If 1, we either predict the costs or weights. If 2, we predict both the costs
            # and weights.
            num_targets: Literal[1, 2] = targetNum,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):

    error_msg = ('num_target must be 1 or 2. If 1, we either predict the costs or weights. If 2, we predict both the '
                 'costs and weights.')
    assert num_targets in {1, 2}, error_msg

    net_layers = [nn.Linear(num_features, intermediate_size),activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    # In the original version, the number of output units was 2. They basically predict the cost and weight for item
    # independently. To align with our implementation, we predict the joinly predict the cost and weight for all the
    # items.
    net_layers.append(nn.Linear(intermediate_size, itemNum * num_targets))
    net_layers.append(activation_fn())
    return nn.Sequential(*net_layers)
        

class MyCustomDataset():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """


        Returns
        -------
        A tuple with two torch.Tensor, the feature and target vectors. The feature vector has shape
        (n_examples, n_fetures). The target vector has shape (n_examples, n_items, n_targets), where n_targets must be 1
        or 2. If 1, the target is either the cost or weight array. If 2, the target are both the cost and weight arrays.
        """
        return self.feature[idx], self.value[idx]


import sys
import ip_model_whole as ip_model_wholeFile
from ip_model_whole import IPOfunc

class Intopt:
    def __init__(self, c, h, A, b, purchase_fee, compensation_fee, n_features,
                 pretrain_epochs: Annotated[int, "Must be greater than 0"],
                 train_epochs: Annotated[int, "Must be greater than 0"],
                 num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
                 damping=0.5, target_size=targetNum, optimizer=optim.Adam,
                 batch_size=itemNum, **hyperparams):
        """

        Parameters
        ----------
        pretrain_epochs: number of training epochs using the MSE as the loss function.
        train_epochs: number of epochs using the post-hoc regret as the loss function.
        epochs: total number of epochs (simply the sum of pretrain and train epochs).
        """

        assert pretrain_epochs >= 0, "pretrain_epochs must be greater than or equal 0"
        assert train_epochs >= 0, "train_epochs must be greater than or equal 0"
        assert train_epochs + pretrain_epochs >= 1, "Either train_epochs or pretrain_epochs must be greather than 0"

        self.c = c
        self.h = h
        self.A = A
        self.b = b
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers
        self.purchase_fee = purchase_fee
        self.compensation_fee = compensation_fee

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.pretrain_epochs = pretrain_epochs
        self.train_epochs = train_epochs
        self.epochs = pretrain_epochs + train_epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

#        self.model = Net(n_features=n_features, target_size=target_size)
        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)
        #self.model.apply(weight_init)
#        w1 = self.model[0].weight
#        print(w1)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, dataset_features, dataset_values):
        mse_history = []
        regret_history = []

        init_mse, init_post_hoc_regret = self.val_loss(capacity, dataset_features, dataset_values)

        print('Initial results: ')
        print(f'MSE: {init_mse} | Post-hoc regret: {init_post_hoc_regret}')

        logging.info("Intopt")
        train_df = MyCustomDataset(dataset_features, dataset_values)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
          total_loss = 0
#          for parameters in self.model.parameters():
#            print(parameters)
          if e < self.pretrain_epochs:
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                # TODO: check
                op = torch.reshape(op, (itemNum, targetNum))
    #                print(feature, value, op)
    #                print(feature.shape, value.shape, op.shape)
                # targetNum=1: torch.Size([10, 4096]) torch.Size([10]) torch.Size([10])
                # targetNum=2: torch.Size([10, 4096]) torch.Size([10, 2]) torch.Size([10, 2])
    #                print(value, op)

                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            grad_list[e] = total_loss
            print("Epoch{} ::loss {} ->".format(e,total_loss))
            # self.val_loss(capacity, feature_train, value_train)
          else:
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
            num = 0
            batchCnt = 0
            loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
            for feature, value in train_dl:

                value = torch.squeeze(value)

                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                op = torch.reshape(op, (itemNum, targetNum))
                # FIXME: this might lead to infinite loop.
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                # while torch.isnan(op).any() or torch.isinf(op).any():
                    # print('NN has been reinitialized')
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
                    op = torch.reshape(op, (itemNum, targetNum))
                    # mse_loss, post_hoc_regret = self.val_loss(capacity, dataset_features, dataset_values)
                    # print(f'After reinitialization - MSE: {mse_loss} | post-hoc regret: {post_hoc_regret}')

                price = np.zeros(itemNum)
                for i in range(itemNum):
                    # TODO: check
                    price[i] = np.squeeze(self.c[num][i])
                    
                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(self.h).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                
                G_torch = torch.zeros((itemNum+1, itemNum))
                for i in range(itemNum):
                    G_torch[i][i] = 1
                G_torch[itemNum] = value[:, 1]
                trueWeight = value[:, 1]
                
#                op_torch = torch.zeros((itemNum+1, itemNum))
#                for i in range(itemNum):
#                    op_torch[i][i] = 1
#                op_torch[itemNum] = op
                
#                print(G_torch)
#                print(op_torch)
                x_s2 = IPOfunc(A=A_torch, b=b_torch, h=h_torch, cTrue=-c_torch, GTrue=G_torch, purchase_fee=self.purchase_fee, compensation_fee=self.compensation_fee, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                x_s1 = ip_model_wholeFile.x_s1
                
#                trueWeight = trueWeight.numpy()
##                print(x, c_torch)
##                newLoss = (x * c_torch).sum() + torch.dot(torch.mul(c_torch, penalty), torch.mul(x, 1-1/ip_model_wholeFile.violateFactor))
#                x_true = get_xTrue(price, capacity, trueWeight, 1)
#                x_true = torch.from_numpy(x_true)
#                price = price.tolist()
#                predPrice = op[:, 0].detach().tolist()
#                predWeight = op[:, 1].detach().tolist()
#                x_s1_true, x_s2_true = get_Xs1Xs2(price, predPrice, capacity, trueWeight, predWeight)
##                print(x_s1_true,x_s2_true)
#                x_s1_true = torch.from_numpy(x_s1_true)
#                x_s2_true = torch.from_numpy(x_s2_true)
                
                newLoss = - (purchase_fee * (x_s2 * c_torch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_torch, abs(x_s2-x_s1).float()))
#                print(x_s2,x_s1,newLoss)
#                newLoss.data = purchase_fee * (x_true * c_torch).sum() - (purchase_fee * (x_s2_true * c_torch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_torch, abs(x_s2_true-x_s1_true).float()))
#                print(x_s2_true,x_s1_true,newLoss)
#                print(newLoss)
#                newLoss = - (x * c_torch).sum()
#                loss = loss - (x * c_torch).sum()
                loss = loss.detach() + newLoss.detach()
                batchCnt = batchCnt + 1
#                print(loss)
#                loss = torch.dot(-c_torch, x)
#                print(loss.shape)
                  
#                print(x)
                #loss = -(x * value).mean()
                #loss = Variable(loss, requires_grad=True)
                total_loss += newLoss.item()
                # op.retain_grad()
                #print(loss)
                
                newLoss.backward()
                #print("backward1")
                self.optimizer.step()
                
                # when training size is large
                if batchCnt % 70 == 0:
                    print(newLoss)
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1
            grad_list[e] = total_loss/trainSize
            print("Epoch{} ::train loss {} ->".format(e,grad_list[e]))

          logging.info("EPOCH Ends")

          epoch_mse, epoch_post_hoc_regret = self.val_loss(capacity, dataset_features, dataset_values)
          print(f'Epoch: {e+1}: train MSE: {epoch_mse} | train post-hoc regret: {epoch_post_hoc_regret}')
          mse_history.append(epoch_mse)
          regret_history.append(epoch_post_hoc_regret)

          #print("Epoch{}".format(e))
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
          if e > 1 and abs(grad_list[e] - grad_list[e-1]) <= 0.01:
            break

        return grad_list, {'mse': mse_history, 'regret': regret_history}

    # TODO: check
    def val_loss(self, cap, feature, value) -> Tuple[float, float]:
        valueTemp = value.numpy()
#        test_instance = len(valueTemp) / self.batch_size
        test_instance = np.size(valueTemp, 0) / self.batch_size
#        itemVal = self.c.tolist()
        itemVal = self.c
        # Cost of the true optimal solutions.
        real_obj = (
            actual_obj(valueTemp=itemVal,
                       cap=cap,
                       # The element on index 1 in the target array is the weight array.
                       weightTemp=value[:, :,  1],
                       n_instance=int(test_instance))
        )
#        print(np.sum(real_obj))

        # self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(value=value, feature=feature)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=1, shuffle=False)

        obj_list = []
        corr_obj_list = []
        len = np.size(valueTemp, 0)
        predVal = torch.zeros((len, 2))
        
        num = 0
        mse_loss = 0
        for feature, value in valid_dl:
            # Remove the fake batch dimension.
            value = torch.squeeze(value)
            op = self.model(feature).squeeze().detach()
            op = torch.reshape(op, (itemNum, targetNum))
#            print(op)
            loss = criterion(op, value).item()
            mse_loss += loss

            realWT = {}
            predWT = {}
            realPrice = {}
            predPrice = {}
            for i in range(itemNum):
                realWT[i] = value[i][1]
                predWT[i] = op[i][1]
                realPrice[i] = value[i][0]
                predPrice[i] = op[i][0]
                # predVal[num][0] = op[i][0]
                # predVal[num][1] = op[i][1]

            opt_res = (
                correction_single_obj(realPrice,
                                      predPrice,
                                      cap,
                                      realWT,
                                      predWT,
                                      purchase_fee=purchase_fee,
                                      compensation_fee=compensation_fee,
                                      item_num=itemNum))
            corr_obj_list.append(opt_res.obj)
            num = num + 1

        post_hoc_regret  = np.mean(abs(np.array(corr_obj_list) - real_obj))

        # self.model.train()
#        print(corr_obj_list)
#        print(corr_obj_list-real_obj)
#        print(np.sum(corr_obj_list))
        return mse_loss, post_hoc_regret

#c_dataTemp = np.loadtxt('KS_c.txt')
#c_data = c_dataTemp[:itemNum]

h_data = np.ones(itemNum+1)
h_data[itemNum] = capacity
A_data = np.zeros((2, itemNum))
b_data = np.zeros(2)


print("*** HSD ****")

testTime = 5
recordBest = np.zeros((1, testTime))
stopCriterior = 15

mse_exp_history = []
regret_exp_history = []

for testi in range(testTime):
    print(f'Test #{testi}')

    # x_train = np.loadtxt('./data/train_features/train_features(0).txt')
    x_train = pd.read_csv(
        'data/synthetic/features_kp_50.csv',
        index_col=0)
    x_train = x_train.values
    # c_train = np.loadtxt(f'./data/train_prices/train_prices({testi}).txt')
    c_train = pd.read_csv(
        'data/synthetic/values_kp_50.csv', index_col=0)
    c_train = c_train.values
    # c_train has shape (n_items, 1). We replace the fake additional dimension (on axis 1) and repeat the
    # array for each training examples. The final shape is (n_examples * n_items).
    # TODO: why this shape?
    c_train = np.squeeze(c_train)
    c_train = np.tile(c_train, x_train.shape[0])
    # y_train1 = np.loadtxt(f'./data/train_prices/train_prices({testi}).txt')
    # y_train2 = np.loadtxt('./data/train_weights/train_weights(' + str(testi) + ').txt')
    y_train1 = c_train.copy()
    y_train2 = pd.read_csv(
        'data/synthetic/targets_kp_50.csv', index_col=0)
    y_train2 = y_train2.values
    # Similarly as for c_train, we build a weights array with shape(n_examples * n_itmes).
    # TODO: why this shape?
    y_train2 = y_train2.reshape(-1)

    meanPriceValue = np.mean(c_train)
    meanWeightValue = np.mean(y_train2)

    # We build a target array of shape (n_examples * n_items, 2), where index 0 (on axis 1) refers to the cost and index
    # 1 (on axis 1) refers to the weights.
    y_train = np.zeros((y_train1.size, 2))
    for i in range(y_train1.size):
        y_train[i][0] = y_train1[i]
        y_train[i][1] = y_train2[i]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()
    # TODO: check
    value_train = value_train.reshape(trainSize, itemNum, targetNum)
    c_train = c_train.reshape(trainSize, itemNum, 1)
    
    # c_test = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    # x_test = np.loadtxt('./data/test_features/test_features(0).txt')
    # y_test1 = np.loadtxt('./data/test_prices/test_prices(' + str(testi) + ').txt')
    # y_test2 = np.loadtxt('./data/test_weights/test_weights(' + str(testi) + ').txt')

    c_test = c_train.copy()
    x_test = x_train.copy()
    y_test2 = y_train2.copy()

    # y_test = np.zeros((y_test1.size, 2))
    # for i in range(y_test1.size):
    #     y_test[i][0] = y_test1[i]
    #     y_test[i][1] = y_test2[i]

    y_test = y_train.copy()

    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()
    
    start = time.time()
    damping = 1e-2
    thr = 1e-3
    lr = 1e-5
    bestTrainCorrReg = float("inf")
    for j in range(1):
        # TODO: check
        clf = (
            Intopt(c_train, h_data, A_data, b_data, purchase_fee, compensation_fee,
                   damping=damping,
                   lr=lr,
                   n_features=featureNum,
                   thr=thr,
                   pretrain_epochs=0,
                   train_epochs=10,
                   # In the original version the batch size is the number of items, since the dataset has shape
                   # (n_examples * n_items). They use the batch size to group together features and targets for the same
                   # example. The actual batch size (in its more general meaning) is 1.
                   batch_size=1))
        _, history = clf.fit(feature_train, value_train)
        mse_exp_history.append(history['mse'])
        regret_exp_history.append(history['regret'])

        with open(f'kp-capacity-{capacity}-mse-history.pkl', 'wb') as file:
            pickle.dump(mse_exp_history, file)

        with open(f'kp-capacity-{capacity}-regret-history.pkl', 'wb') as file:
            pickle.dump(regret_exp_history, file)

        train_mse, train_post_hoc_regret = clf.val_loss(capacity, feature_train, value_train)
        # avgTrainCorrReg = np.mean(train_rslt)
        # trainHSD_rslt = 'train: ' + str(np.mean(train_rslt))

        if train_post_hoc_regret < bestTrainCorrReg:
            bestTrainCorrReg = train_post_hoc_regret
            torch.save(clf.model.state_dict(), 'model.pkl')

        print(f'Final result: {train_post_hoc_regret}')
        
        if train_post_hoc_regret < stopCriterior:
            break


    clfBest = Intopt(c_test, h_data, A_data, b_data, purchase_fee, compensation_fee, damping=damping, lr=lr, n_features=featureNum, thr=thr, train_epochs=1, pretrain_epochs=1)
    clfBest.model.load_state_dict(torch.load('model.pkl'))

    val_mse, val_post_hoc_regret = clfBest.val_loss(capacity, feature_test, value_test)
    end = time.time()

#     predTestVal = predTestVal.detach().numpy()
# #    print(predTestVal.shape)
#     predTestVal1 = predTestVal[:, 0]
#     predTestVal2 = predTestVal[:, 1]
#     predValuePrice = np.zeros((predTestVal1.size, 2))
#     for i in range(predTestVal1.size):
# #        predValue[i][0] = int(i/itemNum)
#         predValuePrice[i][0] = y_test1[i]
#         predValuePrice[i][1] = predTestVal1[i]
#     np.savetxt('./data/2S_prices/2S_prices_cap' + str(capacity) + '_compensation' + str(compensation_fee) + '(' + str(testi) + ').txt', predValuePrice, fmt="%.2f")
#     predValueWeight = np.zeros((predTestVal2.size, 2))
#     for i in range(predTestVal2.size):
# #        predValue[i][0] = int(i/itemNum)
#         predValueWeight[i][0] = y_test2[i]
#         predValueWeight[i][1] = predTestVal2[i]
#     np.savetxt('./data/2S_weights/2S_weights_cap' + str(capacity) + '_compensation' + str(compensation_fee) + '(' + str(testi) + ').txt', predValueWeight, fmt="%.2f")
    
    HSD_rslt = 'test: ' + str(val_post_hoc_regret)
    print(HSD_rslt)
    print ('Elapsed time: ' + str(end-start))
    print('\n' + '-'*50)
    print()

