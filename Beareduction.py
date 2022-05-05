import inspect
import os
import torch
import torch.nn as nn

class Beareduction():
    def __init__(self, train_get_loss_acc, net, mode, device=None, unit: int = 100, loss_threshold: float = 0.05, acc_threshold: float = 0.02) -> None:
        self.unit = unit
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.train_get_loss_acc = train_get_loss_acc
        self.net = net
        self.device = device

    def getVariable(self, net):
        variable_list = []
        for i in inspect.getmembers(net):
          
            # to remove private and protected
            # functions
            if not i[0].startswith('_'):

                # To remove other methods that
                # doesnot start with a underscore
                if not inspect.ismethod(i[1]): 
                    variable_list.append(i[1])
        return variable_list

    def getLinearLayers(self, var_list):
        linear_layers = []
        for var in var_list:
            if (var.__class__.__name__ == 'Linear'):
                linear_layers.append(var)
            elif (var.__class__.__name__ == 'Sequential'):
                linear_layers.extend(self.getLiearFromSeq(var))
        return linear_layers

    def getLiearFromSeq(self, seq):
        linear_layers = []
        for layer in seq:
            if layer.__class__.__name__ == 'Linear':
                linear_layers.append(layer)
        return linear_layers

    def print_linear(self, linear_layer):
        out_feature_list = []
        for i in range(len(linear_layer) - 1):
            out_feature_list.append(linear_layer[i].out_features)
        print(out_feature_list)
        return out_feature_list
        
    def decrease(self, curr_layer, next_layer, unit, net):
        curr_layer.__init__(curr_layer.in_features, curr_layer.out_features  - unit)
        next_layer.__init__(next_layer.in_features - unit, next_layer.out_features)
        if self.device:
            net.to(self.device)
        
    def increase(self, curr_layer, next_layer, unit, net):
        curr_layer.__init__(curr_layer.in_features, curr_layer.out_features + unit)
        next_layer.__init__(next_layer.in_features + unit, next_layer.out_features)
        if self.device:
            net.to(self.device)
        
    def initialize_weights(self,net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

    def get_mec(self, linear_layer):
        mec = linear_layer[0].in_features * linear_layer[0].out_features
        for i in range(1, len(linear_layer)):
            mec = mec + linear_layer[i].out_features
        return mec

    def reduce(self):
        variable_list = self.getVariable(self.net)
        linear_layers = self.getLinearLayers(variable_list)
     
        print("############### Base Round ###############")
        self.initialize_weights(self.net)
        self.print_linear(linear_layers)
        base_acc, base_loss = self.train_get_loss_acc(self.net)
        
        print("base_loss: " + str(base_loss))
        print("base_accuracy: " + str(base_acc))

        linear_layer_change = []
        mec_change = []

        linear_layer_change.append(self.print_linear(linear_layers))
        mec_change.append(self.get_mec(linear_layers))

        stop = False
        
        while not stop:
            for i in range(len(linear_layers) - 1):
                if linear_layers[i].out_features - self.unit <= 0:
                    continue
                self.initialize_weights(self.net)
                self.decrease(linear_layers[i], linear_layers[i + 1], self.unit, self.net)
                
                new_acc, new_loss = self.train_get_loss_acc(self.net)

                within_acc_thres = new_acc > base_acc * (1 - self.acc_threshold)
                within_loss_thres = new_loss < base_loss * (1 + self.loss_threshold)

                if self.mode == 'acc':
                    within_thred = within_acc_thres
                elif self.mode == 'loss':
                    within_thred = within_loss_thres
                else:
                    within_thred = within_acc_thres or within_loss_thres

                if within_thred:
                    layer_size = linear_layers[i].out_features
                    for j in range(i + 1, len(linear_layers) - 1):
                        if linear_layers[j].out_features > layer_size:
                            self.decrease(linear_layers[j], linear_layers[j + 1], linear_layers[j].out_features - layer_size, self.net)
                    print("loss: " + str(new_loss))
                    print("acc: " + str(new_acc))
                    print("############### New Round ###############")
                    linear_layer_change.append(self.print_linear(linear_layers))
                    mec_change.append(self.get_mec(linear_layers))
                    self.initialize_weights(self.net)
                    base_acc, base_loss = self.train_get_loss_acc(self.net)
                    print("base loss: " + str(base_loss))
                    print("base acc: " + str(base_acc))
                    break
                elif i == len(linear_layers) - 2:
                    print("############### Training End ###############")
                    stop = True
                    break
                    
                self.increase(linear_layers[i], linear_layers[i + 1], self.unit, self.net)
            
        print("linear layer changes for each step: ")
        print(linear_layer_change)

        print("mec changes for each step: ")
        print(mec_change)  