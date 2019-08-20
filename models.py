import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.init import constant_, normal_

class MLP_Mdl(nn.Module):
    def __init__(self, neurons_list): #including input neurons and output neurons
        super().__init__()

        self.neurons_list = neurons_list        
        for i in range(len(neurons_list)-1):
            self.__setattr__(f'fc{i}', nn.Linear(self.neurons_list[i], self.neurons_list[i+1]))

            normal_(self.__getattr__(f'fc{i}').weight, 0, 2/self.neurons_list[i])
            constant_(self.__getattr__(f'fc{i}').bias, 0)
        
        
    def forward(self, input):
        x = input.view(-1, self.neurons_list[0])
        for i in range(len(self.neurons_list)-1):
            x = self.__getattr__(f'fc{i}')(x)
            x = F.relu(x) 
        return x

class simple_rnn_2(nn.Module):
    def __init__(self, input_size, output_size, n_layers = 1, hidden_size=128):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, self.hidden_size,
                                num_layers=self.n_layers, 
                                nonlinearity='relu',
                                batch_first=True)

        self.dropout_layer = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        
        normal_(self.linear1.weight, 0, 1/self.hidden_size)
        constant_(self.linear1.bias, 1)
    
    def forward(self, input):
        x = input.unsqueeze(0)
        self.rnn.flatten_parameters()
        self.hidden0 = self.initHidden().cuda()
        
        out, hn = self.rnn(x, self.hidden0)  
        out = self.dropout_layer(out)             
        out = self.linear1(out)
        
        return out.view(-1, self.output_size)

    def initHidden(self):
        return Variable(torch.randn(self.n_layers, 1, self.hidden_size), requires_grad=True)


class temporal_conv(nn.Module):
    def __init__(self, ): #including input neurons and output neurons
        super().__init__()

class simple_rnn_3(nn.Module):
    def __init__(self, input_size, output_size, n_layers = 1, hidden_size=128):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_size, self.hidden_size,
                                num_layers=self.n_layers, 
                                nonlinearity='relu',
                                batch_first=True)

        self.dropout_layer = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(self.hidden_size, self.output_size)
        
        normal_(self.linear1.weight, 0, 1/self.hidden_size)
        constant_(self.linear1.bias, 1)
    
    def forward(self, input):
        x = input
        self.rnn.flatten_parameters()
        self.hidden0 = self.initHidden(x.shape[0])
        
        out, hn = self.rnn(x, self.hidden0)  
        out = self.dropout_layer(out)             
        out = self.linear1(out[:, -1, :])
        
        return out.view(-1, self.output_size)

    def initHidden(self,minibatch):
        return Variable(torch.randn(self.n_layers, minibatch, self.hidden_size), requires_grad=True).cuda()

class simple_lstm_2(nn.Module):
    def __init__(self, input_size, output_size, n_layers = 3, hidden_size=128, bidirectional=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bi = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, 
                            self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        
        self.dropout_layer = nn.Dropout(p=0.2)

        self.linear1 = nn.Linear(self.hidden_size*self.bi, self.output_size)

        normal_(self.linear1.weight, 0, 1/(self.bi*self.hidden_size))
        constant_(self.linear1.bias, 0)

    def forward(self, input):
        x = input.unsqueeze(0)
        self.lstm.flatten_parameters()
        
        hidden0 = self.initHidden()    
        outs, (ht, ct) = self.lstm(x, hidden0)
        out = self.dropout_layer(outs)
        out = self.linear1(out)
        return out.view(-1, self.output_size)

    def initHidden(self):
        random_var1 = Variable(torch.randn(self.n_layers*self.bi, 1, self.hidden_size), requires_grad=True)
        random_var2 = Variable(torch.randn(self.n_layers*self.bi, 1, self.hidden_size), requires_grad=True)
        return (random_var1.cuda(), random_var2.cuda())
        
class simple_lstm_3(nn.Module):
    def __init__(self, input_size, output_size, n_layers = 3, hidden_size=128, bidirectional=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bi = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, 
                            self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        
        self.dropout_layer = nn.Dropout(p=0.2)

        self.linear1 = nn.Linear(self.hidden_size*self.bi, self.output_size)

        normal_(self.linear1.weight, 0, 1/(self.bi*self.hidden_size))
        constant_(self.linear1.bias, 0)

    def forward(self, input):
        x = input#.unsqueeze(0)
        self.lstm.flatten_parameters()
        
        hidden0 = self.initHidden(x.shape[0])    
        outs, (ht, ct) = self.lstm(x, hidden0)
        out = self.dropout_layer(outs)
        out = self.linear1(out[:, -1, :])
        return out.view(-1, self.output_size)

    def initHidden(self, minibatch):
        random_var1 = Variable(torch.randn(self.n_layers*self.bi, minibatch, self.hidden_size), requires_grad=True)
        random_var2 = Variable(torch.randn(self.n_layers*self.bi, minibatch, self.hidden_size), requires_grad=True)
        return (random_var1.cuda(), random_var2.cuda())
        
class simple_tcn_2(nn.Module):
    def __init__(self, in_feature_size, out_feature_size):
        super().__init__()
        
        hidden_feature_size = 128
        self.output_size = out_feature_size

        self.temporal_conv1 = nn.Conv1d(1, hidden_feature_size, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv2 = nn.Conv1d(hidden_feature_size, hidden_feature_size*2, kernel_size=3, stride=1, padding=1)  #16, 4
        self.dropout_layer = nn.Dropout(p=0.2)

        self.temporal_conv3 = nn.Conv1d(hidden_feature_size*2, hidden_feature_size*4, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv4 = nn.Conv1d(hidden_feature_size*4, hidden_feature_size*4, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv5 = nn.Conv1d(hidden_feature_size*4, self.output_size, kernel_size=3, stride=1, padding=1)  #8, 1
        self.gmp = nn.AdaptiveMaxPool1d(output_size=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

        normal_(self.temporal_conv1.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv1.bias, 0)
        normal_(self.temporal_conv2.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv2.bias, 0)

        normal_(self.temporal_conv3.weight, 0, 2/(2*hidden_feature_size))
        constant_(self.temporal_conv3.bias, 0)
        normal_(self.temporal_conv4.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv4.bias, 0)
        normal_(self.temporal_conv5.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv5.bias, 0)

    def forward(self, input):            # input: batch_size x in_feature_size x timesteps
        out= self.temporal_conv1(input.unsqueeze(1))  # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv2(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv3(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv4(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv5(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out = self.gmp(F.relu(out))          # out  : batch_size x out_feature_size x 1
        return out.view(-1, self.output_size)


      
class simple_tcn_3(nn.Module):
    def __init__(self, in_feature_size, out_feature_size):
        super().__init__()
        
        hidden_feature_size = 128
        self.output_size = out_feature_size

        self.temporal_conv1 = nn.Conv1d(in_feature_size, hidden_feature_size, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv2 = nn.Conv1d(hidden_feature_size, hidden_feature_size*2, kernel_size=3, stride=1, padding=1)  #16, 4
        self.dropout_layer = nn.Dropout(p=0.4)

        self.temporal_conv3 = nn.Conv1d(hidden_feature_size*2, hidden_feature_size*4, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv4 = nn.Conv1d(hidden_feature_size*4, hidden_feature_size*4, kernel_size=3, stride=1, padding=1)  #16, 4
        self.temporal_conv5 = nn.Conv1d(hidden_feature_size*4, self.output_size, kernel_size=3, stride=1, padding=1)  #8, 1
        self.gmp = nn.AdaptiveMaxPool1d(output_size=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)

        normal_(self.temporal_conv1.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv1.bias, 0)
        normal_(self.temporal_conv2.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv2.bias, 0)

        normal_(self.temporal_conv3.weight, 0, 2/(2*hidden_feature_size))
        constant_(self.temporal_conv3.bias, 0)
        normal_(self.temporal_conv4.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv4.bias, 0)
        normal_(self.temporal_conv5.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv5.bias, 0)

    def forward(self, input):            # input: batch_size x in_feature_size x timesteps
        out= self.temporal_conv1(input.transpose(1,2))#.unsqueeze(1))  # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv2(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv3(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv4(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out= self.temporal_conv5(F.relu(self.max_pool(out)))    # out  : batch_size x out_feature_size x (timesteps -stride)!
        out = self.gmp(F.relu(out))          # out  : batch_size x out_feature_size x 1
        return out.view(-1, self.output_size)



class simple_tcn_skip_2(nn.Module):
    def __init__(self, in_feature_size, out_feature_size):
        super().__init__()
        
        hidden_feature_size = 128
        self.output_size = out_feature_size

        self.temporal_conv1 = nn.Conv1d(in_feature_size, hidden_feature_size, kernel_size=3, stride=1, padding=1)  
        self.temporal_conv2 = nn.Conv1d(hidden_feature_size, hidden_feature_size*2, kernel_size=3, stride=1, padding=1)  
        self.temporal_conv3 = nn.Conv1d(hidden_feature_size*2, hidden_feature_size, kernel_size=3, stride=1, padding=1)  
        self.batch_norm1 = nn.BatchNorm1d(hidden_feature_size)
        
        self.temporal_conv4 = nn.Conv1d(hidden_feature_size, hidden_feature_size*2, kernel_size=3, stride=1, padding=1)  
        self.temporal_conv5 = nn.Conv1d(hidden_feature_size*2, hidden_feature_size*4, kernel_size=3, stride=1, padding=1) 
        self.temporal_conv6 = nn.Conv1d(hidden_feature_size*4, hidden_feature_size*2, kernel_size=3, stride=1, padding=1) 
        self.batch_norm2 = nn.BatchNorm1d(hidden_feature_size*2)

        self.temporal_conv7 = nn.Conv1d(hidden_feature_size*2, self.output_size, kernel_size=3, stride=1, padding=1)  
        
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.avg_pool = nn.AvgPool1d(kernel_size=4)
        self.gmp = nn.AdaptiveMaxPool1d(output_size=1)
       
        self.dropout_layer = nn.Dropout(p=0.2)

        normal_(self.temporal_conv1.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv1.bias, 0)
        normal_(self.temporal_conv2.weight, 0, 2/hidden_feature_size)
        constant_(self.temporal_conv2.bias, 0)

        normal_(self.temporal_conv3.weight, 0, 2/(2*hidden_feature_size))
        constant_(self.temporal_conv3.bias, 0)
        normal_(self.temporal_conv4.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv4.bias, 0)
        normal_(self.temporal_conv5.weight, 0, 2/(4*hidden_feature_size))
        constant_(self.temporal_conv5.bias, 0)
        normal_(self.temporal_conv6.weight, 0, 2/(2*hidden_feature_size))
        constant_(self.temporal_conv6.bias, 0)
        normal_(self.temporal_conv7.weight, 0, 2/(2*hidden_feature_size))
        constant_(self.temporal_conv7.bias, 0)

    def forward(self, input):            
        out1 = self.temporal_conv1(input.transpose(1,2))  
        out = self.temporal_conv2(F.relu(self.max_pool(out1)))
        out = self.temporal_conv3(F.relu(self.max_pool(out)))
        out = self.batch_norm1(out)
        out3 = self.dropout_layer(out)

        combined = torch.cat((out1, out3), dim=-1)

        out4= self.temporal_conv4(F.relu(self.avg_pool(combined)))    
        out= self.temporal_conv5(F.relu(self.max_pool(out4)))    
        out= self.temporal_conv6(F.relu(self.max_pool(out)))    
        #out = self.batch_norm2(out)
        out6 = self.dropout_layer(out)

        combined = torch.cat((out4, out6), dim=-1)

        out= self.temporal_conv7(F.relu(self.avg_pool(combined)))
        out= self.gmp(F.relu(out))          

        return out.view(-1, self.output_size)