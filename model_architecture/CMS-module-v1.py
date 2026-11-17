import torch
import torch.nn as nn

# Class lining out the standard architecture of the MLP blocks that will make up the MLP chain.
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size0, hidden1_size, output_size):
        super(MLP, self).__init__()
        self.step_count = 0
        self.hidden = nn.Linear(input_size, hidden_size0)  # Hidden layer
        self.hidden1 = nn.Linear(hidden_size0, hidden1_size)  # Hidden layer
        self.output = nn.Linear(hidden1_size, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.hidden(x))  # Apply ReLU activation to the hidden layer
        x = torch.relu(self.hidden1(x))
        x = torch.sigmoid(self.output(x))  # Apply sigmoid activation to the output layer
        self.step_count += 1
        if self.step_count >= 100:
            self.step_count 
        return x


class CMS(nn.Module):
    def __init__(self, device, token_dim, hidden_dim1, hidden_dim2, chunk_size, numOfSteps1, numOfSteps2, numOfSteps3, numOfSteps4, numOfSteps5):
        super(CMS, self).__init__()
        # Initialize of the MLP blocks (with identical parameters) that will make up the final chain.
        self.MLP1 = MLP(token_dim, hidden_dim1, hidden_dim2, token_dim).to(device)
        self.MLP2 = MLP(token_dim, hidden_dim1, hidden_dim2, token_dim).to(device)
        self.MLP3 = MLP(token_dim, hidden_dim1, hidden_dim2, token_dim).to(device)
        self.MLP4 = MLP(token_dim, hidden_dim1, hidden_dim2, token_dim).to(device)
        self.MLP5 = MLP(token_dim, hidden_dim1, hidden_dim2, token_dim).to(device)
        # Initialize the variables that will store the number of steps each layer will wait before updating their gradients.
        self.n1 = numOfSteps1
        self.n2 = numOfSteps2
        self.n3 = numOfSteps3
        self.n4 = numOfSteps4
        self.n5 = numOfSteps5
        # Initialize the step counter and the size it will reach before we end up resetting it.
        self.step_counter = 0
        self.step_reset = chunk_size *  numOfSteps5 * 10
    
    
    def forward_train(self, x):
        # Pass the input (which should be the token representations outputted by the titans modules) through the MLP chain.
        x = self.MLP1(x)
        x = self.MLP2(x)
        x = self.MLP3(x)
        x = self.MLP4(x)
        Result = self.MLP5(x)

        self.step_counter += 1 # Increment the step counter by one

        # If step_counter is a multiple of the number of steps a layer must wait before updating its parameters, unfreeze those parameters.
        # Otherwise, freeze them (and/or keep them frozen).
        if self.step_counter % self.n1 == 0:
            self.update_block_parameters(self.MLP1, True)
        else:
            self.update_block_parameters(self.MLP1, False)
        
        if self.step_counter % self.n2 == 0:
            self.update_block_parameters(self.MLP2, True)
        else:
            self.update_block_parameters(self.MLP2, False)
        
        if self.step_counter % self.n3 == 0:
            self.update_block_parameters(self.MLP3, True)
        else:
            self.update_block_parameters(self.MLP3, False)
        
        if self.step_counter % self.n4 == 0:
            self.update_block_parameters(self.MLP4, True)
        else:
            self.update_block_parameters(self.MLP4, False)

        if self.step_counter % self.n5 == 0:
            self.update_block_parameters(self.MLP5, True)
        else:
            self.update_block_parameters(self.MLP5, False)
        
        # If our step counter has reached the max size, reset it to zero.
        if self.step_counter == self.step_reset:
            self.step_counter = 0
        
        return Result # Return the output of the MLP chain.
    
    # Version of the forward pass used at test time. Parameters do not need to be frozen or unfrozen since we are
    # not updating our parameters during testing. We only need to pass the input through the chain.
    def forward(self, x):
        x = self.MLP1(x)
        x = self.MLP2(x)
        x = self.MLP3(x)
        x = self.MLP4(x)
        Result = self.MLP5(x)
        return Result
    
    # Freeze/Unfreeze an MLP blocks parameters.
    def update_block_parameters(self, block, unfreeze):
        for params in block.parameters():
                params.requires_grad = unfreeze
    
    # Function for resetting the step counter manually.
    def reset_steps(self):
        self.step_counter = 0
