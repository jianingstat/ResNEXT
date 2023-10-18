### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.configs = configs # Model configurations (Recorded by self.configs)
        self.network = MyNetwork(self.configs)
        self.entropy = nn.CrossEntropyLoss()

    def model_setup(self):
        bestmodelfile = os.path.join(self.configs['save_dir'], 'model-best.ckpt')
        ckpt = torch.load(bestmodelfile, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(bestmodelfile), flush=True)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_configs = configs
        # Define optimizer and scheduler
        optimizer = torch.optim.SGD(self.network.parameters(), lr=training_configs['learning_rate'], momentum=0.9) 
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_configs['lr_scheduler_milestones'], gamma=0.1)
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        num_batches = num_samples // training_configs['batch_size']
        # Record the best validation accuracy if x_valid, y_valid is no None
        accuracy_valid_best = 0.85
        for epoch in range(1, training_configs['max_epoch']+1):
            # Record the running time, summation of loss and training accuracy in each epoch
            start_time = time.time()
            sum_entropy_loss = torch.tensor(0.0).to(device)
            preds = torch.tensor([]).to(device)
            labels = torch.tensor([]).to(device)
            # Shuffle data
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            
            # Training
            self.network.train()
            for i in range(num_batches):
                # Construct the current batch. Don't forget to use "parse_record" to perform data preprocessing.
                x_minibatch = []
                for j in range(training_configs['batch_size'] * i, training_configs['batch_size'] * (i+1)):
                    x_minibatch.append(parse_record(curr_x_train[j], training=True))
                y_minibatch = curr_y_train[range(training_configs['batch_size'] * i, training_configs['batch_size'] * (i+1))]
                x_minibatch = torch.tensor(np.array(x_minibatch)).float().to(device)
                y_minibatch = torch.tensor(y_minibatch).long().to(device)

                # Pass it through the network to get the predicted probabilities
                logits = self.network(x_minibatch)
                preds = torch.cat((preds, torch.argmax(logits, 1)), 0) 
                labels = torch.cat((labels, y_minibatch), 0)

                # L2 Regularization    
                l2_loss = torch.tensor(0.0).to(device)
                for param in self.network.parameters():
                    l2_loss += torch.norm(param)
                loss = training_configs['weight_decay'] * l2_loss + self.entropy(logits, y_minibatch)
                sum_entropy_loss += self.entropy(logits, y_minibatch)
                
                # Backward propagation                          
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Update learning rate if necessary
            scheduler.step()
            duration = time.time() - start_time
            print('Epoch {:d} : Duration {:.3f} seconds. Sum of loss {:.6f}. Training accuracy {:.4f}.'.format(epoch, duration, sum_entropy_loss, torch.sum(preds==labels)/labels.shape[0]), flush=True)

            # Save the model every few epochs (default=10)
            if epoch % self.configs['save_interval'] == 0:
                checkpoint_path = os.path.join(self.configs['save_dir'], 'model-%d.ckpt'%(epoch))
                os.makedirs(self.configs['save_dir'], exist_ok=True)
                torch.save(self.network.state_dict(), checkpoint_path)
                print("Sequential checkpoint has been created.", flush=True)
            
            # Compute the validation accuracy and save the model if it's better
            if (x_valid is not None) and (y_valid is not None):
                # Evaluation
                self.network.eval()
                preds = torch.tensor([]).to(device)
                for i in range(x_valid.shape[0]):
                    x_valid_i = parse_record(x_valid[i], training=False).reshape((1, 3, 32, 32))
                    x_valid_i = torch.tensor(x_valid_i).float().to(device)
                    logits = self.network(x_valid_i)
                    preds = torch.cat((preds, torch.argmax(logits, 1)), 0)
                y_valid = torch.tensor(y_valid).long().to(device)
                accuracy_valid = torch.sum(preds==y_valid)/y_valid.shape[0]
                if (accuracy_valid > accuracy_valid_best):
                    accuracy_valid_best = accuracy_valid
                    print('Validation Accuracy: {:.4f}'.format(accuracy_valid), flush=True)
                    checkpoint_path = os.path.join(self.configs['save_dir'], 'model-best.ckpt')
                    os.makedirs(self.configs['save_dir'], exist_ok=True)
                    torch.save(self.network.state_dict(), checkpoint_path)
                    print("Best model checkpoint has been created.", flush=True)

    def evaluate(self, x, y):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the saved model and Evaluate
        self.model_setup()
        self.network.eval()
        preds = torch.tensor([]).to(device)
        for i in range(x.shape[0]):
            x_test = parse_record(x[i], training=False).reshape((1, 3, 32, 32))
            x_test = torch.tensor(x_test).float().to(device)
            logits = self.network(x_test)
            preds = torch.cat((preds, torch.argmax(logits, 1)), 0)
        y = torch.tensor(y).long().to(device)
        print('Test Accuracy: {:.4f}'.format(torch.sum(preds==y)/y.shape[0]), flush=True)

    def predict_prob(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the saved model and Evaluate
        self.model_setup()
        self.network.eval()
        preds = []
        for i in range(x.shape[0]):
            x_test = parse_record(x[i], training=False).reshape((1, 3, 32, 32))
            x_test = torch.tensor(x_test).float().to(device)
            logits = self.network(x_test)
            probs = nn.functional.softmax(logits).cpu().detach().numpy()
            preds.append(probs)
        preds = np.squeeze(preds) 
        return preds

### END CODE HERE
