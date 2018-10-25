import torch
import random
import numpy as np
import torch.nn as nn
from model import HardLSTM
from model import BasicLSTM
import torch.optim as optim
from model import ConcreteLSTM
import matplotlib.pyplot as plt
from torchtext import data, datasets
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator

def set_dropout_state(model, value):

    def set_dropout_state_in_module(module):

        if module.__class__.__name__.endswith('LSTMWithCDropout'):
            module.use_dropout = value

    model.apply(set_dropout_state_in_module)

#Metric
def binary_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum()/len(correct)
    return acc

#Training step
def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    set_dropout_state(model, True)
    
    for batch in iterator:
        
        optimizer.zero_grad()
                
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label) + model.regularisation_loss()
        
        acc = binary_accuracy(predictions, batch.label)
        
        #loss.backward()
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Model evaluation
def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    set_dropout_state(model, False)
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label) #+ model.regularisation_loss()
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def plotLoss ( train_loss , valid_loss ):

    ax = figure(figsize = (20,8)).gca()
    plt.title("Training Loss vs Epochs", fontsize = 20)
    plt.plot(train_loss, label = "Train")
    plt.plot(valid_loss, label = "Validation")
    plt.xlabel('Epochs',fontsize = 18)
    plt.ylabel('Loss', fontsize = 18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params('y', colors='black')
    plt.legend(fontsize = 18)
    plt.savefig('NLP2_Loss_3.png')

def plotAccuracy ( train_acc , valid_acc ):

    ax = figure(figsize = (20,8)).gca()
    plt.title("Accuracy vs Epochs", fontsize = 20)
    plt.plot(train_acc, label = "Train")
    plt.plot(valid_acc, label = "Validation")
    plt.xlabel('Epochs',fontsize = 18)
    plt.ylabel('Accuracy Percentage', fontsize = 18)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params('y', colors='black')
    plt.legend(fontsize = 18)
    plt.savefig('NLP2_Accuracy_3.png')


def main():

    ## Set data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Fix seed for reproducibility
    SEED = 1234
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    #Tokenize words from dataset
    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    #TEXT = data.Field(lower=True)
    #LABEL = data.Field(sequential=False)

    BATCH_SIZE = 64

    ## Load data ##

    #Split training, validation and test set
    train_data, valid_data, test_data = datasets.SST.splits(TEXT, LABEL)
    
    #print(vars(train_data.examples[0]))
    #print(vars(valid_data.examples[0]))
    #print(vars(test_data.examples[0]))

    # Pretrained Globe vectors
    #TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")

    TEXT.build_vocab(train_data, max_size=25000) # 250000 most common words
    LABEL.build_vocab(train_data)

    #Prepare batches

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size=BATCH_SIZE,
        device=device)

    ### Model stage 

    ## Set model hyperparameters
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1

    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    ##### Define model

    # Basic RNN
    #model = HardLSTM(vocab_size=INPUT_DIM, output_dim = OUTPUT_DIM)  

    # Basic LSTM
    #model = BasicLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)

    # Concrete Dropout LSTM
    dropout_data = {'gpu_id': 0, 'wr': 1e-6, 'dr': 2e-5,
                    'init_min': 0.05, 'init_max': 0.5}

    model = ConcreteLSTM(vocab_size = INPUT_DIM, output_dim = OUTPUT_DIM, dropout = dropout_data).to(device)

    # Optimizer
    #optimizer = optim.RMSprop(model.parameters(), lr=1e-1)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr = 5e-2)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # If there's GPU available
    model = model.to(device)
    criterion = criterion.to(device)

    # Start training
    N_EPOCHS = 20

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []


    for epoch in range(N_EPOCHS):

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        train_losses.append(train_loss)
        val_losses.append(valid_loss)
        train_accs.append(train_acc)
        val_accs.append(valid_acc)

        print(np.array([module.p.cpu().data.numpy()[0] for module in model.modules() if hasattr(module, 'p')]) )
        
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.5f}% | Val. Loss: {valid_loss:.9f} | Val. Acc: {valid_acc*100:.9f}% |')

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

    plotLoss(train_losses,val_losses)
    plotAccuracy(train_accs,val_accs)


if __name__ == '__main__':
    main()