import torch
import numpy as np


# Evaluation function
def evaluate(model, dataloader):
    ''' Evaluate a model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader to evaluate the model on.

    Returns:
        float: The average loss.
        float: The perplexity
    '''
    model.eval()
    total_loss = 0
    total_items = 0
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss.item()
            total_loss += loss * batch['input_ids'].shape[0]  # Assuming 'input_ids' is your input tensor
            total_items += batch['input_ids'].shape[0]
    
    average_loss = total_loss / total_items
    perplexity = np.exp(average_loss)
    
    return average_loss, perplexity