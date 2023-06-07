import torch
import torch.nn.functional as F

def tversky_index(y_pred, y_true, beta=0.5, epsilon=1e-5):
    """
    y_pred: (N, C, H, W)
    y_true: (N, H, W)
    beta: set to 0.5 by default. Here TI is equal to dice coefficient
    """
    intersection = (y_pred * y_true).sum()  # intersection = True Positives
    false_positive = (y_pred*(1-y_true)).sum() 
    false_negative = (y_true*(1-y_pred)).sum()
    
    tversky_index = (intersection/(intersection+(beta*false_positive)+((1-beta)*false_negative)+epsilon))
    
    return tversky_index


def calc_tversky_loss(y_pred, y_true, beta=0.5, gamma=1, num_classes=1):
    """
    Notation: N=Batch size, C=No.of channels, H=Height, W=Width
    y_pred: (N, C, H, W)
    y_true: (N, H, W): Conver to dimensions of (N, C, H, W)
    beta: set to 0.5 by default -> Tversky Index = Dice Coefficient
    gamma: set to 1 by default -> Focal Tversky = Tversky. Set to other than 1 to calculate focal tversky
    num_classes: set to 1 by default
    """
    y_true = y_true.type(torch.long)
#     print("Before one_hot: ", y_true.unique())
    if num_classes > 1:
        y_true = F.one_hot(y_true, num_classes)
#         print("After one__hot: ", y_true.unique())
        y_true = y_true.permute(0, 3, 1, 2)
    
    TI = tversky_index(y_pred, y_true, beta)
    
    loss = 1 - TI
    loss = loss.pow(gamma)
    
    return loss
    

    # check max value in the tensor - make sure it's normalized
    # make sure that dimensions match
    