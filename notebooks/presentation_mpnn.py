import pandas as pd
from grape.utils import DataSet, split_data, train_model, test_model, pred_metric
from torch import nn
from grape.models import MEGNet_gnn
from grape.utils import EarlyStopping
from torch.optim import lr_scheduler
import random
import dgl
import torch
import os


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    # np.random.seed(seed) # turn it off during optimization
    random.seed(seed)
    torch.manual_seed(seed)  # annote this line when ensembling
    dgl.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True # Faster than the command below and may not fix results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    set_seed(0)

    # Load the data
    df = pd.read_excel('/Users/faerte/Desktop/grape/data-sets/BMP_model_predictions.xlsx')

    # Load it into DataSet format
    data = DataSet(smiles=df.SMILES, target=df.Target, filter=False)


    # convert given labels to a list of numbers and split dataset
    labels = df.Tag.apply(lambda x: ['Train', 'Val', 'Test'].index(x)).to_list()
    train, val, test = split_data(data, custom_split=labels)

    # Define model
    node_hidden_dim = 80
    batch_size = 32

    model = MEGNet_gnn(node_in_dim=data.num_node_features, edge_in_dim=data.num_edge_features,
                       depth=1)


    print('Full model:\n--------------------------------------------------')
    print(model)

    device = torch.device('cpu')

    loss_func = nn.functional.mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=10**(-2.6), weight_decay=1e-6)

    early_stopper = EarlyStopping(patience=50, model_name='best_model_mpnn')

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, min_lr=0.0000000000001, patience=30)

    train_loss, val_loss = train_model(model = model,
                                     loss_func = 'mse',
                                     optimizer = optimizer,
                                      train_data_loader= train,
                                      val_data_loader = val,
                                      batch_size=batch_size,
                                      epochs=500,
                                      early_stopper=early_stopper,
                                       scheduler=scheduler)

    from grape.plots import loss_plot
    from matplotlib import pyplot as plt
    loss_plot([train_loss, val_loss], ['train loss', 'test loss'], early_stopper.stop_epoch)
    plt.show()

    model.load_state_dict(torch.load('best_model_mpnn.pt'))

    preds = test_model(model=model,
                       test_data_loader=test,
                       loss_func=None,
                       return_latents=True)

    pred_metric(prediction=preds,target=test.y, metrics='all', print_out=True, rescale_data=data)

    train_preds = test_model(model=model, loss_func='mae',test_data_loader=train)
    val_preds = test_model(model=model, loss_func='mae',test_data_loader=val)
    test_preds = test_model(model=model, loss_func='mae',test_data_loader=test)

    overall = 0
    overall += pred_metric(prediction=train_preds,target=train.y, metrics='r2', print_out=False)['r2']
    overall += pred_metric(prediction=val_preds,target=val.y, metrics='r2', print_out=False)['r2']
    overall += pred_metric(prediction=test_preds,target=test.y, metrics='r2', print_out=False)['r2']
    print(f'Overall R2: {overall/3}')

    overall = 0
    overall += pred_metric(prediction=train_preds,target=train.y, metrics='mae', print_out=False, rescale_data=data)['mae']
    overall += pred_metric(prediction=val_preds,target=val.y, metrics='mae', print_out=False, rescale_data=data)['mae']
    overall += pred_metric(prediction=test_preds,target=test.y, metrics='mae', print_out=False, rescale_data=data)['mae']
    print(f'Overall MAE: {overall/3}')

    overall = 0
    overall += pred_metric(prediction=train_preds,target=train.y, metrics='rmse', print_out=False, rescale_data=data)['rmse']
    overall += pred_metric(prediction=val_preds,target=val.y, metrics='rmse', print_out=False, rescale_data=data)['rmse']
    overall += pred_metric(prediction=test_preds,target=test.y, metrics='rmse', print_out=False, rescale_data=data)['rmse']
    print(f'Overall RMSE: {overall/3}')


    ##################################################################
    ########     Model test on n-alkanes                     #########
    ##################################################################


    #df_test = pd.read_excel('/Users/faerte/Desktop/grape/data-sets/n-alkanes.xlsx')
    #data_test = DataSet(smiles=df_test.SMILES, target=df_test.nC, filter=False)

    #alkanes_pred = test_model(model=model, loss_func='mae',test_data_loader=data_test.data)

