import numpy as np

train = np.random.random(1000)
val = np.random.random(100)
test = np.random.random(100)

target_train = np.random.random(1000)
target_val = np.random.random(100)
target_test = np.random.random(100)

from grape.plots import res idual_density_plot
residual_density_plot(train_pred=train, val_pred=val, test_pred=test,
                      train_target=target_train, val_target=target_val, test_target=target_test)