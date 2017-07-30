import os
import subprocess

scheduler_env = os.environ.copy()
scheduler_env.update({
  "DMLC_ROLE": "scheduler",
  "DMLC_PS_ROOT_URI": "127.0.0.1",
  "DMLC_PS_ROOT_PORT": "9000",
  "DMLC_NUM_SERVER": "1",
  "DMLC_NUM_WORKER": "1",
  "PS_VERBOSE": "2"
})
subprocess.Popen("python -c ‘import mxnet’", shell=True, env=scheduler_env)

server_env = os.environ.copy()
server_env.update({
  "DMLC_ROLE": "server",
  "DMLC_PS_ROOT_URI": "127.0.0.1",
  "DMLC_PS_ROOT_PORT": "9000",
  "DMLC_NUM_SERVER": "1",
  "DMLC_NUM_WORKER": "1",
  "PS_VERBOSE": "2"
})
subprocess.Popen(“python -c ‘import mxnet’”, shell=True, env=server_env)

os.environ.update({
  "DMLC_ROLE": "worker",
  "DMLC_PS_ROOT_URI": "127.0.0.1",
  "DMLC_PS_ROOT_PORT": "9000",
  "DMLC_NUM_SERVER": "1",
  "DMLC_NUM_WORKER": "1",
  "PS_VERBOSE": "2"
})
import mxnet
import numpy as np
import logging
kv_store = mxnet.kv.create(‘dist_async’)

logging.getLogger().setLevel(logging.DEBUG)
train_data = np.random.uniform(0, 1, [100, 2])
train_label = np.array([train_data[i][0] + 2 * train_data[i][1] for i in range(100)])
batch_size = 1
X = mxnet.sym.Variable(‘data’)
Y = mxnet.symbol.Variable(‘lin_reg_label’)
fully_connected_layer = mxnet.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
lro = mxnet.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")
model = mxnet.mod.Module(
 symbol = lro ,
 data_names=[‘data’],
 label_names = [‘lin_reg_label’]# network structure
)
#Evaluation Data
eval_data = np.array([[7,2],[6,10],[12,2]])
eval_label = np.array([11,26,16])
train_iter = mxnet.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name=’lin_reg_label’)
eval_iter = mxnet.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)
model.fit(train_iter, eval_iter,
 optimizer_params={‘learning_rate’:0.005, ‘momentum’: 0.9},
 num_epoch=50,
 eval_metric=’mse’,
 batch_end_callback = mxnet.callback.Speedometer(batch_size, 2),
 kvstore=kv_store)