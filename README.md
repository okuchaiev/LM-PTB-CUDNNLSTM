# LM-PTB-CUDNNLSTM
This code was initally taken from Tensorflow's PTB tutorial: https://www.tensorflow.org/tutorials/recurrent/ 
It was modified to use cuDNN's LSTM.
On Titan X (Pascal), I am getting about 9000 wps on large model
On GP100, I am getting over 12,100 wps on large model