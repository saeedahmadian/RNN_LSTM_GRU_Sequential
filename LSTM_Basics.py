import tensorflow as tf

"""
What are the inputs and output shapes of LSTMs?
In keras (Tensorflow), each LSTM layers consists of n LSTM cells and 
each cell returns the last hidden state (h_t) of the given sequence


              ______________               ________________
c_t-1-------> |             |-------->c_t  |                |--------> c_t+1
              |LSTM  Cell 1 |              | LSTM Cell2     |
h_t-1-------> |             |--------->h_t |                |--------> h_t+1
              |_____________|              |________________|
                     |                              |
                     |                              |
                     |                              |
                     |                              |
                    x_t                             x_t+1

By default when we feed an input sequence with (x_0,x_1,...,x_m) to and LSTM layer with n unit it will return
the last hidden state of the sequence h_m and since we asked for n units we will have n number of h_m.

Pay attention that units are different from cells,
each cell captures one step (single chain) of the given sequence and when you feed
a sequence with m-steps (m-chains), while within each unit the keras will create m LSTM cells.
for example in the above figure, we have 2 LSTM cells that they can be embedded into 1 unit


lets first see the LSTM layer with 1 unit 
(this doesn't mean the length of sequence is 1 it means that we only have 1 set of LSTM cells (len= seq_len))

"""

"Single hidden state as out put"
"sequence with 5 time-step (t1,t2,t3,t4,t5) or 5 sequence(s1,s2,s3,s4,s5) "
"Remember to work with LSTM layers your input must be in shape of (batch_size,time-steps,num_feature)"
"The number of features in case of forcasting are number of regressors (predictors) or vocabulary size in case of NLP  "


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################
"return the last hidden state"

"The values we consider are (t1=.1,t2=.2,t3=.3,t4=.4,t5=.5) and it must be 3 dimensional"
"batch size is one and also we have one feature (one regressor)"
sequence= tf.constant([[[.1],[.2],[.3],[.4],[.5]]],dtype=tf.float32)
in_ = tf.keras.layers.Input(shape=(5,1))
out_ = tf.keras.layers.LSTM(units=1,input_shape=(5,1))(in_)
model=tf.keras.Model(inputs=in_,outputs=out_)
"Now what happens if I feed sequence? t1...t5"
"The LSTM layer with 1 unit (lstm_cell_1,....,lstm_cell_5) will return h5"
h_last= model(sequence)
"""
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.14631353]], dtype=float32)>
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################
"return last hidden state and last cell state"
out_ = tf.keras.layers.LSTM(units=1,input_shape=(5,1),return_state=True)(in_)
model=tf.keras.Model(inputs=in_,outputs=out_)
h_all, h_last, c_last= model(sequence)
"""
h_all
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-0.18638201]], dtype=float32)>
h_last
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-0.18638201]], dtype=float32)>
c_last
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-0.33437306]], dtype=float32)>


"""

"Now the question is that what is the difference between the first h_all and the second h_last?"
"Generally h_all returns all the sequence states if the return_sequences are true else only the last"
"but when we use return state =True then in addition to h_all the LSTM returns the h_last,c_last "


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################
"return the all hidden states (all short memory information)"
out_ = tf.keras.layers.LSTM(units=1,input_shape=(5,1),return_sequences=True)(in_)
model=tf.keras.Model(inputs=in_,outputs=out_)

"Now  I can see the h1 to h5"
h_all= model(sequence)
"""
h_all
<tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
array([[[0.00394411],
        [0.01108295],
        [0.02098049],
        [0.03339735],
        [0.04821834]]], dtype=float32)>
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################
"return the all hidden states (all short memory information)"
"now what if I want to see c_last as well?"
out_ = tf.keras.layers.LSTM(units=1,input_shape=(5,1),return_sequences=True,return_state=True)(in_)
model=tf.keras.Model(inputs=in_,outputs=out_)
h_all,h_last,c_last= model(sequence)
"""
h_all
<tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
array([[[0.00394411],
        [0.01108295],
        [0.02098049],
        [0.03339735],
        [0.04821834]]], dtype=float32)>

h_last        
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.04821834]], dtype=float32)>

c_last
<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[0.09506062]], dtype=float32)>
"""

"Remeber in this case h_all[:,-1,:] is equal to h_last"


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
########################################################################
"""
Generally speaking LSTM layers with n units and sequence of m length  and d features 
would be graphically as below 
"""
"lest assume d is 1"

"""                                          
                                              _____                       ______
                                      c_0  --|      |--c_1--  ... --c_4--|      |--c_m
                            /unit_1    h_0 --|______|--h_1--  ... --h_4--|______|--h_m
                           /                  _____                       ______
                          /  unit_2    c_0 --|      |--c_1--  ... --c_4--|      |--c_m 
                         /             h_0 --|______|--h_1--  ... --h_4--|______|--h_m
t=t1...tm               /       .
x=[x_t1,x_t2,...,x_tm] /        .
                       \\        .
                        \\       .
                         \\      .
                          \\     .             _____                       ______
                           \\  unit_n   c_0 --|      |--c_1--  ... --c_4--|      |--c_m 
                            \\          h_0 --|______|--h_1--  ... --h_4--|______|--h_m


"""













