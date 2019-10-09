from keras.layers import Input
from keras.models import Model
from Channel_Wise_Attention import *
from Spatial_Attention import  *
inputs=Input(shape=(256,256,3))
can=Channel_Wise_Attention()(inputs)
sp=Spatial_Attention()(can)
model=Model(inputs=inputs,outputs=sp)
model.compile(optimizer="adam",loss="binary_crossentropy")
model.summary()