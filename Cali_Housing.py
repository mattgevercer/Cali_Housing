import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

#fetch data
housing=fetch_california_housing()
x=housing.data
y=housing.target

#split into training and testing sets

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#add callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.5):
      print("\nBelow 0.5 MSE so cancelling training!")
      self.model.stop_training = True

#initiate instance of callback
callbacks=myCallback()

#Buils sequential model
model=tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, activation='relu',input_shape=(8,)))
model.add(tf.keras.layers.Dense(8,activation='relu'))
model.add(tf.keras.layers.Dense(1))

#compile model
model.compile(optimizer='adam',loss='mean_squared_error')
history=model.fit(x_train,y_train, epochs=30, callbacks=[callbacks],
          validation_data=(x_test,y_test))

#visualize how validatio loss changes over the epochs
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]
epochs   = range(len(loss))
plt.plot  ( epochs, loss, label='Training Loss' )
plt.plot  ( epochs, val_loss, Label='Validation Loss' )
plt.title ('Training and Validation Loss Over Epochs')
plt.legend(loc="upper right")
plt.ylim(0, 20)
