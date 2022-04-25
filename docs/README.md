# README

You can replicate our results by opening the document within this directory titled 'notebook.ipynb'.
The notebook is configured to alert if you are utilizing Google Colab Pro.  Colab Pro is suggested as it guarantees GPU availability, which will significantly improve the performance of JAX.

Here are some variables you may wish to tweak:

### Changing sample size

In the category "Define Training Data" code block 11:

```
#Sample 5 birds

classes = set(random.sample(df['primary_label'].unique().tolist(),  5))

print(classes)
```

The third parameter of the set method will allow you to pick more birds species from the training data to work with.  The more species you choose will insure more time.

You must also adjust the first line of the following code block #24 in the category "Model Development" to mach the number of species you have chosen.

```
NUM_CLASSES = 5  #CHOSEN FROM THE 10 SAMPLE BIRDS RANDOMLY SELECTED ABOVE

class  VGG19(nn.Module):

@nn.compact
def  __call__(self, x, training):
x = self._stack(x,  64, training)
x = self._stack(x,  64, training)
x = nn.max_pool(x, window_shape=(2,  2), strides=(2,  2))
x = self._stack(x,  128, training)
x = self._stack(x,  128, training)
x = nn.max_pool(x, window_shape=(2,  2), strides=(2,  2))
x = self._stack(x,  256, training)
x = self._stack(x,  256, training)
x = nn.max_pool(x, window_shape=(2,  2), strides=(2,  2))
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = nn.max_pool(x, window_shape=(2,  2), strides=(2,  2))
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = self._stack(x,  512, training)
x = nn.max_pool(x, window_shape=(2,  2), strides=(2,  2))
x = x.reshape((x.shape[0],  -1))
x = nn.Dense(features=4096)(x)
x = nn.BatchNorm(use_running_average=not training)(x)
x = nn.relu(x)
x = nn.Dropout(0.5, deterministic=not training)(x)
x = nn.Dense(features=4096)(x)
x = nn.BatchNorm(use_running_average=not training)(x)
x = nn.relu(x)
x = nn.Dropout(0.5, deterministic=not training)(x)
x = nn.Dense(features=NUM_CLASSES)(x)
return x
@staticmethod
def  _stack(x, features, training, dropout=None):
x = nn.Conv(features=features, kernel_size=(3,  3), padding='SAME')(x)
x = nn.BatchNorm(use_running_average=not training)(x)
x = nn.relu(x)
return x
```

### Changing Number of Epochs

You may also choose to change the number of Epochs the VGG19 neural network undergoes.

This can be found in code block #31 of "Execute Training".

```
start = time.time()
final_state, training_accuracies = train(state, num_epochs=10)
print("Total time: ", time.time() - start,  "seconds")
```

To modify this, change the third parameter of the train function located on line 2.

Happy birding :)
