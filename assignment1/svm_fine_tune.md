# Hyperparameter optimization progress in SVM
## Step 1
First, there are 2 hyperparameters to optimize: learning_rate and regularization_strength.    
And we have 2 strategies for opimization: grid search and random search.
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171107150745579.png)
Using the gird search means you may take the risk of missing the best value.    
Random search randomly samples in an interval, and the sampled values are sort of inelegent generally(e.g. learning_rate = 2.120143e-06).   
So i decide to use random search for long-range search. When the interval becomes small, i  will change my strategy to grid search to find the relatively best and elegant value.
Start-up code snippet:
```python
learning_rates = [1e-7, 5e-5]
regularization_strengths = [2.5e4, 5e4]
num_iters = 200
# Random search
max_count = 50
for i in range(max_count):
    # Choice learning rate and regularization strength randomly
    lr = random.uniform(learning_rates[0], learning_rates[1])
    reg = random.uniform(regularization_strengths[0],
                         regularization_strengths[1])
    svm = LinearSVM()
    _ = svm.train(X_train, y_train, learning_rate=lr, reg=reg,
                  num_iters=num_iters)
    train_pred = svm.predict(X_train)
    train_acc = np.mean((train_pred == y_train))
    val_pred = svm.predict(X_val)
    val_acc = np.mean((val_pred == y_val))
    # Store
    results[(lr, reg)] = (train_acc, val_acc)
    if (val_acc > best_val):
        best_val = val_acc
        best_svm = svm

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
    lr, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```
Output:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171107154011526.png)   
Use the code snippet below to visualize the result:
```python
# Visualize the cross-validation results
import math
x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.subplot(2, 1, 1)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')

# plot validation accuracy
colors = [results[x][1] for x in results] # default size of markers is 20
plt.subplot(2, 1, 2)
plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.show()
```
Output(Brighter point represents higher accuracy):   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171107154202246.png)
## Step 2
You may note that there is a worrying fact: The best value is in the lower left corner of the plot. That means our parameter intervals may be inappropriate.   
So i shift the learning_rates and regularization_strengths intervals. And the num_iters should increase a little bit due to the left shift of the learning_rates.   
```python
learning_rates = [1e-8, 5e-7]
regularization_strengths = [5e3, 5e4]
num_iters = 500
```
Output:    
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171107160650378.png)   
It seems we found some appropriate range of the learning_rates.   
Visualize it:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171107161026713.png)
## Step 2+1
I call it 'Setp 2+1' instead of 'Step 3', because this step is the refining of
the 'Step 2' and may repeat for several times.
According to the result of the previous step, i shrink the learning_rate interval and apply left shift upon regularization_strengths. The num_iters is increased to 5000.

```python
learning_rates = [1e-8, 1e-7]
regularization_strengths = [5e2, 3e4]
num_iters = 5000
```
Output:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171108135230624.png)   
It seems that learning_rates within [1e-8, 3e-8] and regularization_strengths within [5e3, 3e4] behave well.    
Visualize it:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171108135349691.png)   
Just repeat the progress.   
```python
learning_rates = [1e-8, 3e-8]
regularization_strengths = [5e3, 3e4]
num_iters = 5000
```
Output:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-2017110912322905.png)    
Visualize it:
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-2017110912324417.png)    
10^(-7.8)≈1.585\*10^(-8)    
Refine the learning_rates interval. Increase the num_iters to 10k, because we are close to the final result.
```python
learning_rates = [1.5e-8, 3e-8]   
regularization_strengths = [5e3, 3e4]
num_iters = 10000
```
Output:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171109134707936.png)   
Visualize it:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171109134739199.png)   
10^4.2≈15849    
Refine the regularization_strengs interval.   
```python
learning_rates = [1.5e-8, 3e-8]   
regularization_strengths = [5e3, 1.5e4]
num_iters = 10000
```
Output:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171109154018115.png)   
Visualize it:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171109154031659.png)   
10^(-7.6)≈2.512\*10^(-8)
It seems we find the final intervals of the 2 hyperparameters.
## Step 3
Here is the final intervals:    
```python
learning_rates = [1.5e-8, 2.6e-8]
regularization_strengths = [5e3, 1e4]
num_iters = 10000
```
And we now use the grid search strategy to make the results more elegent.   
Code snippet:   
```python
# Grid search
lr_grid = np.arange(learning_rates[0], learning_rates[1], 0.1e-8)
reg_grid = np.arange(regularization_strengths[0], regularization_strengths[1],
                     1e3)
for lr in lr_grid:
    for reg in reg_grid:
        svm = LinearSVM()
        _ = svm.train(X_train, y_train, learning_rate=lr, reg=reg,
                      num_iters=num_iters)
        train_pred = svm.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        val_pred = svm.predict(X_val)
        val_acc = np.mean(val_pred == y_val)
        # Store
        results[(lr, reg)] = (train_acc, val_acc)
        if (val_acc > best_val):
            best_val = val_acc
            best_svm = svm
```
And the best validation accuracy is 0.402000   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171110195127280.png)   
Visualize it:
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171110195140682.png)
And our best svm gives 0.382000 accuracy on test set:   
![](http://oo1ncxa8y.bkt.clouddn.com//markdown-img-paste-20171110195303636.png)
