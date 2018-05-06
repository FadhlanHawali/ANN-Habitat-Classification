import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

zoo = pd.read_csv("./input/zoo2.data.csv")
zoo.head()
print("This ZOO dataset is consised of",len(zoo),"rows.")

#menghitung jumlah animal berdasarkan class nya
sns.countplot(zoo['habitat'],label="Count")

#plotting dari 16 fitur
corr = zoo.loc[:,('animal_name','airborne','aquatic','breathes','fins','class_type')].corr()
colormap = sns.diverging_palette(220, 10, as_cmap = True)
plt.figure(figsize=(14,14))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 12},
            cmap = colormap, linewidths=0.1, linecolor='white')
plt.title('Correlation of ZOO Features', y=1.05, size=15)
#plt.show()
#kolom prediksi untuk training. Itu ada -1 karena kolom yang class type nya dihilangkan
x_data = zoo.loc[:,('animal_name','airborne','aquatic','breathes','fins','class_type')]
x_data.head()
#untuk membandingkan dengan hasil prediksi
y_data = zoo.iloc[:,-1:]
y_data.head()

#(101,17) , harusnya kan 18 karena class type nya dihapus jadi 17 doang
print("Feature Data :", x_data.shape)
#(101,1), labelnya sendiri kan bakalan cuma 1 kolom
print("Label Data :", y_data.shape)

#Data displit jadi 70:30, kayak yang dijelasin pake K-Map
train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3, random_state=100, stratify=y_data)
print("Training Data has",train_x.shape)
print("Testing Data has",test_x.shape)


train_name = train_x['animal_name' ] #save animal name untuk hasil
test_name = test_x['animal_name'] #save animal buat test

train_x = train_x.iloc[:,1:] #animal_name di drop karena tidak penting 
test_x = test_x.iloc[:,1:]

print("Training Data has",train_x.shape)
print("Testing Data has",test_x.shape)

X = tf.placeholder(tf.float32, [None,5]) #zoo data kan punya 16 fitur
Y = tf.placeholder(tf.int32, [None, 1]) #outputnya cuma 1 kolom, yaitu class animal nya
Y_one_hot = tf.one_hot(Y, 3)  # one hot encoding
Y_one_hot = tf.reshape(Y_one_hot, [-1, 3])

#W = weight (16,7), 16 karena ada 16 feature, sedangkan 7 karena mau ada 7 hasil berdasarkan class nya
W = tf.Variable(tf.random_normal([5, 3],seed=0), name='weight')
#bias, kenapa 7? karena mau ada 7 layer (tipe)
b = tf.Variable(tf.random_normal([3],seed=0), name='bias')
#Output = Weight * Input + Bias
logits = tf.matmul(X, W) + b
# hypothesis = tf.nn.softmax(logits)
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis)))

train  = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)
#train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost) 

#compare original vs prediksi
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        sess.run(train, feed_dict={X: train_x, Y: train_y})
        if step % 1000 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: train_x, Y: train_y})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
            
    train_acc = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
    test_acc,test_predict,test_correct = sess.run([accuracy,prediction,correct_prediction], feed_dict={X: test_x, Y: test_y})
    print("Model Prediction =", train_acc)
    print("Test Prediction =", test_acc)

sub = pd.DataFrame()
sub['Name'] = test_name
sub['Predict_Type'] = test_predict
sub['Origin_Type'] = test_y
sub['Correct'] = test_correct
sub

sub[['Name','Predict_Type','Origin_Type','Correct']].to_csv('submission.csv',index=False)
