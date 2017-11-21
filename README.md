# Predictive Maintenance
Leveraging Deep Learning Solutions for Predictive Maintenance of Batteries in Industrial Datasets

# Problem
Predictive maintenance which is an age old problem, have been gaining attention of late due to the popularity of Internet of Things and applications of machine learning. Over the past few years, deep learning solutions have produced state of the art results in various domains. This thesis aims to provide solutions for providing predictive maintenance of batteries in industrial machines, in a way that both maintenance and repair activities can be predicted beforehand. It leverages deep learning solutions by exploring the application of sequence learning to represent the failure profiles and understand the patterns that lead to failures in batteries. It tries to address these problems by using recurrent neural networks.

# Dataset

A test battery dataset for calculating remaining useful life of aero engines was considered which contains failure events for about half a million events from last couple of years. Batteries are continuously cycled with randomly generated current profiles. Reference charging and discharging cycles are also performed after a fixed interval of randomized usage in order to provide reference benchmarks for battery state of health.

• Battery Data: This dataset contains 931 atrributes which consists of features like time, battery type, state of charge, historical state of charge, stat aging insistance value, stat aging capacity value, number of clamping cycles, tension value, charge time, discharge time, charge time value, discharge time value, sum of state of charge, displayed state of charge, charging cycle time, discharging cycle time, state of health, etc.

• Aero Engine Data: Aero Engine Data has 155 attributes which all engine related data like engine model, start time, end time, down time, battery type, timestamp

• Failure Events: This dataset contains 118 attributes which contain all the failure events , source, source type and timestamps of failures for all types of batteries.

# Approach
A binary classification approach has been used to label the battery attributes with the aero engine attributes using the state of health of battry as the failure metric to prognostically determine the failure patterns. The failure events of the batteries were used to train the model and then subsequently tested.

# Model
Grid LSTM has been used to define a novel two-dimensional translation model, the Reencoder where translation is done in a two-dimensional mapping. One dimension processes the source sequence whereas the other dimension produces the target sequence.

# Implementation
Multi layered LSTM (2-LSTM and Stacked LSTM) is trained. The first layer is a Grid LSTM layer with 100 units followed by another Grid LSTM layer with 50 units. Dropout is also applied after each LSTM layer to control overfitting. Final layer is a Dense output layer with single unit and sigmoid activation since this is a binary classification problem. Mini-batches of size 15 are used and the network is optimized using Adam and a learning rate decay of 0.95 (model.py).

# Apache Kafka Consumer
The Apache Kafka consumer (consumer.py) consumes the failure events from different timestamps. 

# Apache Spark

The models written in TensorFlow run on top of Apache Spark (pySpark) to do both hyperparameter tuning and deployment at scale (processing_spark.py)

⦿ Hyperparameter Tuning : Apache Spark was used to broadcast the common elements such as data and model description, and then schedule the individual repetitive computations across a cluster of machines in a fault-tolerant manner

⦿ Deploying models at scale: Apache Spark was used to apply trained neural network model on our dataset.
The model is first distributed to the workers of the clusters, using Spark’s built-in broadcasting mechanism. Then this model is loaded on each node and applied to the sequences.


# Saving Results to Amazon S3
The results of the model were exported to Amazon S3 (export.py).


# Serializing Data
Attributes of battery data was serialized using Google Protocol Buffer into tf.Sequence.example format as Tensorflow is being used for the sequence to sequence labelling (sequence.py)

# Results
Grid LSTM outperformed traditional LSTM for higher dimensional spatio temporal data when input data is noisy with an accuracy of 0.77 as compared to 0.73 for LSTM






