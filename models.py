import datetime

starttime = datetime.datetime.now()
from pyspark.conf import SparkConf
import argparse
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import OneVsRest
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('model_name', type=str, help='model name: SVM or LR or MLP')
parser.add_argument('output_path', type=str, help='output file name')
parser.add_argument('input_train', type=str, help='output file name')
parser.add_argument('input_test', type=str, help='output file name')
parser.add_argument('-local', type=bool, help='output file name', required=False)
args = parser.parse_args()
assert args.model_name in ["LR", "SVM", "MLP"]
assert args.input_train in ["train", "train_normed"]
assert args.input_test in ["test", "test_normed"]
conf = SparkConf()

if args.local:
    conf.setMaster("local").setAppName(args.model_name)
else:
    conf.setMaster("spark://192.168.cmds.cmds:7077").setAppName(args.output_path)

sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("WARN")

# spark = SparkSession.builder.master("local").appName("LR").getOrCreate()

if args.local:
    spark = SparkSession.builder.master("local").appName(args.model_name).getOrCreate()
else:
    spark = SparkSession.builder.master("spark://192.168.cmds.cmds:7077").appName(args.output_path).getOrCreate()

train = spark.read.load(args.input_train)
'''
train_df = train.toPandas()
train_df.to_csv('train.csv')
'''
test = spark.read.load(args.input_test)
if args.input_train == 'train':
    feature_name = 'tf_idf_vector'
else:
    feature_name = 'normed_tf_idf_vector'

if args.model_name == "SVM":
    lsvc = LinearSVC(featuresCol=feature_name, labelCol='label')
    ovr = OneVsRest(classifier=lsvc, featuresCol=feature_name, labelCol='label')
    ovrModel = ovr.fit(train)
    result = ovrModel.transform(test)
if args.model_name == "LR":
    lr = LogisticRegression(featuresCol=feature_name, labelCol='label')
    lrModel = lr.fit(train)
    result = lrModel.transform(test)
'''
if args.model_name == "MLP":
    input_size = train.head(cmds)[0].tf_idf_vector.size
    output_size = 14
    if args.local:
        layers = [input_size, output_size]
    else:
        layers = [input_size, 128, 128, output_size]
    trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, seed=0, featuresCol=feature_name,
                                             labelCol='label')
    # train the model
    model = trainer.fit(train)
    # compute accuracy on the test set
    result = model.transform(test)
    
if args.model_name == 'biLR':
    lr = LogisticRegression(featuresCol=feature_name, labelCol='label')
    ovr = OneVsRest(classifier=lr, featuresCol=feature_name, labelCol='label')
    ovrModel = ovr.fit(train)
    result = ovrModel.transform(test)
if args.model_name in ['decision_tree', 'random_forest']:
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(train)
    featureIndexer = VectorIndexer(inputCol="tf_idf_vector", outputCol="indexedFeatures").fit(train)
    if args.model_name == 'decision_tree':
        dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])
    if args.model_name == 'random_forest':
        rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
        labelConverter = IndexToString(inputCol="prediction", outputCol="label",
                                       labels=labelIndexer.labels)
        pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])
    model = pipeline.fit(train)
    result = model.transform(test)
'''
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracy = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(result)
print("Accuracy:", accuracy)
f = open(args.output_path, 'w')
f.write("Accuracy: " + str(accuracy) + '\n')
endtime = datetime.datetime.now()
time_elapsed = (endtime - starttime).seconds
f.write("Time elapsed: " + str(time_elapsed) + 's')
f.close()
