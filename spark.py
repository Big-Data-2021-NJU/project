# coding: utf-8
import os
import pandas as pd
import jieba
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType

# from hdfs import InsecureClient
# client = InsecureClient('hdfs://master001:9000')
# fnames = client.list('/data/2021spring/3-news_classification')
sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")
spark = SparkSession.builder.appName("DataFrame").getOrCreate()
# stopwords = open("hdfs://master001:9000/data/2021spring/3-news_classification/cn_stopwords.txt", encoding='UTF-8')
stopwords = spark.read.text("hdfs://master001:9000/data/2021spring/3-news_classification/cn_stopwords.txt")
# stopwords = spark.read.text("3-news_classification_sample/cn_stopwords.txt")
stopwords_list = list(stopwords.select('value').toPandas().value)
# stopwords_list = stopwords.read().split('\n')


URI = sc._gateway.jvm.java.net.URI
Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
fs = FileSystem.get(URI("hdfs://master001:9000"), Configuration())
train_dir = '/data/2021spring/3-news_classification/train'


def load_data(directory):
    status = fs.listStatus(Path(directory))
    dataframe = None
    i = 0  # label
    count = 0
    for fileStatus in status:
        dir_path = fileStatus.getPath()
        dir_status = fs.listStatus(Path(str(dir_path)[len("hdfs://master001:9000"):]))
        temp_df = None
        for text_status in dir_status:
            file_path = text_status.getPath()
            count += 1
            print(count)
            if temp_df is None:
                temp_df = spark.read.text(str(file_path), wholetext=True)
                # assert(0)
            else:
                temp_df1 = spark.read.text(str(file_path), wholetext=True)
                temp_df = temp_df.union(temp_df1)
        temp_df = temp_df.withColumn("label", F.lit(i))
        i += 1
        if dataframe is None:
            dataframe = temp_df
        else:
            dataframe = dataframe.union(temp_df)
    return dataframe


df = load_data(train_dir)
'''
data_dir = "hdfs://master001:9000/data/2021spring/3-news_classification/train"
dirs = os.listdir(data_dir)

for directory in dirs:  # 遍历文件夹
    temp_dir = data_dir + '/' + directory
    files = os.listdir(temp_dir)
    temp_df = None
    for file in files:
        if temp_df is None:
            temp_df = sc.read.text(temp_dir + '/' + file, wholetext=True)
        else:
            temp_df1 = sc.read.text(temp_dir + '/' + file, wholetext=True)
            temp_df = temp_df.union(temp_df1)

    temp_df = temp_df.withColumn("label", F.lit(dirs.index(directory)))
    if df is None:
        df = temp_df
    else:
        df = df.union(temp_df)
'''
print((df.count(), len(df.columns)))


def f(x):
    print(x)


# jieba_instance = jieba.Tokenizer()


def cut(row):
    return list(jieba.cut(row.value)), row.label


def _tokenize(content):
    return [i for i in jieba.cut(content)]


'''
tokenize = F.udf(_tokenize, ArrayType(StringType()))
df_cut = df.select([tokenize('value').alias('content'), 'label'])
df_cut.show()
'''
# 分词

df_cut_rdd = df.rdd.map(cut)
'''
df_cut_rdd = df_cut_rdd.map(lambda x: x)
df_cut_rdd.count()
for element in df_cut_rdd.collect():
    print(element)
'''
print(df_cut_rdd.count())
df_cut = df_cut_rdd.toDF()
df_cut = df_cut.selectExpr("_1 as content", "_2 as label")

from pyspark.ml.feature import StopWordsRemover

sw_removal = StopWordsRemover(inputCol='content', outputCol='cleaned', stopWords=stopwords_list)
new_df = sw_removal.transform(df_cut)
# new_df.select(['content', 'label', 'cleaned']).show(4, False)

# TF-IDF
from pyspark.ml.feature import HashingTF, IDF

hashing_vector = HashingTF(inputCol='cleaned', outputCol='tf_vector')
hashing_df = hashing_vector.transform(new_df)
# hashing_df.select(['label', 'cleaned', 'tf_vector']).show()
# IDF
tf_idf_vector = IDF(inputCol='tf_vector', outputCol='tf_idf_vector')
tf_idf_df = tf_idf_vector.fit(hashing_df).transform(hashing_df)
# tf_idf_df.select(['label', 'tf_idf_vector']).show(4, False)

from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import OneVsRest

# tf-idf vector
traincontent, testcontent = tf_idf_df.randomSplit([0.95, 0.05])
lsvc = LinearSVC(featuresCol='tf_idf_vector', labelCol='label')
ovr = OneVsRest(classifier=lsvc, featuresCol='tf_idf_vector', labelCol='label')
ovrModel = ovr.fit(traincontent)
resultcontent = ovrModel.transform(testcontent)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

accuracycontent = MulticlassClassificationEvaluator(labelCol='label', metricName='accuracy').evaluate(resultcontent)
precisioncontent = MulticlassClassificationEvaluator(labelCol='label', metricName='weightedPrecision').evaluate(
    resultcontent)
