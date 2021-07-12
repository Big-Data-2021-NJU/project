# coding: utf-8
import datetime

starttime = datetime.datetime.now()
import os
import pandas as pd
import jieba
from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
import argparse
from pyspark.mllib.feature import Normalizer
from pyspark.ml.feature import Normalizer

dic = {}
parser = argparse.ArgumentParser()
parser.add_argument('-local', type=bool, help='output file name', required=False)
args = parser.parse_args()


def f(x):
    print(x)


# jieba_instance = jieba.Tokenizer()


def cut(row):
    return list(jieba.cut(row.value)), row.label  # , row.test


def _filter_words(row):
    global dic
    return [word for word in row if word in dic]


def _tokenize(content):
    return [i for i in jieba.cut(content)]


def _normalize(row):
    global nor
    return nor.transform(row)


'''
tokenize = F.udf(_tokenize, ArrayType(StringType()))
df_cut = df.select([tokenize('value').alias('content'), 'label'])
df_cut.show()
'''


def chinese(word):
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def load(directory, outdir, outdir2, train):
    if args.local:
        status = os.listdir(directory)
    else:
        status = fs.listStatus(Path(directory))
    dataframe = None
    i = 0  # label
    count = 0
    for fileStatus in status:  # fileStatus is dir on local
        # dir_status = fs.listStatus(Path(str(dir_path)[len("hdfs://master001:9000"):]))
        if args.local:
            temp_df = sc.wholeTextFiles(directory + '/' + fileStatus + '/*').toDF()
        else:
            dir_path = fileStatus.getPath()
            temp_df = sc.wholeTextFiles(str(dir_path) + '/*').toDF()
        '''
        for text_status in dir_status:
            file_path = text_status.getPath()
            count += cmds
            print(count)
            if temp_df is None:
                temp_df = spark.read.text(str(file_path), wholetext=True)
                # assert(0)
            else:
                temp_df1 = spark.read.text(str(file_path), wholetext=True)
                temp_df = temp_df.union(temp_df1)
        '''
        temp_df = temp_df.drop('_1')
        temp_df = temp_df.selectExpr("_2 as value")
        temp_df = temp_df.withColumn("label", F.lit(i))
        i += 1
        if dataframe is None:
            dataframe = temp_df
        else:
            dataframe = dataframe.union(temp_df)
        print(i, '/', len(status), 'done!')

    df_cut_rdd = dataframe.rdd.map(cut)

    '''
    df_cut_rdd = df_cut_rdd.map(lambda x: x)
    df_cut_rdd.count()
    for element in df_cut_rdd.collect():
        print(element)
    '''

    # print(df_cut_rdd.count())
    df_cut = df_cut_rdd.toDF()

    df_cut.write.save(outdir)
    df_cut = spark.read.load(outdir)

    # df_cut = df_cut.selectExpr("_1 as content", "_2 as label", "_3 as test")
    df_cut = df_cut.selectExpr("_1 as content", "_2 as label")

    from pyspark.ml.feature import StopWordsRemover

    sw_removal = StopWordsRemover(inputCol='content', outputCol='cleaned', stopWords=stopwords_list)
    new_df = sw_removal.transform(df_cut)
    new_df = new_df.selectExpr("cleaned as cleaned", "label as label")
    return new_df

    # new_df.select(['content', 'label', 'cleaned']).show(4, False)


def tfidf(train, test):
    # TF-IDF
    from pyspark.ml.feature import HashingTF, IDF

    hashing_vector = HashingTF(inputCol='cleaned', outputCol='tf_vector')
    hashing_df_train = hashing_vector.transform(train)
    hashing_df_test = hashing_vector.transform(test)
    normalizer = Normalizer(inputCol="tf_vector", outputCol="normed_tf_vector", p=1.0)
    hashing_df_train = normalizer.transform(hashing_df_train)
    hashing_df_test = normalizer.transform(hashing_df_test)

    '''
    normalize= F.udf(_normalize, ArrayType(StringType()))
    hashing_df_train_normalized = hashing_df_train.select(
        [normalize('tf_vector').alias('normalized_tf_vector'), 'label', 'tf_vector'])
    '''

    # IDF
    tf_idf_vector = IDF(inputCol='tf_vector', outputCol='tf_idf_vector')
    idf = tf_idf_vector.fit(hashing_df_train)
    tf_idf_df_train = idf.transform(hashing_df_train)
    tf_idf_df_test = idf.transform(hashing_df_test)
    # tf_idf_df.select(['label', 'tf_idf_vector']).show(4, False)

    # tf-idf vector
    tf_idf_df_train, tf_idf_df_test = tf_idf_df_train.selectExpr("tf_idf_vector as tf_idf_vector",
                                                                 "label as label"), tf_idf_df_test.selectExpr(
        "tf_idf_vector as tf_idf_vector",
        "label as label")
    tf_idf_df_train.write.save("train")
    tf_idf_df_test.write.save("test")

    tf_idf_vector = IDF(inputCol='normed_tf_vector', outputCol='normed_tf_idf_vector')
    idf = tf_idf_vector.fit(hashing_df_train)
    tf_idf_df_train = idf.transform(hashing_df_train)
    tf_idf_df_test = idf.transform(hashing_df_test)
    # tf-idf vector
    tf_idf_df_train, tf_idf_df_test = tf_idf_df_train.selectExpr("normed_tf_idf_vector as normed_tf_idf_vector",
                                                                 "label as label"), tf_idf_df_test.selectExpr(
        "normed_tf_idf_vector as normed_tf_idf_vector",
        "label as label")
    tf_idf_df_train.write.save("train_normed")
    tf_idf_df_test.write.save("test_normed")


from pyspark.conf import SparkConf

conf = SparkConf()
if args.local:
    conf.setMaster("local").setAppName("feature")
else:
    conf.setMaster("spark://192.168.cmds.cmds:7077").setAppName("feature")

# from hdfs import InsecureClient
# client = InsecureClient('hdfs://master001:9000')
# fnames = client.list('/data/2021spring/3-news_classification')
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("WARN")
if args.local:
    spark = SparkSession.builder.master("local").appName("feature").getOrCreate()
else:
    spark = SparkSession.builder.master("spark://192.168.cmds.cmds:7077").appName("feature").getOrCreate()
# df = sc.wholeTextFiles("3-news_classification_sample/sample/体育/*").toDF()

if args.local:
    stopwords = spark.read.text("3-news_classification_sample/cn_stopwords.txt")
else:
    stopwords = spark.read.text("hdfs://master001:9000/data/2021spring/3-news_classification/cn_stopwords.txt")
stopwords_list = list(stopwords.select('value').toPandas().value)
# stopwords_list = stopwords.read().split('\n')


URI = sc._gateway.jvm.java.net.URI
Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
Configuration = sc._gateway.jvm.org.apache.hadoop.conf.Configuration
if args.local:
    train_dir = '../3-news_classification/train'
    test_dir = '../3-news_classification/test'
    # train_dir = '3-news_classification_sample/sample'
    # test_dir = '3-news_classification_sample/sample_test'
else:
    fs = FileSystem.get(URI("hdfs://master001:9000"), Configuration())
    train_dir = '/data/2021spring/3-news_classification/train'
    test_dir = '/data/2021spring/3-news_classification/test'

train = load(train_dir, "train_dataset", "train_filtered", True)
test = load(test_dir, "test_dataset", "test_filtered", False)
tfidf(train, test)

f = open("feature_original.txt", 'w')
endtime = datetime.datetime.now()
time_elapsed = (endtime - starttime).seconds
print("Time elapsed: " + str(time_elapsed) + 's')
f.write("Time elapsed: " + str(time_elapsed) + 's')
f.close()
# wrong. should load train and test separately.
'''
df = load_data(train_dir)
df = df.withColumn("test", F.lit(0))
df_test = load_data(test_dir)
df_test = df_test.withColumn("test", F.lit(cmds))
df = df.union(df_test)
'''

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

'''
tokenize = F.udf(_tokenize, ArrayType(StringType()))
df_cut = df.select([tokenize('value').alias('content'), 'label'])
df_cut.show()
'''
