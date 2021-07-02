import jieba
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import ArrayType, StringType

spark = SparkSession.builder.appName("word2vec_cluster").getOrCreate()

df = spark.createDataFrame([
    (0, '不用加水光？'),
    (0, '不用加水光？'),
    (0, '不用加水光？'),
    (1, '加水光多少钱啊？'),
    (1, '加水光多少钱啊？'),
    (1, '加水光多少钱啊？'),
    (2, '单纯水光多少'),
    (2, '单纯水光多少'),
    (2, '单纯水光多少'),
], ["id", "content"])
df.show()


def _tokenize(content):
    """此udf在每个partiton中加载一次词典，
    "不用加水光", 其中的"水光"会被保留，结果为[不用, 加, 水光, ？]

    :param content: {str} 要分词的内容
    :return: list[word, word, ...]
    """
    '''
    if not jieba.dt.initialized:
        # 词典中有"水光"这个词
        jieba.load_userdict(
            '/Users/chant/sy/dwh/utils/static/soyoung.item.jieba.dic')
    '''
    return [i for i in jieba.cut(content)]


tokenize = F.udf(_tokenize, ArrayType(StringType()))
df.select(tokenize('content').alias('words')).show()
print('df 的partition 数为：', df.rdd.getNumPartitions())

