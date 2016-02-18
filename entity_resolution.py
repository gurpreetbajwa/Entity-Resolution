import re
import operator
from pyspark import SQLContext,SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import operator
from pyspark.sql.types import StringType, ArrayType

from pyspark.sql.functions import concat, col, lit, concat_ws


conf = SparkConf().setAppName('entity_res')
sc = SparkContext(conf=conf)
sqlCt = SQLContext(sc)

class EntityResolution:
    def __init__(self, dataFile1, dataFile2, stopWordsFile):
        self.f = open(stopWordsFile, "r")
        self.stopWords = set(self.f.read().split("\n"))
        self.stopWordsBC = sc.broadcast(self.stopWords).value
        self.df1 = sqlCt.read.parquet(dataFile1).cache()
        self.df2 = sqlCt.read.parquet(dataFile2).cache()

    def preprocessDF(self, df, cols):

        new_df=df.withColumn("joinkey",concat_ws('-', *cols))


        def transform(raw):
            words=[]
            s = re.split(r'\W+', raw)

            final_list=[]
            for i in s:
             fin_s=i.lower()
             if(len(fin_s)>0):
                final_list.append(fin_s)


            for i in final_list:
                if i not in stop_word:
                    words.append(i)

            return words
        stop_word=self.stopWordsBC
        slen=udf(transform, ArrayType(StringType()))
        df1=new_df.withColumn("joinkey", slen(new_df.joinkey))

        return df1


    def filtering(self, df1, df2):
        def f(x): return x
        new_one=df1.select(col("id"), col("joinkey"))
        new_two=df2.select(col("id"), col("joinkey"))

        flat_one=new_one.rdd.flatMapValues(f)
        data_one=sqlCt.createDataFrame(flat_one,('id','joinkey')).alias("data_one")

        flat_two=new_two.rdd.flatMapValues(f)
        data_two=sqlCt.createDataFrame(flat_two,('id','joinkey')).alias("data_two")

        cond = [col('data_one.joinkey') == col('data_two.joinkey')]
        entity_join=data_one.join(data_two,cond,'inner').select(col("data_one.id").alias('id1'),(col("data_two.id").alias('id2'))).distinct().alias("entity")
        entity_join.show()

        cond1 = [col('entity.id1') == df1['id']]
        df_join=entity_join.join(df1, cond1, 'inner').select(df1['id'],df1['joinkey'],(col("id2"))).alias("table1")

        cond1 = [col('table1.id2') == df2['id']]
        final_join=df_join.join(df2, cond1, 'inner').select((col("table1.id").alias('id1')),(col("table1.joinkey").alias('joinkey1')),(df2['id'].alias('id2')),(df2['joinkey'].alias('joinkey2')))
        return final_join

    def verification(self, candDF, threshold):
       def jaccardCal(key1, key2):
            list1=set(key1)
            list2=set(key2)
            union_row=list1.union(list2)
            inter_row=list1.intersection(list2)
            score=float(len(inter_row))/len(union_row)
            return score

       slen=udf(jaccardCal, FloatType())
       df1=candDF.withColumn("jaccard", slen(candDF.joinkey1,candDF.joinkey2))
       filter_df=df1.filter(df1['jaccard']>threshold)
       return filter_df

    def evaluate(self, result, groundTruth):
        list1=set(result)
        list2=set(groundTruth)
        inter_row=list1.intersection(list2)
        precision=float(len(inter_row))/len(result)
        recall=float(len(inter_row))/len(groundTruth)
        fmeasure=(2*precision*recall)/(precision+recall)
        return precision,recall,fmeasure

    def jaccardJoin(self, cols1, cols2, threshold):
        newDF1 = self.preprocessDF(self.df1, cols1)
        newDF2 = self.preprocessDF(self.df2, cols2)
        print "Before filtering: %d pairs in total" %(self.df1.count()*self.df2.count())
        candDF = self.filtering(newDF1, newDF2)
        print "After Filtering: %d pairs left" %(candDF.count())

        resultDF = self.verification(candDF, threshold)
        print "After Verification: %d similar pairs" %(resultDF.count())

        return resultDF


    def __del__(self):
        self.f.close()


if __name__ == "__main__":
    er = EntityResolution('Amazon_sample','Google_sample', 'stopwords.txt')
    amazonCols = ["title", "manufacturer"]
    googleCols = ["name", "manufacturer"]
    resultDF = er.jaccardJoin(amazonCols, googleCols, 0.5)

    result = resultDF.map(lambda row: (row.id1, row.id2)).collect()
    groundTruth = sqlCt.read.parquet("sample/Amazon_Google_perfectMapping_sample") \
                          .map(lambda row: (row.idAmazon, row.idGoogle)).collect()
    print "(precision, recall, fmeasure) = ", er.evaluate(result, groundTruth)
