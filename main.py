from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF, Tokenizer
from pyspark.sql.functions import split, col, expr, concat_ws
from sparknlp.base import DocumentAssembler, Pipeline, EmbeddingsFinisher
from sparknlp.annotator import BertSentenceEmbeddings, SentenceDetector
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as psf
import sparknlp


def stopWords(df):
    split_regex = "((,)?\s|[-])"
    df1 = df.withColumn("description", split(col("description"), split_regex))
    sw = StopWordsRemover(inputCol="description", outputCol="DescriptionWithoutStopWords")
    df2 = sw.transform(df1)
    df2 = df2.withColumn("DescriptionWithoutStopWords",
                         expr("transform(DescriptionWithoutStopWords,x-> replace(x,':',''))")).drop("description")

    df2 = df2.withColumn("DescriptionWithoutStopWords", concat_ws(",", col("DescriptionWithoutStopWords")))

    print("Dataframe after remove Stop Words:")
    df2.select("DescriptionWithoutStopWords", "label").show()

    return df2


def wordEmbed(df):
    tokenizer = Tokenizer(inputCol="DescriptionWithoutStopWords", outputCol="words")
    wordsData = tokenizer.transform(df)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    rescaledData = rescaledData.withColumnRenamed("fraudulent", "label")
    rescaledData = rescaledData.withColumn("label", rescaledData["label"].cast(IntegerType()))

    print("Dataframe after Word Embeddings:")
    rescaledData.select("features", "label").drop("DescriptionWithoutStopWords", "words", "rawFeatures").show()

    layers = [20, 5, 4, 2]
    mlpc(rescaledData, "features", "label", layers)

    return rescaledData


def bert(df):
    document_assembler = DocumentAssembler() \
        .setInputCol("description") \
        .setOutputCol("document")

    sentence = SentenceDetector() \
        .setInputCols(["document"]) \
        .setOutputCol("sentence")

    embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128") \
        .setInputCols(["sentence"]) \
        .setOutputCol("sentence_bert_embeddings")

    embeddingsFinisher = EmbeddingsFinisher() \
        .setInputCols(["sentence_bert_embeddings"]) \
        .setOutputCols("finished_embeddings") \
        .setOutputAsVector(True)

    pipeline = Pipeline().setStages([
        document_assembler,
        sentence,
        embeddings,
        embeddingsFinisher,
    ])

    result = pipeline.fit(df).transform(df)
    print("Το dataframe που προκύπτει από την εκτέλεση του Bert Embeddings:")
    result.show()

    df1 = result.withColumn("label", result["label"].cast(IntegerType()))
    df1 = df1.withColumn("clear_finished_embeddings", result["finished_embeddings"].getItem(0))
    df1 = df1.drop("description", "document", "sentence", "sentence_bert_embeddings", "embeddings",
                   "finished_embeddings")
    df1.show()

    layers = [128, 5, 4, 2]
    mlpc(df1, "clear_finished_embeddings", "label", layers)

    return df1


def mlpc(df, featuresCol, labelCol, layers):
    trainer = MultilayerPerceptronClassifier(featuresCol=featuresCol, labelCol=labelCol,
                                             maxIter=500, stepSize=0.00001, layers=layers, blockSize=128, seed=1234)

    training, valid = df.select(featuresCol, labelCol).randomSplit([0.8, 0.2])

    print("Training set:")
    training.groupBy("label").agg(psf.count("label")).show()
    print("Test set:")
    valid.groupBy("label").agg(psf.count("label")).show()

    model = trainer.fit(training)
    result = model.transform(valid)

    predictionAndLabels = result.select("prediction", "label")
    accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
    precision = MulticlassClassificationEvaluator(metricName="precisionByLabel")
    recall = MulticlassClassificationEvaluator(metricName="recallByLabel")
    f1 = MulticlassClassificationEvaluator(metricName="f1")

    print("Accuracy = " + str(accuracy.evaluate(predictionAndLabels)))
    print("Precision = " + str(precision.evaluate(predictionAndLabels)))
    print("Recall = " + str(recall.evaluate(predictionAndLabels)))
    print("F1 = " + str(f1.evaluate(predictionAndLabels)))

    return predictionAndLabels


def main():
    spark = sparknlp.start()

    csv_file = spark.read.option("header", "true") \
        .option("sep", ",") \
        .option("multiLine", "true") \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .option("ignoreTrailingWhiteSpace", True) \
        .csv("job_postings.csv")

    df = csv_file.select("description", "fraudulent")
    df.show()
    print("Number of rows per class:")
    df.groupBy("fraudulent").agg(psf.count("fraudulent").alias("Number of rows")).show()

    df = df.withColumnRenamed("fraudulent", "label")
    df1 = stopWords(df)
    wordEmbed(df1)
    bert(df)


if __name__ == '__main__':
    main()
