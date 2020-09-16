import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{HashingTF, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object TweetAnalysis {
  def main(args: Array[String]): Unit = {
    if(args.length!=2){
      println("Usage: TweetParse InputDir OutputDir")
    }

    // Environment:
    val sc  = new SparkContext(new SparkConf().setAppName("Tweet Analysis"))
    val spark = SparkSession.builder().appName("TweetAnalysis").getOrCreate()
    val sqlContext = spark.sqlContext
    import spark.implicits._

    // Loading data:
    val input = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load(args(0))
    val cleanData = input.filter($"text".isNotNull)

    val tokenize = new Tokenizer().setInputCol("text").setOutputCol("words")
    val stopwordRemover = new StopWordsRemover().setInputCol(tokenize.getOutputCol).setOutputCol("stopWords")
    val hashingTF = new HashingTF().setInputCol(stopwordRemover.getOutputCol).setOutputCol("features")
    val Labelencoder = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label")

    // Train model:
    val lrmodel = new LogisticRegression().setMaxIter(10).setFeaturesCol(hashingTF.getOutputCol).setLabelCol(Labelencoder.getOutputCol)

    val pipeline = new Pipeline().setStages(Array(tokenize, stopwordRemover, hashingTF, Labelencoder, lrmodel))
    val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(10, 100, 1000)).addGrid(lrmodel.regParam, Array(0.1, 0.01)).build()

    // Evaluation
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val crossvalidate = new CrossValidator().setEstimator(pipeline).setEvaluator(evaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

    val Array(training, test) = cleanData.randomSplit(Array(0.75,0.25))

    val trainModel = crossvalidate.fit(training)

    val testModel = trainModel.transform(test)

    val PredictionAndLabelsLR = testModel.select("prediction", "label").rdd.map{case Row(prediction: Double, label: Double) => (prediction, label)}

    val metrics = new MulticlassMetrics(PredictionAndLabelsLR)

    var ResultTemp = "Homework 2.2: Tweet Analysis"

    ResultTemp += "\nPrecision :    " + metrics.weightedPrecision
    ResultTemp += "\nRecall :   " + metrics.weightedRecall
    ResultTemp += "\nAccuracy :   " + metrics.accuracy
    ResultTemp += "\nF1Score :   " + metrics.weightedFMeasure

    val Result = List(ResultTemp)
    val outputRDD = sc.parallelize(Result)
    outputRDD.saveAsTextFile(args(1))
  }
}
