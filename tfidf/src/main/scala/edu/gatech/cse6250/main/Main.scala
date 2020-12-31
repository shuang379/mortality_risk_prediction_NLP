package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.features.FeatureConstruction
import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper }
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import edu.gatech.cse6250.model.{ Diagnosis, LabEvent, Medication, Dead, Alive, Admission, Mortality, TfIdf }

import org.apache.spark.mllib.classification.{ LogisticRegressionModel, LogisticRegressionWithLBFGS }
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.feature.PCA

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils

import org.apache.spark.mllib.feature.ElementwiseProduct
import org.apache.spark.mllib.linalg.Vectors

import scala.io.Source

import java.sql.Date
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    /** initialize loading of data */
    val (admissionsRDD, diagnosesRDD, medicationRDD, labEventsRDD, mortalityRDD, tfIdfRDD) = loadRddRawData(spark)

    // filter data to include events in observation window only
    // for deceased patients: index date is 30 days prior to the day of death
    // for alive patients: index date is 30 days prior to the last day of event
    // observation period is 2000 days
    val (diagFilteredRDD, medFilteredRDD, labFilteredRDD, tfIdfFilteredRDD) = filterEvents(admissionsRDD, diagnosesRDD, medicationRDD, labEventsRDD, tfIdfRDD)

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagFilteredRDD),
      FeatureConstruction.constructLabFeatureTuple(labFilteredRDD),
      FeatureConstruction.constructMedicationFeatureTuple(medFilteredRDD),
      FeatureConstruction.constructWordFeatureTuple(tfIdfFilteredRDD)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)
    val rawFeatures_RDD = rawFeatures.map(x => (x._1, x._2))

    // construct the label (mortality)
    val mortality = mortalityRDD.map(x => (x.subject_id, x.mortality.toDouble)).groupByKey().map(x => (x._1, x._2.sum)).collect()
    val mortality_RDD = sc.parallelize(mortality.map(x => (x._1, if (x._2 > 0) { 1.0 } else { 0.0 })))

    // random split of datasets
    val joined_withID = (rawFeatures_RDD join mortality_RDD).collect()
    val joined = sc.parallelize(joined_withID.map(x => (x._2._2, x._2._1)))

    val data = joined.map {
      case (label, vector) => LabeledPoint(label, vector)
    }

    val splits = data.randomSplit(Array(0.6, 0.4), seed = 6250L)
    val training_set_before_pca = splits(0).cache()
    val testing_set_before_pca = splits(1)

    // // read in the training and testing set ID
    // val training_ID = spark.read.format("csv").option("header", "true").load("data/trainIDs.csv")
    // val testing_ID = spark.read.format("csv").option("header", "true").load("data/testIDs.csv")
    // val training_ID_final = training_ID.dropDuplicates().toDF().rdd.map(r => r(0)).collect()
    // val testing_ID_final = testing_ID.dropDuplicates().toDF().rdd.map(r => r(0)).collect()

    // // join the label and features
    // val joined_withID = (rawFeatures_RDD join mortality_RDD).collect()
    // val training_joined = sc.parallelize(joined_withID.filter(x => training_ID_final.contains(x._1)).map(x => (x._2._2, x._2._1)))
    // val testing_joined = sc.parallelize(joined_withID.filter(x => testing_ID_final.contains(x._1)).map(x => (x._2._2, x._2._1)))
    // val training_set_before_pca = training_joined.map { case (label, vector) => LabeledPoint(label, vector) }
    // val testing_set_before_pca = testing_joined.map { case (label, vector) => LabeledPoint(label, vector) }

    // PCA is coming
    // val pca = new PCA(50).fit(training_set_before_pca.map(_.features))
    // val training_set_after_pca = training_set_before_pca.map(p => p.copy(features = pca.transform(p.features)))
    // val testing_set_after_pca = testing_set_before_pca.map(p => p.copy(features = pca.transform(p.features)))

    val training_set = training_set_before_pca.cache()
    val testing_set = testing_set_before_pca

    // logistics regression
    val model_logistic = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training_set)
    val predictionAndLabels_logistic = testing_set.map {
      case LabeledPoint(label, features) =>
        val prediction = model_logistic.predict(features)
        (prediction, label)
    }
    model_logistic.save(sc, "output/tfidf_logistic")

    val metrics_logistic = new MulticlassMetrics(predictionAndLabels_logistic)
    val metrics_logistic_ = new BinaryClassificationMetrics(predictionAndLabels_logistic)

    val accuracy_logistic = metrics_logistic.accuracy
    println(s"Accuracy (TF-IDF, Logistic Regression) = $accuracy_logistic")
    val confusion_logistic = metrics_logistic.confusionMatrix
    println("Confusion matrix (TF-IDF, Logistic Regression):" + confusion_logistic)
    val AUROC_logistic = metrics_logistic_.areaUnderROC
    println("Area under ROC (TF-IDF, Logistic Regression) = " + AUROC_logistic)
    val tfidf_logistic = Array(accuracy_logistic, AUROC_logistic, confusion_logistic(0, 0), confusion_logistic(0, 1), confusion_logistic(1, 0), confusion_logistic(1, 1))
    sc.parallelize(tfidf_logistic).saveAsTextFile("output/tfidf_logistic/result")

    // decision tree
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 10
    val maxBins = 32
    val model_decisiontree = DecisionTree.trainClassifier(training_set, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val predictionAndLabels_decisiontree = testing_set.map {
      case LabeledPoint(label, features) =>
        val prediction = model_decisiontree.predict(features)
        (prediction, label)
    }
    model_decisiontree.save(sc, "output/tfidf_decisiontree")

    val metrics_decisiontree = new MulticlassMetrics(predictionAndLabels_decisiontree)
    val metrics_decisiontree_ = new BinaryClassificationMetrics(predictionAndLabels_decisiontree)

    val accuracy_decisiontree = metrics_decisiontree.accuracy
    println(s"Accuracy (TF-IDF, Decision Tree) = $accuracy_decisiontree")
    val confusion_decisiontree = metrics_decisiontree.confusionMatrix
    println("Confusion matrix (TF-IDF, Decision Tree):" + confusion_decisiontree)
    val AUROC_decisiontree = metrics_decisiontree_.areaUnderROC
    println("Area under ROC (TF-IDF, Decision Tree) = " + AUROC_decisiontree)
    val tfidf_decisiontree = Array(accuracy_decisiontree, AUROC_decisiontree, confusion_decisiontree(0, 0), confusion_decisiontree(0, 1), confusion_decisiontree(1, 0), confusion_decisiontree(1, 1))
    sc.parallelize(tfidf_decisiontree).saveAsTextFile("output/tfidf_decisiontree/result")

    // gradient boosting tree
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    boostingStrategy.numIterations = 50
    boostingStrategy.treeStrategy.numClasses = 2
    boostingStrategy.treeStrategy.maxDepth = 5
    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()

    val model_boostingtree = GradientBoostedTrees.train(training_set, boostingStrategy)
    val predictionAndLabels_boostingtree = testing_set.map {
      case LabeledPoint(label, features) =>
        val prediction = model_boostingtree.predict(features)
        (prediction, label)
    }
    model_boostingtree.save(sc, "output/tfidf_boostingtree")

    val metrics_boostingtree = new MulticlassMetrics(predictionAndLabels_boostingtree)
    val metrics_boostingtree_ = new BinaryClassificationMetrics(predictionAndLabels_boostingtree)

    val accuracy_boostingtree = metrics_boostingtree.accuracy
    println(s"Accuracy (TF-IDF, Boosting Tree) = $accuracy_boostingtree")
    val confusion_boostingtree = metrics_boostingtree.confusionMatrix
    println("Confusion matrix (TF-IDF, Boosting Tree):" + confusion_boostingtree)
    val AUROC_boostingtree = metrics_boostingtree_.areaUnderROC
    println("Area under ROC (TF-IDF, Boosting Tree) = " + AUROC_boostingtree)
    val tfidf_boostingtree = Array(accuracy_boostingtree, AUROC_boostingtree, confusion_boostingtree(0, 0), confusion_boostingtree(0, 1), confusion_boostingtree(1, 0), confusion_boostingtree(1, 1))
    sc.parallelize(tfidf_boostingtree).saveAsTextFile("output/tfidf_boostingtree/result")

  }

  def sqlDateParser(input: String, pattern: String = "yyyy-MM-dd HH:mm:ss"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def localToSqlDateParser(input: String, pattern: String = "yyyy-MM-dd"): java.sql.Date = {
    val dateFormat = new SimpleDateFormat(pattern)
    new java.sql.Date(dateFormat.parse(input).getTime)
  }

  def filterEvents(admissions: RDD[Admission], diagnoses: RDD[Diagnosis], medication: RDD[Medication], labEvents: RDD[LabEvent], tfIdf: RDD[TfIdf]): (RDD[Diagnosis], RDD[Medication], RDD[LabEvent], RDD[TfIdf]) = {

    val spark = SparkHelper.spark
    val sc = spark.sparkContext

    val deadObservDatesRDD = admissions.filter(e => e.mortality == "1").map {
      case Admission(subject_id, deathtime, mortality) =>
        val endDate = LocalDate.parse(deathtime, DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")).minusDays(30)
        val startDate = endDate.minusDays(2000)
        (subject_id, startDate, endDate)
    }

    val patientsAliveSet = admissions.filter(e => e.mortality == "0").map(e => e.subject_id).collect.toSet
    val eventsDateUnionRDD = sc.union(
      diagnoses.map(e => (e.subject_id, e.admittime)),
      medication.map(e => (e.subject_id, e.admittime)),
      labEvents.map(e => (e.subject_id, e.charttime))
    ).distinct
    val aliveObservDatesRDD = eventsDateUnionRDD.filter { case (subject_id, charttime) => patientsAliveSet(subject_id) }.reduceByKey((date1, date2) => if (date1.before(date2)) date2 else date1).map {
      case (subject_id, endDate) =>
        val endDate_trans = LocalDate.parse(endDate.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd"))
        val startDate = LocalDate.parse(endDate_trans.minusDays(2000).toString, DateTimeFormatter.ofPattern("yyyy-MM-dd"))
        (subject_id, startDate, endDate_trans)
    }

    val observDatesRDD = deadObservDatesRDD.union(aliveObservDatesRDD)

    // filter events
    val observDatesKeyValueRDD = observDatesRDD.map { case (subject_id, startDate, endDate) => (subject_id, (startDate, endDate)) }
    val diagnosesKeyValueRDD = diagnoses.map { case Diagnosis(subject_id, icd9_code, admittime) => (subject_id, (icd9_code, LocalDate.parse(admittime.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd")))) }
    val medicationKeyValueRDD = medication.map { case Medication(subject_id, drg_code, admittime) => (subject_id, (drg_code, LocalDate.parse(admittime.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd")))) }
    val labKeyValueRDD = labEvents.map { case LabEvent(subject_id, itemid, valuenum, charttime) => (subject_id, (itemid, valuenum, LocalDate.parse(charttime.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd")))) }
    val wordKeyValueRDD = tfIdf.map { case TfIdf(subject_id, word, tfidf, charttime) => (subject_id, (word, tfidf, LocalDate.parse(charttime.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd")))) }

    val diagFilteredRDD = diagnosesKeyValueRDD.join(observDatesKeyValueRDD).filter { case (subject_id, ((icd9_code, admittime), (startDate, endDate))) => startDate.isBefore(admittime) && admittime.isBefore(endDate) }.map { case (subject_id, ((icd9_code, admittime), (startDate, endDate))) => Diagnosis(subject_id, icd9_code, localToSqlDateParser(admittime.toString)) }
    val medFilteredRDD = medicationKeyValueRDD.join(observDatesKeyValueRDD).filter { case (subject_id, ((drg_code, admittime), (startDate, endDate))) => startDate.isBefore(admittime) && admittime.isBefore(endDate) }.map { case (subject_id, ((drg_code, admittime), (startDate, endDate))) => Medication(subject_id, drg_code, localToSqlDateParser(admittime.toString)) }
    val labFilteredRDD = labKeyValueRDD.join(observDatesKeyValueRDD).filter { case (subject_id, ((itemid, valuenum, charttime), (startDate, endDate))) => startDate.isBefore(charttime) && charttime.isBefore(endDate) }.map { case (subject_id, ((itemid, valuenum, charttime), (startDate, endDate))) => LabEvent(subject_id, itemid, valuenum, localToSqlDateParser(charttime.toString)) }
    val tfIdfFilteredRDD = wordKeyValueRDD.join(observDatesKeyValueRDD).filter { case (subject_id, ((word, tfidf, charttime), (startDate, endDate))) => startDate.isBefore(charttime) && charttime.isBefore(endDate) }.map { case (subject_id, ((word, tfidf, charttime), (startDate, endDate))) => TfIdf(subject_id, word, tfidf, localToSqlDateParser(charttime.toString)) }

    (diagFilteredRDD, medFilteredRDD, labFilteredRDD, tfIdfFilteredRDD)
  }

  def loadRddRawData(spark: SparkSession): (RDD[Admission], RDD[Diagnosis], RDD[Medication], RDD[LabEvent], RDD[Mortality], RDD[TfIdf]) = {

    // mortality RDD
    val admDFCsv = spark.read.format("com.databricks.spark.csv").option("header", "true").option("mode", "DROPMALFORMED").option("delimiter", ",").load("data/admissions.csv")
    val admissions: RDD[Admission] = admDFCsv.rdd.map { row =>
      new Admission(row.getString(0), row.getString(1), row.getString(2))
    }
    val mortality = admissions.map { case Admission(subject_id, deathtime, mortality) => new Mortality(subject_id, mortality) }

    val admDFCsv_dead = admDFCsv.filter(admDFCsv("hospital_expire_flag") === 1)
    val admDFCsv_alive = admDFCsv.filter(admDFCsv("hospital_expire_flag") === 0).select("subject_id", "hospital_expire_flag")
    val patients_dead: RDD[Dead] = admDFCsv_dead.rdd.map { row =>
      new Dead(row.getString(0), sqlDateParser(row.getString(1)), row.getString(2))
    }
    val patient_alive: RDD[Alive] = admDFCsv_alive.rdd.map { row =>
      new Alive(row.getString(0), row.getString(1))
    }

    // diagnoses RDD
    val diagDFCsv = spark.read.format("com.databricks.spark.csv").option("header", "true").option("mode", "DROPMALFORMED").option("delimiter", ",").load("data/diagnoses.csv")
    val diagnoses: RDD[Diagnosis] = diagDFCsv.rdd.map { row =>
      new Diagnosis(row.getString(0), "DIAG" + row.getString(2), sqlDateParser(row.getString(3)))
    }

    // medication RDD
    val medDFCsv = spark.read.format("com.databricks.spark.csv").option("header", "true").option("mode", "DROPMALFORMED").option("delimiter", ",").load("data/drgcodes.csv")
    val medication: RDD[Medication] = medDFCsv.rdd.map { row =>
      new Medication(row.getString(0), "MED" + row.getString(2), sqlDateParser(row.getString(3)))
    }

    // labEvents RDD
    val labDFCsv = spark.read.format("com.databricks.spark.csv").option("header", "true").option("mode", "DROPMALFORMED").option("delimiter", ",").load("data/labevents.csv").na.drop()
    val labEvents: RDD[LabEvent] = labDFCsv.rdd.map { row =>
      new LabEvent(row.getString(0), "LAB" + row.getString(2), row.getString(3).toDouble, sqlDateParser(row.getString(4)))
    }

    // wordCount_tfidf RDD
    val tfIdfDFCsv = spark.read.format("com.databricks.spark.csv").option("header", "true").option("mode", "DROPMALFORMED").option("delimiter", ",").load("data/wordCount_tfidf.csv")
    val tfIdf: RDD[TfIdf] = tfIdfDFCsv.rdd.map { row =>
      new TfIdf(row.getString(0), row.getString(1), row.getString(2).toDouble, sqlDateParser(row.getString(3)))
    }

    // val mortality: RDD[Diagnosis] = spark.sparkContext.emptyRDD

    (admissions, diagnoses, medication, labEvents, mortality, tfIdf)
  }

}
