package edu.gatech.cse6250.features

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import edu.gatech.cse6250.model.{ Diagnosis, LabEvent, Medication }

import java.sql.Date

object FeatureConstruction {

  type FeatureTuple = ((String, String), Double)

  // Aggregate feature tuples from diagnostic with COUNT aggregation,
  def constructDiagnosticFeatureTuple(diagnoses: RDD[Diagnosis]): RDD[FeatureTuple] = {
    diagnoses.map { case Diagnosis(subject_id, icd9_code, admittime) => ((subject_id, icd9_code), 1.0) }.reduceByKey(_ + _)
  }

  // Aggregate feature tuples from medication with COUNT aggregation,
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    medication.map { case Medication(subject_id, drg_code, admittime) => ((subject_id, drg_code), 1.0) }.reduceByKey(_ + _)
  }

  // Aggregate feature tuples from lab result, using AVERAGE aggregation
  def constructLabFeatureTuple(labEvents: RDD[LabEvent]): RDD[FeatureTuple] = {
    labEvents.map { case LabEvent(subject_id, itemid, valuenum, charttime) => ((subject_id, itemid), valuenum) }
      .aggregateByKey((0.0, 0))(
        (acc, value) => (acc._1 + value, acc._2 + 1),
        (part1, part2) => (part1._1 + part2._1, part1._2 + part2._2))
      .map { case ((subject_id, itemid), (sum, count)) => ((subject_id, itemid), sum / count) }
  }

  // Given a feature tuples RDD, construct features in vector
  // format for each patient. feature name should be mapped
  // to some index and convert to sparse feature format.

  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** transform input feature */

    /** create a feature name to id map */

    /**
     * Functions maybe helpful:
     * collect
     * groupByKey
     */

    val featureMap = feature.map { case ((patientID, feature), value) => feature }.distinct.zipWithIndex.collect.toMap
    val scFeatureMap = sc.broadcast(featureMap)
    val numFeature = scFeatureMap.value.size

    val finalSamples = feature.map { case ((patientID, feature), value) => (patientID, (feature, value)) }.groupByKey.map {
      case (patientID, features) =>
        val indexedFeatures = features.toList.map { case (feature, value) => (scFeatureMap.value(feature).toInt, value) }
        (patientID, Vectors.sparse(numFeature, indexedFeatures))
    }

    finalSamples

    /** The feature vectors returned can be sparse or dense. It is advisable to use sparse */

  }
}