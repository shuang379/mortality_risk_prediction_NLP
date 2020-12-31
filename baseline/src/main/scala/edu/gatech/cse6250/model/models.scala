package edu.gatech.cse6250.model

import java.sql.Date

case class Diagnosis(subject_id: String, icd9_code: String, admittime: Date)

case class Medication(subject_id: String, drg_code: String, admittime: Date)

case class LabEvent(subject_id: String, itemid: String, valuenum: Double, charttime: Date)