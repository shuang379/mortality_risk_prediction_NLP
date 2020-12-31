package edu.gatech.cse6250.model

import java.sql.Date

case class Diagnosis(subject_id: String, icd9_code: String, admittime: Date)
case class Medication(subject_id: String, drg_code: String, admittime: Date)
case class LabEvent(subject_id: String, itemid: String, valuenum: Double, charttime: Date)
case class Count(subject_id: String, word: String, count: Integer, chartime: Date)
case class Dead(subject_id: String, deathtime: Date, mortality: String)
case class Alive(subject_id: String, mortality: String)
case class Admission(subject_id: String, deathtime: String, mortality: String)
case class Mortality(subject_id: String, mortality: String)