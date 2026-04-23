require("dotenv").config();
const express = require("express");
const axios = require("axios");
const cors = require("cors");
const mongoose = require("mongoose");

const app = express();
app.use(cors());
app.use(express.json());

// Connect to local MongoDB
mongoose.connect("mongodb://localhost:27017/AiDataBase")
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB error:", err));

// Schema
const patientSchema = new mongoose.Schema({
  patientName: String,
  patientId:   String,
  HR:    Number,
  BP:    String,
  Temp:  Number,
  SpO2:  Number,
  time:  String,
  date:  String,
  risk:   Number,
  status: String,
  recordedAt: { type: Date, default: Date.now },
});

const Patient = mongoose.model("Patient_detail", patientSchema, "Patient_detail");

const historyMap = {}; // per-patient history

// POST /predict — save vitals + get prediction
app.post("/predict", async (req, res) => {
  try {
    const { HR, BP, Temp, SpO2, patientName, patientId, time, date } = req.body;

    const systolic = parseFloat(BP.toString().split('/')[0]);
    const isCritical = SpO2 < 90 || HR > 120 || HR < 40 || Temp > 103 || systolic > 180 || systolic < 70;

    if (!historyMap[patientId]) {
      // Load last 5 readings from MongoDB for this patient
      const recent = await Patient.find({ patientId })
        .sort({ recordedAt: -1 })
        .limit(5)
        .lean();
      historyMap[patientId] = recent.reverse().map(r => {
        const sys = parseFloat(r.BP.toString().split('/')[0]);
        return [r.HR, sys, r.Temp, r.SpO2];
      });
    }
    const history = historyMap[patientId];

    history.push([HR, systolic, Temp, SpO2]);
    if (history.length > 5) history.shift();

    if (!isCritical && history.length < 5) {
      // Save reading without prediction yet
      await Patient.create({ patientName, patientId, HR, BP, Temp, SpO2, time, date });
      return res.json({
        message: `Need ${5 - history.length} more readings`,
        history
      });
    }

    // If critical, pad sequence with current reading to make it 5
    let sequence = history;
    if (isCritical && history.length < 5) {
      const current = [HR, systolic, Temp, SpO2];
      sequence = Array(5).fill(current);
    }

    // Get prediction from Flask
    const response = await axios.post("http://127.0.0.1:5000/predict", {
      sequence
    });

    const { risk, status } = response.data;

    // Send SMS alert if risk > 40%
    if (risk > 40) {
      const criticalParams = [];
      if (SpO2 < 95)             criticalParams.push(`SpO2:${SpO2}%(low)`);
      if (HR > 100 || HR < 60)   criticalParams.push(`HR:${HR}bpm(abnormal)`);
      if (Temp > 99.5)           criticalParams.push(`Temp:${Temp}F(fever)`);
      const sys = parseFloat(BP.toString().split('/')[0]);
      if (sys < 90 || sys > 140) criticalParams.push(`BP:${sys}mmHg(abnormal)`);
      console.log("HR:", HR, "BP:", BP, "Temp:", Temp, "SpO2:", SpO2);
      console.log("Critical params:", criticalParams);
      await sendSMSAlert(patientName, risk, status, criticalParams);
    }

    // Save to MongoDB with prediction
    await Patient.create({ patientName, patientId, HR, BP, Temp, SpO2, time, date, risk, status });

    res.json({ sequence: history, result: { risk, status } });

  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "ML API error" });
  }
});

// GET /records — fetch all saved records
app.get("/records", async (req, res) => {
  try {
    const records = await Patient.find().sort({ recordedAt: -1 }).limit(50);
    res.json(records);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch records" });
  }
});

const twilio = require("twilio");

async function sendSMSAlert(patientName, risk, status, criticalParams) {
  try {
    const client = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
    const paramText = criticalParams.length > 0
      ? ` Params:${criticalParams.join(',')}`
      : '';
    await client.messages.create({
      body: `PULSENET ALERT\nPatient:${patientName}\nRisk:${risk}%\nStatus:${status}${paramText}`,
      from: process.env.TWILIO_PHONE,
      to: `+91${process.env.ALERT_PHONE}`,
    });
    console.log(`SMS alert sent for ${patientName} - Risk: ${risk}%`);
  } catch (err) {
    console.error("SMS failed:", err.message);
  }
}

app.listen(3001, () => console.log("AI Backend running on port 3001"));
