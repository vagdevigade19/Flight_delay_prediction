import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array


MODEL_PATH = "/Users/nikhilsubramanya/Desktop/flight-project/models/lr_model"

spark = (SparkSession.builder
         .appName("FlightDelayPredictor")
         .getOrCreate())

model = PipelineModel.load(MODEL_PATH)

st.title("✈️ Flight Delay Predictor")
st.caption("Predict probability of flight delay ≥15 minutes")

origin = st.text_input("Origin (IATA)", "JFK").upper()
dest = st.text_input("Destination (IATA)", "SFO").upper()
carrier = st.text_input("Carrier Code", "AA").upper()
month = st.number_input("Month (1–12)", 1, 12, 7)
dow = st.number_input("Day of Week (1=Mon … 7=Sun)", 1, 7, 5)
dep_time = st.text_input("Scheduled Departure Time (HHMM)", "1730")

def get_hour(hhmm):
    try:
        return int(hhmm[:2])
    except:
        return None

dep_hour = get_hour(dep_time)

if st.button("Predict Delay Probability"):
    row = {"MONTH": month, "DAY_OF_WEEK": dow, "DEP_HOUR": dep_hour,
           "OP_UNIQUE_CARRIER": carrier, "ORIGIN": origin, "DEST": dest}
    pdf = pd.DataFrame([row])
    df = spark.createDataFrame(pdf)

    pred = model.transform(df)

# convert probability vector -> array, then take P(class=1)
    pred = pred.withColumn("probability_arr", vector_to_array("probability"))
    pred = pred.withColumn("p_delayed", F.col("probability_arr")[1])

    result = pred.select("p_delayed", "prediction").toPandas().iloc[0]


    prob = result["p_delayed"]
    pred_class = int(result["prediction"])
    st.metric("Chance of Delay (≥15 min)", f"{prob*100:.1f}%")
    st.write("Prediction:", "Delayed" if pred_class == 1 else "On-time")

