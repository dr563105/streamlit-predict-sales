import pandas as pd
import streamlit as st
import datetime

def read_parquet_files(filename):
    """
    Read parquet file format for given filename and returns the contents
    """
    df = pd.read_parquet(filename, engine="pyarrow")
    return df


df_test_preds = read_parquet_files("lgb_preds.parquet")

df_items = read_parquet_files("items.parquet")
@st.cache


def predict(store_nbr: int, item_nbr: int, date1):
    """
    Takes the json inputs, processes it and outputs the unit sales
    """
    try:
        idx = pd.IndexSlice
        # df_items.sample(1).index[0]
        x = df_test_preds.loc[idx[store_nbr, item_nbr, str(date1)]][
            "unit_sales"
        ]
    except KeyError:
        print("This item is not present this store. Try some other item")
        return -0.0
    else:
        return float(round(x, 2))

def main():
    """
    Streamlit main function
    """
    st.markdown("# Grocery Sales Forecasting")
    store_nbr = st.slider('Select Store:', min_value=1, max_value=54, value=4, step=1)
    prediction_date = st.date_input("Select prediction date:", min_value=datetime.date(2017, 8, 16), max_value=datetime.date(2017,8,31), value=datetime.date(2017,8,25))
    gen = st.button("Random item generator")
    
    if gen:
        item = df_items.sample(1)
        item_idx, item_family = item.index[0], item["family"].values[0]
        predicted_unit_sales = predict(store_nbr, item_idx, prediction_date)
        result = {"Store": store_nbr, " item": int(item_idx), "Family": item_family, "Prediction date":str(prediction_date), "Predicted Unit_sales": predicted_unit_sales}
        st.write(result)

if __name__ == "__main__":
    main()
