from src.data_extraction import load_taxi_data

if __name__ == "__main__":
    file_path = "data/yellow_tripdata_2015-01.csv"

    df = load_taxi_data(file_path)

    print(f"Кількість рядків у DataFrame: {df.count()}")
    print(f"Кількість колонок: {len(df.columns)}")
