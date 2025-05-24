import pandas as pd

try:
    df = pd.read_parquet("hf://datasets/OpenMedical/medical-data/data/train-00000-of-00001.parquet")
    print("Датасет успешно загружен!")
    print("Первые 5 строк:")
    print(df.head())
    print("\nИнформация о колонках:")
    print(df.info())
    # Посмотрите на уникальные значения в потенциально полезных колонках
    # if 'text_column_name' in df.columns: # Замените 'text_column_name' на реальное имя колонки
    #     print(f"\nПример текста: {df['text_column_name'].iloc[0]}")
except Exception as e:
    print(f"Ошибка при загрузке или обработке датасета: {e}")
