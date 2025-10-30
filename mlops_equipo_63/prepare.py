import pandas as pd
from sklearn.impute import SimpleImputer

def load_and_clean_data(raw_path):
    print(f"→ Cargando archivo desde: {raw_path}")
    df = pd.read_csv(raw_path)

    print("→ Convirtiendo todas las columnas a valores numéricos...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("→ Eliminando filas sin etiqueta 'shares'...")
    before_rows = len(df)
    df.dropna(subset=['shares'], inplace=True)
    print(f"   Filas eliminadas: {before_rows - len(df)}")

    print("→ Quitando columnas irrelevantes...")
    for col in ['url', 'timedelta']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print("→ Imputando valores nulos restantes con la mediana...")
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print(f"→ Datos procesados: {df_imputed.shape[0]} filas, {df_imputed.shape[1]} columnas")

    return df_imputed

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"→ Datos guardados en: {output_path}")

# En tu main
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python prepare.py <input_raw_path> <output_processed_path>")
        sys.exit(1)

    raw_path, output_path = sys.argv[1], sys.argv[2]
    df = load_and_clean_data(raw_path)
    save_processed_data(df, output_path)
