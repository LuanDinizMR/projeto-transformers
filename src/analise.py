
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/jena_climate_2009_2016.csv")

print("Dimensões:", df.shape)
print("\nColunas:")
print(df.columns.tolist())

print("\nEstatísticas descritivas:")
print(df.describe().T)

print("\nValores ausentes:")
print(df.isnull().sum())

# Converter datas
df["Date Time"] = pd.to_datetime(
    df["Date Time"], format="%d.%m.%Y %H:%M:%S"
)

# Visualizar a temperatura
plt.figure(figsize=(12, 4))
plt.plot(df["Date Time"], df["T (degC)"], linewidth=0.5)
plt.title("Temperatura ao longo do tempo")
plt.xlabel("Data")
plt.ylabel("Temperatura (°C)")
plt.tight_layout()
plt.savefig("results/temperatura.png")
plt.show()
