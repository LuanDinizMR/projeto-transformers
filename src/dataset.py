
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class ClimateWindows(Dataset):
    def __init__(self, values, tokens_in=24, tokens_out=1, steps_per_token=6):
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.token_size = steps_per_token * values.shape[1]

        usable = (len(values) // steps_per_token) * steps_per_token

        self.tokens = values[:usable].reshape(
            -1, self.token_size
        ).astype(np.float32)

        self.count = max(
            0, len(self.tokens) - tokens_in - tokens_out + 1
        )

        if not self.count:
            raise ValueError(
                "Dados insuficientes para as janelas solicitadas."
            )

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.tokens_in]
        y = self.tokens[
            idx + self.tokens_in:
            idx + self.tokens_in + self.tokens_out
        ]
        return x.copy(), y.copy()


def get_loaders(csv_path, tokens_in=24, tokens_out=1, batch_size=64):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {path}")

    # 1. Carregar os dados
    df = pd.read_csv(path)

    features = [c for c in df.columns if c != "Date Time"]

    if len(features) != 14:
        raise ValueError(
            f"Eram esperadas 14 variáveis; encontradas {len(features)}"
        )

    numeric = df[features].apply(
        pd.to_numeric, errors="coerce"
    )

    # 2. Corrigir os valores inválidos de vento
    wind_columns = ["wv (m/s)", "max. wv (m/s)"]

    print("ANTES da correção:")
    print((numeric[wind_columns] == -9999).sum())

    numeric[wind_columns] = numeric[wind_columns].replace(
        -9999.0, 0.0
    )

    print("\nDEPOIS da correção:")
    print((numeric[wind_columns] == -9999).sum())

    # 3. Verificar valores ausentes
    if numeric.isna().any().any():
        raise ValueError(
            "Foram encontrados valores ausentes/inválidos."
        )

    # 4. Converter para NumPy
    data = numeric.to_numpy(dtype=np.float32)

    # 5. Divisão cronológica
    n = len(data)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)

    raw_train = data[:i_train]
    raw_val = data[i_train:i_val]
    raw_test = data[i_val:]

    # 6. Normalização usando apenas o treinamento
    scaler = StandardScaler()

    train = scaler.fit_transform(raw_train).astype(np.float32)
    val = scaler.transform(raw_val).astype(np.float32)
    test = scaler.transform(raw_test).astype(np.float32)

    # 7. Criar as janelas temporais
    datasets = [
        ClimateWindows(x, tokens_in, tokens_out)
        for x in (train, val, test)
    ]

    # 8. Criar os DataLoaders
    loaders = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(i == 0),
            num_workers=0
        )
        for i, ds in enumerate(datasets)
    ]

    print(f"\nVariáveis: {features}")
    print(
        f"Janelas: treino={len(datasets[0])}, "
        f"validação={len(datasets[1])}, "
        f"teste={len(datasets[2])}"
    )

    return (*loaders, scaler, features)
