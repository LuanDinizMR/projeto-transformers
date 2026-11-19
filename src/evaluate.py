from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


def evaluate_model(model, loader, scaler, features, history, out_dir, device, tokens_out, steps_per_token=6):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    squared_error, absolute_error, total = 0.0, 0.0, 0
    attn_sum, attn_count = None, 0
    example = None
    temp_idx = features.index('T (degC)')
    temp_actual, temp_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prediction, attention = model(x, return_attention=True)
            pred = prediction.cpu().numpy()
            real = y.numpy()
            delta = pred - real
            squared_error += np.sum(delta ** 2, dtype=np.float64)
            absolute_error += np.sum(np.abs(delta), dtype=np.float64)
            total += delta.size
            # Atenção do último bloco, primeira cabeça: (B, tokens, tokens)
            weights = attention[:, 0].cpu().numpy()
            batch_attn = weights.sum(axis=0, dtype=np.float64)
            attn_sum = batch_attn if attn_sum is None else attn_sum + batch_attn
            attn_count += len(x)
            # Erro de temperatura na escala original, considerando todas as medições futuras
            p = pred.reshape(-1, tokens_out, steps_per_token, len(features))[:, :, :, temp_idx]
            r = real.reshape(-1, tokens_out, steps_per_token, len(features))[:, :, :, temp_idx]
            temp_pred.append((p * scaler.scale_[temp_idx] + scaler.mean_[temp_idx]).reshape(-1))
            temp_actual.append((r * scaler.scale_[temp_idx] + scaler.mean_[temp_idx]).reshape(-1))
            if example is None:
                example = ((r[0].reshape(-1) * scaler.scale_[temp_idx] + scaler.mean_[temp_idx]).copy(),
                           (p[0].reshape(-1) * scaler.scale_[temp_idx] + scaler.mean_[temp_idx]).copy())
    rmse = float(np.sqrt(squared_error / total))
    mae = float(absolute_error / total)
    temp_pred = np.concatenate(temp_pred)
    temp_actual = np.concatenate(temp_actual)
    temp_rmse = float(np.sqrt(np.mean((temp_pred - temp_actual)**2)))
    temp_mae = float(np.mean(np.abs(temp_pred - temp_actual)))
    print(f'Teste - RMSE normalizado={rmse:.5f} | MAE normalizado={mae:.5f}')
    print(f'Temperatura - RMSE={temp_rmse:.3f} °C | MAE={temp_mae:.3f} °C')
    (out_dir / 'metricas.txt').write_text(
        f'RMSE normalizado: {rmse:.6f}\nMAE normalizado: {mae:.6f}\n'
        f'RMSE temperatura (°C): {temp_rmse:.6f}\nMAE temperatura (°C): {temp_mae:.6f}\n',
        encoding='utf-8')

    plt.figure(figsize=(8, 4))
    plt.plot(history['train'], label='Treino')
    plt.plot(history['val'], label='Validação')
    plt.xlabel('Época')
    plt.ylabel('MSE normalizado')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'perdas.png', dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(example[0])) / steps_per_token, example[0], label='Real')
    plt.plot(np.arange(len(example[1])) / steps_per_token, example[1], label='Previsto')
    plt.xlabel('Horas a partir do início da previsão')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'temperatura_prevista.png', dpi=160)
    plt.close()

    mean_attention = attn_sum / attn_count
    np.save(out_dir / 'atencao_media.npy', mean_attention)
    plt.figure(figsize=(6, 5))
    plt.imshow(mean_attention, cmap='viridis', aspect='auto')
    plt.colorbar(label='Peso de atenção')
    plt.xlabel('Token-chave (hora passada)')
    plt.ylabel('Token-consulta (hora passada)')
    plt.title('Atenção média: última camada, cabeça 1')
    plt.tight_layout()
    plt.savefig(out_dir / 'mapa_atencao.png', dpi=160)
    plt.close()
    return {'rmse_normalizado': rmse, 'mae_normalizado': mae,
            'rmse_temperatura': temp_rmse, 'mae_temperatura': temp_mae}
