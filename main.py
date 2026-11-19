import argparse
from pathlib import Path
import random
import numpy as np
import torch
from src.dataset import get_loaders
from src.model import ClimateTransformer
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/jena_climate_2009_2016.csv')
    parser.add_argument('--horizon', type=int, choices=[1, 24], default=1)
    parser.add_argument('--tokens-in', type=int, default=24)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Dispositivo: {device}')
    train_loader, val_loader, test_loader, scaler, features = get_loaders(
        args.csv, tokens_in=args.tokens_in, tokens_out=args.horizon,
        batch_size=args.batch_size)
    model = ClimateTransformer(
        token_size=6 * len(features), tokens_out=args.horizon,
        dim=args.dim, heads=args.heads, layers=args.layers)
    history = train_model(model, train_loader, val_loader, args.epochs, args.lr, device)
    out_dir = Path(args.out or f'results/h{args.horizon}_m{args.tokens_in}_d{args.dim}_c{args.heads}_b{args.layers}')
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'model_state': model.state_dict(), 'config': vars(args),
                'feature_names': features, 'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_}, out_dir / 'modelo.pt')
    evaluate_model(model, test_loader, scaler, features, history, out_dir, device, args.horizon)
    print(f'Resultados salvos em: {out_dir}')


if __name__ == '__main__':
    main()
