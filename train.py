import torch
from argparse import ArgumentParser
from ultralytics import YOLO, settings

parser = ArgumentParser()

parser.add_argument('--project', type=str, default='.', required=True)
parser.add_argument('--name', type=str, default='model', required=True)
parser.add_argument('--data', type=str, default='data.yaml', required=True)

parser.add_argument('--epochs', type=int, default=50, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--imgsz', type=int, default=640, required=True)

parser.add_argument('--save', type=bool, default=True)
parser.add_argument('--interval', type=int, default=-1)

parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr0', type=float, default=1e-4)
parser.add_argument('--lrf', type=float, default=5e-4)

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--weight_decayf', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--freeze', type=int, default=0)
parser.add_argument('--augment', type=bool, default=True)
parser.add_argument('--close_mosaic', type=int, default=10)

args = parser.parse_args()


if __name__ == '__main__':
    # Enable mlflow logging
    settings.update({'mlflow': True})

    # A small model would suffice
    model = YOLO('models/cardetect/weights/best.pt', task='detect')

    # Train on GPU if is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model on the dataset. Configure training using arguments
    result = model.train(
        project=args.project, name=args.name, data=args.data, plots=True,
        device=device, cache=True, workers=8,
        epochs=args.epochs,  batch=args.batch_size, imgsz=args.imgsz,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf, momentum=args.momentum,
        auto_augment=args.augment, close_mosaic=args.close_mosaic, freeze=args.freeze
    )

    # Export trained model weight in standardizes format
    model.export(format='onnx')

    # Disable changes to ultralytics
    settings.reset()