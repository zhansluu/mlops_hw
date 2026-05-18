import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import yaml
import boto3
from botocore.client import Config
import json


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_dir = self._find_img_dir()
        print(f'Папка с картинками: {self.img_dir}')

    def _find_img_dir(self):
        for folder in ['train_data', 'test_data', 'data']:
            path = self.root_dir / folder
            if path.exists():
                return path
        return self.root_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_path = self.data.iloc[idx]['file_name']
        filename = Path(raw_path).name
        img_path = self.img_dir / filename
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(dataloader, desc='Finetune'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Test'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    epoch_precision = precision_score(all_labels, all_preds, average='weighted')
    epoch_recall = recall_score(all_labels, all_preds, average='weighted')
    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


def download_from_s3(s3_cfg, local_path):
    s3 = boto3.client(
        's3',
        endpoint_url=s3_cfg['endpoint_url'],
        aws_access_key_id=s3_cfg['access_key'],
        aws_secret_access_key=s3_cfg['secret_key'],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    s3.download_file(s3_cfg['bucket'], s3_cfg['model_v1_key'], local_path)
    print(f"Модель скачана из S3 -> {local_path}")


def upload_to_s3(local_path, s3_cfg):
    s3 = boto3.client(
        's3',
        endpoint_url=s3_cfg['endpoint_url'],
        aws_access_key_id=s3_cfg['access_key'],
        aws_secret_access_key=s3_cfg['secret_key'],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    s3.upload_file(local_path, s3_cfg['bucket'], s3_cfg['model_v2_key'])
    print(f"Модель загружена в S3: s3://{s3_cfg['bucket']}/{s3_cfg['model_v2_key']}")


def main():
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    cfg = params['finetune']
    s3_cfg = params['s3']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    Path(cfg['log_dir']).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg['log_dir'])
    writer.add_text('hyperparameters', str(cfg))

    download_from_s3(s3_cfg, cfg['base_model_path'])

    train_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((cfg['img_size'], cfg['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(cfg['train_csv'], cfg['train_data_dir'], train_transform)
    test_dataset = ImageDataset(cfg['test_csv'], cfg['test_data_dir'], test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                             shuffle=False, num_workers=cfg['num_workers'])

    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg['num_classes'])
    model.load_state_dict(torch.load(cfg['base_model_path'], map_location=device))
    model = model.to(device)
    print('Базовая модель загружена')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['scheduler_step_size'],
        gamma=cfg['scheduler_gamma']
    )

    for epoch in range(cfg['num_epochs']):
        print(f'\nEpoch {epoch+1}/{cfg["num_epochs"]}')
        print('-' * 50)

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        writer.add_scalar('Loss/finetune', train_loss, epoch)
        writer.add_scalar('Accuracy/finetune', train_acc, epoch)
        writer.add_scalar('F1/finetune', train_f1, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        print(f'Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}')

    print('\nДообучение завершено!')

    print('\nОценка на Test_2...')
    test_loss, test_acc, test_f1, test_prec, test_rec = validate(
        model, test_loader, criterion, device
    )
    print(f'Test Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  '
          f'F1: {test_f1:.4f}  Prec: {test_prec:.4f}  Rec: {test_rec:.4f}')

    writer.add_scalar('Loss/test', test_loss, 0)
    writer.add_scalar('Accuracy/test', test_acc, 0)
    writer.add_scalar('F1/test', test_f1, 0)
    writer.add_scalar('Precision/test', test_prec, 0)
    writer.add_scalar('Recall/test', test_rec, 0)
    writer.close()

    torch.save(model.state_dict(), cfg['model_save_path'])
    print(f'Модель сохранена: {cfg["model_save_path"]}')

    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec
    }
    with open('models/metrics_v2.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Метрики v2:', metrics)

    upload_to_s3(cfg['model_save_path'], s3_cfg)


if __name__ == '__main__':
    main()
