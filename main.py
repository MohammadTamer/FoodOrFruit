import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
from datetime import datetime
import random
import time
import torch.nn.functional as F

# -------------------- CONFIG --------------------
CONFIG = {
    'data_root': 'data',
    'output_dir': 'stage1_output',
    'model_save_path': 'stage1_output/prototypical_classifier.pth',
    'fruit_model_path': 'stage1_output/fruit_model.pth',
    'siamese_path': 'stage1_output/siamese_embedding.pth',
    'support_mean_path': 'stage1_output/support_mean.pt',
    'log_file': 'stage1_output/training_log.txt',
    'batch_size': 8,
    'num_epochs': 1,
    'learning_rate': 0.001,
    'image_size': 224,
    'validation_split': 0.2,
    'seed': 42,
    'num_workers': 4,
    # additional small params
    'fruit_epochs': 3,
    'fruit_lr': 1e-4,
    'siamese_epochs': 3,
    'siamese_lr': 1e-4,
    'siamese_batch': 16,
    'siamese_margin': 0.5,
    'siamese_pairs_per_class': 200,
    'siamese_threshold': 0.6,
    # logging control: 'DEBUG' (verbose), 'INFO' (concise), 'ERROR' (only errors)
    'log_level': 'INFO'
}


#  Utilities

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_output_directories():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(CONFIG['output_dir'], 'predictions'), exist_ok=True)


def log_message(message, level='INFO'):
    levels = {'DEBUG': 10, 'INFO': 20, 'ERROR': 40}
    configured = levels.get(CONFIG.get('log_level', 'INFO'), 20)
    msg_level = levels.get(level, 20)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full = f"[{timestamp}] [{level}] {message}"
    try:
        with open(CONFIG['log_file'], 'a', encoding='utf-8') as f:
            f.write(full + '')
    except Exception:
        pass
    if msg_level >= configured:
        print(full)


def save_predictions_to_file(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Image Name | TopCategory | TopConf | Subclass | SubConf")
        f.write("=" * 120 + "")
        for pred in predictions:
            image_name = os.path.basename(pred['image_path'])
            subcls = pred.get('subclass', '')
            subconf = pred.get('sub_conf', 0.0)
            f.write(f"{image_name} | {pred['prediction']} | {pred['confidence']:.4f} | {subcls} | {subconf:.4f}")


# Data Preparation

def get_all_image_paths(data_root):
    image_paths = {'Food': [], 'Fruit': []}
    exts = ('.jpg', '.png', '.jpeg')

    food_dir = os.path.join(data_root, 'Food')
    if os.path.exists(food_dir):
        for split in os.listdir(food_dir):
            split_path = os.path.join(food_dir, split)
            if not os.path.isdir(split_path):
                continue
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for file in os.listdir(class_path):
                    if file.lower().endswith(exts):
                        image_paths['Food'].append(os.path.join(class_path, file))

    fruit_dir = os.path.join(data_root, 'Fruit')
    if os.path.exists(fruit_dir):
        for split in os.listdir(fruit_dir):
            split_path = os.path.join(fruit_dir, split)
            if not os.path.isdir(split_path):
                continue
            for class_name in os.listdir(split_path):
                class_path = os.path.join(split_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                images_dir = os.path.join(class_path, 'Images')
                if not os.path.exists(images_dir):
                    continue
                for file in os.listdir(images_dir):
                    if file.lower().endswith(exts):
                        image_paths['Fruit'].append(os.path.join(images_dir, file))

    return image_paths


def split_dataset(image_paths, validation_split=0.2):
    train_paths = {'Food': [], 'Fruit': []}
    val_paths = {'Food': [], 'Fruit': []}

    for category in ['Food', 'Fruit']:
        images = list(image_paths[category])
        random.shuffle(images)
        split_idx = int(len(images) * (1 - validation_split))
        train_paths[category] = images[:split_idx]
        val_paths[category] = images[split_idx:]
        log_message(f"{category}: {len(train_paths[category])} train | {len(val_paths[category])} val", 'DEBUG')

    return {'train': train_paths, 'validation': val_paths}


def load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception:
        return None


class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        assert len(file_paths) == len(labels)
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        image = load_image(path)
        if image is None:
            image = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']), (0, 0, 0))
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)
        return img_tensor, torch.tensor(label, dtype=torch.long)


# COMPLETE MODEL – Binary Classification

def create_complete_model(num_classes=2):
    resnet = models.resnet50()
    embedding_net = nn.Sequential(*list(resnet.children())[:-1])

    classifier_head = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

    full_model = nn.Sequential(
        embedding_net,
        nn.Flatten(),
        classifier_head
    )

    return full_model


def train_epoch(model, train_loader, optimizer, loss_calc_method, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_images, batch_labels in train_loader:
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        model_output = model(batch_images)
        loss = loss_calc_method(model_output, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(model_output.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, loss_calc_method, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            model_output = model(batch_images)
            loss = loss_calc_method(model_output, batch_labels)
            total_loss += loss.item()
            _, predicted = torch.max(model_output.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    epoch_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0) if len(
        all_labels) > 0 else 0.0
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0) if len(
        all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0) if len(all_labels) > 0 else 0.0

    metrics = {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels
    }

    return metrics


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    loss_calc_method = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    train_history = {'loss': [], 'accuracy': [], 'val_accuracy': [], 'val_loss': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_calc_method, device)
        val_metrics = validate_epoch(model, val_loader, loss_calc_method, device)

        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_accuracy'].append(val_metrics['accuracy'])

        # concise epoch summary printed
        log_message(
            f"Epoch {epoch + 1}/{num_epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_metrics['accuracy']:.2f}% | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}",
            'INFO')

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), CONFIG['model_save_path'])
            log_message(f"Best binary model saved (val_acc={best_val_acc:.2f}%).", 'INFO')

    return train_history


def evaluate_on_validation(model, val_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidence = []

    with torch.no_grad():
        for batch_images, batch_labels in val_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            model_output = model(batch_images)
            probs = torch.softmax(model_output, dim=1)
            confidence, predicted = torch.max(probs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_confidence.extend(confidence.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) if len(all_labels) > 0 else 0.0
    precision = precision_score(all_labels, all_predictions, average='weighted') if len(all_labels) > 0 else 0.0
    recall = recall_score(all_labels, all_predictions, average='weighted') if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_predictions, average='weighted') if len(all_labels) > 0 else 0.0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidence': all_confidence
    }

    return metrics


def predict_single_image(model, image_path, transform, device):
    image = load_image(image_path)
    if image is None:
        return None
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        model_output = model(image_tensor)
        probs = torch.softmax(model_output, dim=1)
        confidence, predicted = torch.max(probs, 1)
    categories = ['Food', 'Fruit']
    return {'category': categories[predicted.item()], 'confidence': float(confidence.item())}


# FRUIT MODEL – Multi-class Classification
def build_class_to_paths_for_fruit(data_root: str):
    fruit_root = os.path.join(data_root, 'Fruit')
    class_to_paths = {}
    if not os.path.exists(fruit_root):
        return {}, []
    for split in os.listdir(fruit_root):
        split_path = os.path.join(fruit_root, split)
        if not os.path.isdir(split_path):
            continue
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            images_dir = os.path.join(cls_path, 'Images')
            if not os.path.exists(images_dir):
                continue
            imgs = []
            for f0 in os.listdir(images_dir):
                if f0.lower().endswith(('.jpg', '.png', '.jpeg')):
                    imgs.append(os.path.join(images_dir, f0))
            if len(imgs) == 0:
                continue
            if cls not in class_to_paths:
                class_to_paths[cls] = imgs
            else:
                class_to_paths[cls].extend(imgs)
    class_names = sorted(list(class_to_paths.keys()))
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    mapped = {name_to_idx[name]: class_to_paths[name] for name in class_names}
    return mapped, class_names


def create_fruit_model(num_classes):
    resnet = models.resnet50()
    embedding_net = nn.Sequential(*list(resnet.children())[:-1])
    classifier_head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    full_model = nn.Sequential(
        embedding_net,
        classifier_head
    )
    return full_model


def train_fruit_classifier(fruit_map, fruit_class_names, train_transform, val_transform, device):
    if len(fruit_map) == 0 or len(fruit_class_names) == 0:
        log_message("No fruit classes found; skipping fruit training.", 'DEBUG')
        return None

    log_message("Training Fruit multiclass classifier...", 'INFO')

    fruit_train_files = []
    fruit_train_labels = []
    fruit_val_files = []
    fruit_val_labels = []

    for cls_idx, paths in fruit_map.items():
        paths_copy = list(paths)
        random.shuffle(paths_copy)
        split = int(0.8 * len(paths_copy))
        train_p = paths_copy[:split]
        val_p = paths_copy[split:]
        fruit_train_files.extend(train_p)
        fruit_train_labels.extend([cls_idx] * len(train_p))
        fruit_val_files.extend(val_p)
        fruit_val_labels.extend([cls_idx] * len(val_p))

    if len(fruit_train_files) == 0:
        log_message("Not enough fruit images to train multi-class fruit model.", 'INFO')
        return None

    fruit_train_ds = ImageDataset(fruit_train_files, fruit_train_labels, transform=train_transform)
    fruit_val_ds = ImageDataset(fruit_val_files, fruit_val_labels, transform=val_transform)

    fruit_train_loader = DataLoader(fruit_train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                                    num_workers=CONFIG.get('num_workers', 0))
    fruit_val_loader = DataLoader(fruit_val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                                  num_workers=CONFIG.get('num_workers', 0))

    fruit_model = create_fruit_model(num_classes=len(fruit_class_names)).to(device)
    fruit_optimizer = optim.Adam(fruit_model.parameters(), lr=CONFIG['fruit_lr'])
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(CONFIG.get('fruit_epochs', 3)):
        fruit_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in fruit_train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            fruit_optimizer.zero_grad()
            out = fruit_model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            fruit_optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total if total > 0 else 0.0

        fruit_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vimgs, vlabels in fruit_val_loader:
                vimgs = vimgs.to(device)
                vlabels = vlabels.to(device)
                out = fruit_model(vimgs)
                _, preds = torch.max(out, 1)
                val_correct += (preds == vlabels).sum().item()
                val_total += vlabels.size(0)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0.0

        log_message(
            f"[Fruit] Epoch {epoch + 1}/{CONFIG.get('fruit_epochs', 3)} | TrainAcc: {train_acc:.2f}% | ValAcc: {val_acc:.2f}%",
            'INFO')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(fruit_model.state_dict(), CONFIG['fruit_model_path'])
            log_message(f"Best fruit model saved (val_acc={best_val_acc:.2f}%).", 'INFO')

    return fruit_model


def predict_fruit(model, image_path, transform, device, class_names):
    image = load_image(image_path)
    if image is None:
        return {'class': None, 'confidence': 0.0}
    tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return {'class': class_names[idx.item()], 'confidence': float(conf.item())}


# FOOD MODEL – Siamese / Few-Shot
def create_siamese_embedding(out_dim=512):
    """
    Return a lightweight convolutional embedding model implemented with nn.Sequential.
    The returned model outputs raw embeddings (not L2-normalized). Normalization
    should be applied where the model is called (e.g. in train_siamese), exactly like
    the existing code does: e = F.normalize(e, p=2, dim=1)
    """

    backbone = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 224 -> 112
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),

        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 28 -> 14
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d((1, 1))  # -> (B, 256, 1, 1)
    )

    head = nn.Sequential(
        nn.Flatten(),  # -> (B, 256)
        nn.Linear(256, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, out_dim)  # -> (B, out_dim)
    )

    model = nn.Sequential(backbone, head)
    return model


def train_siamese(embedding_model, class_to_paths, device,
                  epochs=3, batch_size=16, lr=1e-4, margin=0.5, pairs_per_class=200, transform=None):
    embedding_model.to(device)
    optimizer = optim.Adam(embedding_model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss(margin=margin)
    classes = list(class_to_paths.keys())
    total_pairs = max(1, pairs_per_class * max(1, len(classes)))
    steps_per_epoch = max(1, total_pairs // batch_size)
    for epoch in range(epochs):
        embedding_model.train()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            batch_t1 = []
            batch_t2 = []
            batch_y = []
            for b in range(batch_size):
                if random.random() < 0.5:
                    cls = random.choice(classes)
                    paths = class_to_paths[cls]
                    if len(paths) >= 2:
                        p1, p2 = random.sample(paths, 2)
                    else:
                        p1 = p2 = random.choice(paths)
                    label = 1.0
                else:
                    cls1 = random.choice(classes)
                    cls2 = random.choice([c for c in classes if c != cls1]) if len(classes) > 1 else cls1
                    p1 = random.choice(class_to_paths[cls1])
                    p2 = random.choice(class_to_paths[cls2])
                    label = -1.0
                img1 = load_image(p1)
                img2 = load_image(p2)
                if img1 is None:
                    img1 = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']), (0, 0, 0))
                if img2 is None:
                    img2 = Image.new('RGB', (CONFIG['image_size'], CONFIG['image_size']), (0, 0, 0))
                if transform is not None:
                    t1 = transform(img1)
                    t2 = transform(img2)
                else:
                    t1 = transforms.ToTensor()(img1)
                    t2 = transforms.ToTensor()(img2)
                batch_t1.append(t1.unsqueeze(0))
                batch_t2.append(t2.unsqueeze(0))
                batch_y.append(label)
            x1 = torch.cat(batch_t1, dim=0).to(device)
            x2 = torch.cat(batch_t2, dim=0).to(device)
            y = torch.tensor(batch_y, dtype=torch.float, device=device)
            optimizer.zero_grad()
            e1 = embedding_model(x1)
            e2 = embedding_model(x2)
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
            loss = criterion(e1, e2, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / steps_per_epoch
        log_message(f"[Siamese] Epoch {epoch + 1}/{epochs} - AvgLoss: {avg:.4f}", 'INFO')
    torch.save(embedding_model.state_dict(), CONFIG['siamese_path'])
    log_message(f"Siamese embedding saved to {CONFIG['siamese_path']}", 'INFO')
    return embedding_model


def train_siamese_embedding_for_food(food_map, train_transform, val_transform, device):
    log_message("Training Siamese embedding for Food...", 'INFO')
    siamese_model = create_siamese_embedding(out_dim=512)
    siamese_model = train_siamese(siamese_model, food_map, device,
                                  epochs=CONFIG.get('siamese_epochs', 3),
                                  batch_size=CONFIG.get('siamese_batch', 16),
                                  lr=CONFIG.get('siamese_lr', 1e-4),
                                  margin=CONFIG.get('siamese_margin', 0.5),
                                  pairs_per_class=CONFIG.get('siamese_pairs_per_class', 200),
                                  transform=train_transform)
    support_mean = build_support_embeddings(siamese_model, food_map, device, val_transform, max_images_per_class=50)
    return siamese_model, support_mean


def build_support_embeddings(embedding_model, class_to_paths, device, transform, max_images_per_class=50):
    embedding_model.to(device)
    embedding_model.eval()
    support_mean = {}
    with torch.no_grad():
        for cls, paths in class_to_paths.items():
            chosen = paths[:max_images_per_class]
            embs = []
            for p in chosen:
                img = load_image(p)
                if img is None:
                    continue
                t = transform(img).unsqueeze(0).to(device)
                e = embedding_model(t).cpu().squeeze(0)
                e = F.normalize(e, p=2, dim=0)
                embs.append(e)
            if len(embs) == 0:
                support_mean[cls] = None
            else:
                mean_emb = torch.stack(embs).mean(dim=0)
                mean_emb = F.normalize(mean_emb, p=2, dim=0)
                support_mean[cls] = mean_emb
    torch.save(support_mean, CONFIG['support_mean_path'])
    log_message(f"Support embeddings saved to {CONFIG['support_mean_path']}", 'INFO')
    return support_mean


def few_shot_predict(embedding_model, image_path, transform, device, support_mean, class_names, threshold=0.6):
    image = load_image(image_path)
    if image is None:
        return {'class': None, 'confidence': 0.0}
    tensor = transform(image).unsqueeze(0).to(device)
    embedding_model.eval()
    with torch.no_grad():
        emb = embedding_model(tensor).cpu().squeeze(0)
        emb = F.normalize(emb, p=2, dim=0)
    best_sim = -1.0
    best_cls = None
    for cls_idx, mean_emb in support_mean.items():
        if mean_emb is None:
            continue
        sim = F.cosine_similarity(emb.unsqueeze(0), mean_emb.unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_cls = cls_idx
    if best_sim < threshold or best_cls is None:
        return {'class': 'no_match', 'confidence': best_sim}
    return {'class': class_names[best_cls], 'confidence': best_sim}


def build_class_to_paths_for_food(data_root: str):
    food_root = os.path.join(data_root, 'Food')
    class_to_paths = {}
    if not os.path.exists(food_root):
        return {}, []
    for split in os.listdir(food_root):
        split_path = os.path.join(food_root, split)
        if not os.path.isdir(split_path):
            continue
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
            imgs = []
            for f0 in os.listdir(cls_path):
                if f0.lower().endswith(('.jpg', '.png', '.jpeg')):
                    imgs.append(os.path.join(cls_path, f0))
            if len(imgs) == 0:
                continue
            if cls not in class_to_paths:
                class_to_paths[cls] = imgs
            else:
                class_to_paths[cls].extend(imgs)
    class_names = sorted(list(class_to_paths.keys()))
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    mapped = {name_to_idx[name]: class_to_paths[name] for name in class_names}
    return mapped, class_names

# Segmentation

class FruitSegmentationDataset(Dataset):
    """
    Works for:
    - Binary segmentation (mask: 0 background, 255 fruit)
    - Multi-class segmentation (mask: 0 background, 1..30 fruit classes)
    """

    def __init__(self, image_paths, mask_paths, transform_img=None, transform_mask=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        if self.transform_img:
            img = self.transform_img(img)

        if self.transform_mask:
            mask = self.transform_mask(mask)
        else:
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.d1 = DoubleConv(in_channels, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bridge = DoubleConv(256, 512)

        self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))

        b = self.bridge(self.pool(d3))

        x = self.u3(b)
        x = self.c3(torch.cat([x, d3], dim=1))
        x = self.u2(x)
        x = self.c2(torch.cat([x, d2], dim=1))
        x = self.u1(x)
        x = self.c1(torch.cat([x, d1], dim=1))

        return self.out(x)


def train_binary_segmentation(image_paths, mask_paths, device):
    log_message("Training Fruit Binary Segmentation (Part D)...", "INFO")

    model = UNet(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    mask_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    ds = FruitSegmentationDataset(image_paths, mask_paths, img_tf, mask_tf)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    for epoch in range(3):
        model.train()
        loss_sum = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        log_message(f"[Binary Seg] Epoch {epoch + 1}/3 | Loss: {loss_sum / len(loader):.4f}", "INFO")

    torch.save(model.state_dict(), "stage1_output/binary_segmentation.pth")
    log_message("Binary segmentation model saved.", "INFO")
    return model


def train_multiclass_segmentation(image_paths, mask_paths, num_classes, device):
    log_message("Training Fruit Multi-Class Segmentation (Part E)...", "INFO")

    model = UNet(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    img_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Updated mask transform
    def mask_transform(x):
        mask_np = np.array(x).astype(np.int64)  # ensure int64
        # Map unique mask IDs to 0..num_classes-1
        unique_ids = sorted(np.unique(mask_np))
        id_map = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        mask_mapped = np.vectorize(id_map.get)(mask_np)
        mask_mapped = np.clip(mask_mapped, 0, num_classes - 1)  # safety clip
        return torch.from_numpy(mask_mapped).long()

    mask_tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.NEAREST),
        transforms.Lambda(mask_transform)
    ])

    ds = FruitSegmentationDataset(image_paths, mask_paths, img_tf, mask_tf)
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    for epoch in range(3):
        model.train()
        loss_sum = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        log_message(f"[Multi Seg] Epoch {epoch + 1}/3 | Loss: {loss_sum / len(loader):.4f}", "INFO")

    torch.save(model.state_dict(), "stage1_output/multiclass_segmentation.pth")
    log_message("Multi-class segmentation model saved.", "INFO")
    return model


def save_segmentation_output(model, image_path, save_path, device, multiclass=False):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)

    if multiclass:
        mask = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
    else:
        mask = (torch.sigmoid(out) > 0.5).squeeze().cpu().numpy() * 255

    Image.fromarray(mask.astype(np.uint8)).save(save_path)


def collect_fruit_segmentation_data(data_root):
    """
    Collect images and masks for fruit segmentation.
    Matches images in <Class>/Images with masks in <Class>/Mask.
    Filenames must match ignoring extension and '_mask' suffix.
    """
    images = []
    masks = []

    seg_root = os.path.join(data_root, 'Fruit', 'Validation')
    log_message(f"[DEBUG] Looking for classes in: {seg_root}", 'INFO')

    if not os.path.exists(seg_root):
        log_message(f"[ERROR] Segmentation root folder not found: {seg_root}", 'ERROR')
        return images, masks

    class_folders = [f for f in os.listdir(seg_root) if os.path.isdir(os.path.join(seg_root, f))]
    log_message(f"[DEBUG] Found class folders: {class_folders}", 'INFO')

    for cls in class_folders:
        img_dir = os.path.join(seg_root, cls, 'Images')
        mask_dir = os.path.join(seg_root, cls, 'Mask')

        log_message(f"[DEBUG] Processing class: {cls}", 'INFO')
        log_message(f"[DEBUG] Images dir: {img_dir}", 'INFO')
        log_message(f"[DEBUG] Mask dir: {mask_dir}", 'INFO')

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            log_message(f"[WARNING] Missing Images or Mask folder for class {cls}, skipping.", 'WARNING')
            continue

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Build dict of mask filenames without extension / _mask
        mask_dict = {}
        for m in mask_files:
            key = os.path.splitext(m)[0].replace('_mask', '')
            mask_dict[key] = os.path.join(mask_dir, m)

        # Match images
        for img in img_files:
            key = os.path.splitext(img)[0]  # remove extension
            if key in mask_dict:
                images.append(os.path.join(img_dir, img))
                masks.append(mask_dict[key])
                log_message(f"[DEBUG] Matched image & mask: {img} -> {os.path.basename(mask_dict[key])}", 'INFO')
            else:
                log_message(f"[WARNING] No matching mask for image: {os.path.join(img_dir, img)}", 'WARNING')

    log_message(f"[DEBUG] Total images collected: {len(images)}", 'INFO')
    log_message(f"[DEBUG] Total masks collected: {len(masks)}", 'INFO')

    return images, masks


# Main pipeline

def main():
    set_seed(CONFIG['seed'])
    create_output_directories()

    # brief config print
    log_message(
        f"Config summary: data_root={CONFIG['data_root']}, output_dir={CONFIG['output_dir']}, batch_size={CONFIG['batch_size']}, epochs={CONFIG['num_epochs']}",
        'INFO')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}", 'INFO')

    # -------------------- STAGE 1: Data --------------------
    log_message("STAGE 1: Data preparation", 'INFO')
    image_paths = get_all_image_paths(CONFIG['data_root'])
    n_food = len(image_paths['Food'])
    n_fruit = len(image_paths['Fruit'])
    log_message(f"Collected images -> Food: {n_food}, Fruit: {n_fruit}, Total: {n_food + n_fruit}", 'INFO')

    split_data = split_dataset(image_paths, CONFIG['validation_split'])

    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(CONFIG['image_size'], scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_files = split_data['train']['Food'] + split_data['train']['Fruit']
    train_labels = [0] * len(split_data['train']['Food']) + [1] * len(split_data['train']['Fruit'])
    val_files = split_data['validation']['Food'] + split_data['validation']['Fruit']
    val_labels = [0] * len(split_data['validation']['Food']) + [1] * len(split_data['validation']['Fruit'])

    log_message(f"Train samples: {len(train_files)} | Validation samples: {len(val_files)}", 'INFO')

    train_dataset = ImageDataset(train_files, train_labels, transform=train_transform)
    val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=CONFIG.get('num_workers', 0), pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG.get('num_workers', 0), pin_memory=(device.type == 'cuda'))

    # -------------------- STAGE 2: Binary classification --------------------
    log_message("STAGE 2: Build & train binary Food-vs-Fruit model", 'INFO')
    model = create_complete_model(num_classes=2).to(device)

    start_time = time.time()
    train_history = train_model(model, train_loader, val_loader, CONFIG['num_epochs'], CONFIG['learning_rate'], device)
    training_time = time.time() - start_time
    log_message(f"Binary training done in {training_time / 60:.2f} minutes", 'INFO')

    # -------------------- STAGE 3: Evaluation --------------------
    if os.path.exists(CONFIG['model_save_path']):
        try:
            model.load_state_dict(torch.load(CONFIG['model_save_path'], map_location=device))
            log_message(f"Loaded best binary model from {CONFIG['model_save_path']}", 'INFO')
        except Exception as e:
            log_message(f"Failed to load best binary model: {e}", 'ERROR')

    val_metrics = evaluate_on_validation(model, val_loader, device)
    log_message("Binary model evaluation:", 'INFO')
    log_message(
        f"  Accuracy: {val_metrics['accuracy'] * 100:.2f}% | Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}",
        'INFO')

    # -------------------- STAGE 4: Class maps --------------------
    log_message("Building class maps for Food & Fruit", 'INFO')
    food_map, food_class_names = build_class_to_paths_for_food(CONFIG['data_root'])
    fruit_map, fruit_class_names = build_class_to_paths_for_fruit(CONFIG['data_root'])

    # save class names
    with open(os.path.join(CONFIG['output_dir'], 'food_class_names.json'), 'w', encoding='utf-8') as f:
        json.dump(food_class_names, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CONFIG['output_dir'], 'fruit_class_names.json'), 'w', encoding='utf-8') as f:
        json.dump(fruit_class_names, f, indent=2, ensure_ascii=False)

    log_message(f"Food classes: {len(food_class_names)} | Fruit classes: {len(fruit_class_names)}", 'INFO')

    # -------------------- STAGE 5: Fruit classifier --------------------
    fruit_model = train_fruit_classifier(fruit_map, fruit_class_names, train_transform, val_transform, device)

    # -------------------- STAGE 6: Siamese embedding for Food --------------------
    siamese_model, support_mean = train_siamese_embedding_for_food(food_map, train_transform, val_transform, device)

    # -------------------- STAGE 7: Batch predictions --------------------
    log_message("Running batch predictions on validation set (subclass inference)", 'INFO')
    all_val_images = val_files
    integrated_predictions = []
    for img_path in all_val_images:
        top = predict_single_image(model, img_path, val_transform, device)
        if top is None:
            continue
        entry = {'image_path': img_path, 'prediction': top['category'], 'confidence': top['confidence'],
                 'subclass': None, 'sub_conf': 0.0}
        if top['category'] == 'Fruit' and fruit_model is not None and len(fruit_class_names) > 0:
            r = predict_fruit(fruit_model, img_path, val_transform, device, fruit_class_names)
            entry['subclass'] = r['class']
            entry['sub_conf'] = r['confidence']
        elif top['category'] == 'Food' and siamese_model is not None and len(support_mean) > 0:
            r = few_shot_predict(siamese_model, img_path, val_transform, device, support_mean, food_class_names,
                                 threshold=CONFIG.get('siamese_threshold', 0.6))
            entry['subclass'] = r['class']
            entry['sub_conf'] = r['confidence']
        integrated_predictions.append(entry)

    save_predictions_to_file(integrated_predictions, os.path.join(CONFIG['output_dir'], 'predictions',
                                                                  'validation_predictions_with_subclasses.txt'))

    # -------------------- STAGE 8: Fruit Binary Segmentation --------------------
    log_message("STAGE 8: Fruit Binary Segmentation (Part D)", 'INFO')
    seg_images, seg_masks = collect_fruit_segmentation_data(CONFIG['data_root'])
    log_message(f"Seg images found: {len(seg_images)}", 'INFO')
    log_message(f"Seg masks found: {len(seg_masks)}", 'INFO')
    if len(seg_images) > 0:
        binary_seg_model = train_binary_segmentation(seg_images, seg_masks, device)
    else:
        log_message("No fruit segmentation data found — skipping binary segmentation.", 'WARNING')

    # -------------------- STAGE 9: Fruit Multi-Class Segmentation --------------------
    log_message("STAGE 9: Fruit Multi-Class Segmentation (Part E)", 'INFO')
    if len(seg_images) > 0:
        multiclass_seg_model = train_multiclass_segmentation(
            seg_images,
            seg_masks,
            num_classes=len(fruit_class_names) + 1,
            device=device
        )
    else:
        log_message("No fruit segmentation data found — skipping multi-class segmentation.", 'WARNING')

    log_message("Segmentation run completed.", 'INFO')

    # -------------------- Final summary --------------------
    log_message("Run summary:", 'INFO')
    log_message(f"  Total images -> Food: {n_food}, Fruit: {n_fruit}", 'INFO')
    log_message(
        f"  Binary model checkpoint: {'exists' if os.path.exists(CONFIG['model_save_path']) else 'not saved'} ({CONFIG['model_save_path']})",
        'INFO')
    log_message(
        f"  Fruit model checkpoint: {'exists' if os.path.exists(CONFIG['fruit_model_path']) else 'not saved'} ({CONFIG['fruit_model_path']})",
        'INFO')
    log_message(
        f"  Siamese embedding: {'exists' if os.path.exists(CONFIG['siamese_path']) else 'not saved'} ({CONFIG['siamese_path']})",
        'INFO')
    log_message(
        f"  Support mean saved: {'exists' if os.path.exists(CONFIG['support_mean_path']) else 'not saved'} ({CONFIG['support_mean_path']})",
        'INFO')
    log_message(
        f"  Saved predictions: {os.path.join(CONFIG['output_dir'], 'predictions', 'validation_predictions_with_subclasses.txt')}",
        'INFO')

    return model, train_history, val_metrics


if __name__ == "__main__":
    model, train_history, val_metrics = main()
