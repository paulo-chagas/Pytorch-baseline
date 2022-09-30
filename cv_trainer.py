import os
from torchvision.transforms.transforms import CenterCrop, RandomCrop, Resize
import tqdm
import time
import json
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pprint import pprint
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, f1_score, recall_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

from customDataset import GlomDataset
from models import get_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class CVTrainer:
    def __init__(self, config):
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cv_acc = []
        self.cv_precision = []
        self.cv_recall = []
        self.cv_f_score = []
        self.cv_cm = np.zeros(
            (self.config.data.num_classes, self.config.data.num_classes))
        self.best_epochs = []
        self.lr = []

    def train(self):
        """ Train the model """
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        num_workers = 4  # 4 times the number of GPUs

        history = History(None)

        for i in range(1, self.config.trainer.K + 1):
            print(
                '===========================================================')
            print('Fold {} training'.format(i))

            # initializing model
            model = get_model(self.config.model.name,
                              self.config.data.num_classes,
                              self.config.model.freeze_backbone,
                              self.config.model.dp_rate)
            model = model.to(device)
            print("Model {} loaded on device {}".format(
                self.config.model.name, device))
            # setting optmizer and lr scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.model.learning_rate,
                weight_decay=5e-4)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1)

            base_path = os.path.join(self.config.exp.dir, "Fold" + str(i))
            train_loader, val_loader = self.get_cv_dataloaders(
                base_path, num_workers)

            start = datetime.datetime.now()
            best_f1 = -1
            best_epoch = 0

            # store lr
            self.lr = []
            # reset history
            history.reset(base_path)

            if torch.cuda.device_count() >= 1:
                scaler = torch.cuda.amp.GradScaler()  # MIXED PRECISION
            else:
                scaler = None

            for epoch in range(self.config.trainer.num_epochs):
                if self.config.trainer.save_lr and i == 1:
                    self.lr.append(get_lr(optimizer))
                # train for one epoch
                t_loss, t_fscore = self.train_epoch(
                    i, train_loader, model, optimizer, epoch, device, scaler)
                lr_scheduler.step()

                # compute metrics for the validation set
                print('Evaluating on validation set')
                v_loss, acc, f1, f1_avg, precision, recall, _, _, _, _ = self.validate(val_loader, model, epoch, device)

                # update history
                history.update(t_loss, t_fscore, v_loss, f1_avg)

                # get best model considering val f1 score
                is_best = f1 > best_f1
                best_f1 = max(f1, best_f1)
                if is_best:  # save checkpoint
                    best_epoch = epoch
                    self.save_states(filename=os.path.join(base_path, 'best_model.pth'),
                                     epoch=epoch,
                                     model=model,
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler)
            end = datetime.datetime.now()
            self.best_epochs.append(best_epoch)
            # save history
            history.save()

            # plot lr to check scheduler
            if self.config.trainer.save_lr and i == 1:
                plt.figure()
                plt.plot(self.lr, linestyle='-', marker='o', color='b')
                plt.xticks(np.arange(0, len(self.lr)+30, 30))
                plt.savefig(os.path.join(base_path, 'LR_plot.png'))

            # load best model after each K-Fold
            checkpoint = self.load_states(
                filename=os.path.join(base_path, 'best_model.pth'))
            model.load_state_dict(checkpoint['model'])
            print('\nBest model - epoch {}'.format(checkpoint['epoch']))

            _, acc, f1, _, precision, recall, cm, report, all_targets, all_preds = self.validate(val_loader, model, epoch, device)

            self.cv_acc.append(acc)
            self.cv_precision.append(precision)
            self.cv_recall.append(recall)
            self.cv_f_score.append(f1)
            self.cv_cm += cm

            out = 'Start: {}\nEnd: {}\nBest epoch: {}\nReport:{}\n\nConfusion matrix:\n{}\n\n'.format(
                start, end, str(checkpoint['epoch']), str(report), str(cm))
            print(out)
            # save output
            with open(os.path.join(base_path, "out.txt"), "w") as text_file:
                text_file.write(out)
            save_predictions(all_targets, all_preds, base_path)

        print('\nK-Fold Cross-validation complete :D\n')

        print(json.dumps(self.config.toDict(), indent=2, default=str))
        save_average_results(self.config.exp.dir, self.best_epochs, self.cv_cm,
                             self.cv_acc, self.cv_f_score, self.cv_precision, self.cv_recall)

    def train_epoch(self, fold, train_loader, model, optimizer, epoch, device, scaler):
        """ One epoch training """
        model.train()

        losses = AverageMeter()
        f1 = AverageMeter()
        batch_time = AverageMeter()

        batches = tqdm.tqdm(train_loader)
        end = time.time()
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                # Casts operations to mixed precision
                with torch.cuda.amp.autocast():
                    y_pred = model(x)
                    train_loss = self.criterion(y_pred, y)
            else:
                y_pred = model(x)
                train_loss = self.criterion(y_pred, y)

            # calculate f1-score
            f1_batch = f1_score(y.cpu().numpy(),
                                torch.max(F.softmax(y_pred, dim=1), dim=1)[1].cpu().numpy(),
                                average="macro", zero_division=0)

            # sum epoch loss
            losses.update(train_loss.item(), x.size(0))
            f1.update(f1_batch, x.size(0))

            if scaler is not None:
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
            else:
                train_loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print current metrics
            batches.set_description('Fold {fold} '
                                    'Epoch {epoch} '
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                                    'loss: {loss.val:.4f} ({loss.avg:.4f}) '
                                    'f1: {f1.val:.4f} ({f1.avg:.4f}) '.format(fold=fold, epoch=epoch, batch_time=batch_time, loss=losses, f1=f1))
        return losses.avg, f1.avg

    def validate(self, val_loader, model, epoch, device):
        model.eval()

        losses = AverageMeter()
        f1 = AverageMeter()
        all_targets = []
        all_preds = []
        all_confidences = []

        batches = tqdm.tqdm(val_loader)

        with torch.no_grad():
            for x, y in batches:
                x, y = x.to(device), y.to(device)

                y_pred = model(x)
                loss = self.criterion(y_pred, y)

                confidences = torch.max(F.softmax(y_pred, dim=1), dim=1)[
                    0].cpu().numpy()
                y_pred = torch.max(F.softmax(y_pred, dim=1), dim=1)[
                    1].cpu().numpy()

                all_targets.extend(y.cpu().numpy())
                all_preds.extend(y_pred)
                all_confidences.extend(confidences)

                # calculate f1-score
                f1_batch = f1_score(y.cpu().numpy(), y_pred,
                                    average="macro", zero_division=0)

                # sum epoch loss
                losses.update(loss.item(), x.size(0))
                f1.update(f1_batch, x.size(0))

        acc_loader = accuracy_score(all_targets, all_preds)
        f1_loader = f1_score(all_targets, all_preds,
                             average="macro", zero_division=0)
        precision_loader = precision_score(
            all_targets, all_preds, average="macro", zero_division=0)
        recall_loader = recall_score(
            all_targets, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, zero_division=0)

        print(' * Val -> acc {:.4f} f1 {:.4f} precision {:.4f} recall {:.4f}\n'
              .format(acc_loader, f1_loader, precision_loader, recall_loader))

        return losses.avg, acc_loader, f1_loader, f1.avg, precision_loader, recall_loader, cm, report, all_targets, all_preds

    def save_states(self, filename, epoch, model, optimizer, lr_scheduler):
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()}
        torch.save(checkpoint, filename)

    def load_states(self, filename):
        checkpoint = torch.load(filename)
        return checkpoint

    def get_cv_dataloaders(self, base_path, num_workers):
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(self.config.dataloader.image_size),
                A.RandomCrop(height=self.config.dataloader.image_size,
                             width=self.config.dataloader.image_size),
                A.OneOf([
                    A.RandomContrast(),
                    A.RandomGamma(),
                    A.RandomBrightness(),
                ], p=0.3),
                A.OneOf([
                    A.GaussNoise(0.002, p=.5),
                    A.IAAAffine(p=.5),
                ], p=.1),
                # Additional position augmentations
                A.RandomRotate90(p=.5),
                A.HorizontalFlip(p=.5),
                A.VerticalFlip(p=.5),
                A.Cutout(num_holes=10, fill_value=255,
                         max_h_size=int(.1 * \
                                        self.config.dataloader.image_size),
                         max_w_size=int(.1 * \
                                        self.config.dataloader.image_size),
                         p=.1),
            ]
        )

        val_transform = A.Compose(
            [
                A.SmallestMaxSize(self.config.dataloader.image_size),
                #A.RandomCrop(height=self.config.dataloader.image_size, width=self.config.dataloader.image_size),
                A.CenterCrop(height=self.config.dataloader.image_size,
                             width=self.config.dataloader.image_size),
            ]
        )

        train_dataset = GlomDataset(csv_file=os.path.join(base_path, 'train.csv'),
                                    root_dir=self.config.data.path,
                                    transform=train_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.config.trainer.batch_size,
                                  shuffle=True, num_workers=num_workers)

        val_dataset = GlomDataset(csv_file=os.path.join(base_path, 'val.csv'),
                                  root_dir=self.config.data.path,
                                  transform=val_transform)

        val_loader = DataLoader(val_dataset, batch_size=self.config.trainer.batch_size,
                                shuffle=True, num_workers=num_workers)

        return train_loader, val_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class History(object):
    """Stores train/val metrics"""

    def __init__(self, path):
        self.path = path
        self.reset(self.path)

    def reset(self, path):
        self.train_loss = []
        self.train_f1 = []
        self.val_loss = []
        self.val_f1 = []
        self.path = path

    def update(self, t_loss, t_f1, v_loss, v_f1):
        self.train_loss.append(t_loss)
        self.train_f1.append(t_f1)
        self.val_loss.append(v_loss)
        self.val_f1.append(v_f1)

    def save(self):
        data = {'train_loss': self.train_loss,
                'train_f1': self.train_f1,
                'val_loss': self.val_loss,
                'val_f1': self.val_f1}

        df = pd.DataFrame(data, columns=data.keys())
        df.to_csv(os.path.join(self.path, 'history.csv'), sep=';', index=False)

        plt.figure()
        plt.plot(self.train_loss, label='train loss')
        plt.plot(self.val_loss, label='val loss')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'losses_plot.png'))

        plt.figure()
        plt.plot(self.train_f1, label='train f1')
        plt.plot(self.val_f1, label='val f1')
        plt.legend()
        plt.savefig(os.path.join(self.path, 'f1_plot.png'))


def save_predictions(y_real, y_pred, path):
    data = {'y_real': y_real, 'y_pred': y_pred}
    df = pd.DataFrame(data, columns=data.keys())

    df.to_csv(os.path.join(path, 'predictions.csv'), sep=';', index=False)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_average_results(path, epochs, sum_confusion_matrices, accs, fscores, precisions, recalls):
    with open(os.path.join(path, "average.txt"), "w") as text_file:
        text_file.write("Confusion Matrix Sum\n{}".format(
            sum_confusion_matrices))

        text_file.write(
            "\n=========================================\n\nbest_epochs: {}".format(epochs))

        text_file.write(
            "\n=========================================\n\naccuracies: {}".format(accs))
        text_file.write(
            "\nMacro average accuracy: {:.3f} (+/- {:.3f})".format(np.mean(accs), np.std(accs)))

        text_file.write(
            "\n=========================================\n\nf-scores: {}".format(fscores))
        text_file.write(
            "\nMacro average f-score: {:.3f} (+/- {:.3f})".format(np.mean(fscores), np.std(fscores)))

        text_file.write(
            "\n=========================================\n\nprecisions: {}".format(precisions))
        text_file.write("\nMacro average precision: {:.3f} (+/- {:.3f})".format(
            np.mean(precisions), np.std(precisions)))

        text_file.write(
            "\n=========================================\n\nrecalls: {}".format(recalls))
        text_file.write(
            "\nMacro average recall: {:.3f} (+/- {:.3f})".format(np.mean(recalls), np.std(recalls)))

        text_file.write("\n\n\nBest f-score: {} (fold {})".format(
            np.argmax(fscores), str(np.argmax(fscores) + 1)))
