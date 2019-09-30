import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from models import Normal, VAE
from dataset import PHD08
from utils import options, CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST


def cat_loss_function(context, label):
    cat_loss = F.binary_cross_entropy(context, label)
    return cat_loss


def latent_loss_function(mu, logvar):
    latent_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    )
    return latent_loss


def recon_loss_function(recon_x, x):
    recon_loss = F.binary_cross_entropy(recon_x, x)
    return recon_loss


def orth_loss_function(context_latent, addition_latent):
    orth_loss = torch.mean(
        torch.abs(F.cosine_similarity(context_latent, addition_latent, dim=1))
    )
    return orth_loss


def random_character(labels):
    first_dim, middle_dim, last_dim = (
        len(CHOSUNG_LIST),
        len(JUNGSUNG_LIST),
        len(JONGSUNG_LIST),
    )
    contexts = torch.zeros((len(labels), first_dim + middle_dim + last_dim))
    for i, label in enumerate(labels):
        first = label[:first_dim]
        middle = label[first_dim:-last_dim]
        last = label[-last_dim:]

        cases = np.random.choice(3, np.random.randint(3), replace=False)
        for case in cases:
            if case == 0:
                first = torch.eye(first_dim)[np.random.randint(first_dim)].to(
                    labels.device
                )
            elif case == 1:
                middle = torch.eye(middle_dim)[np.random.randint(middle_dim)].to(
                    labels.device
                )
            elif case == 2:
                last = torch.eye(last_dim)[np.random.randint(last_dim)].to(
                    labels.device
                )

        contexts[i] = torch.cat((first, middle, last))

    return contexts


def random_normal(n_samples, addition_dim):
    return torch.randn((n_samples, addition_dim))


def get_accuracy(context, labels):
    _, preds = torch.max(context[:, : len(CHOSUNG_LIST)], 1)
    first = torch.eye(len(CHOSUNG_LIST))[preds]

    _, preds = torch.max(context[:, len(CHOSUNG_LIST) : len(JONGSUNG_LIST)], 1)
    middle = torch.eye(len(JUNGSUNG_LIST))[preds]

    _, preds = torch.max(context[:, -len(JONGSUNG_LIST) :], 1)
    last = torch.eye(len(JONGSUNG_LIST))[preds]

    preds = torch.cat((first, middle, last), dim=1).to(context.device)
    preds = torch.all(torch.eq(preds, labels), dim=1)
    acc = torch.sum(preds, dtype=float) / len(preds)
    return acc


def train_model(
    mode,
    device,
    model,
    dataloaders,
    optimizer_I,
    optimizer_L,
    writer,
    cat_rate=1.0,
    recon_rate=1.0,
    latent_rate=1.0,
    lr_cat_rate=1.0,
    lr_latent_rate=1.0,
    checkpoint_path="best.pth",
    num_epochs=100,
):

    best_model_wts = {k: v.to("cpu") for k, v in model.state_dict().items()}
    best_model_wts = OrderedDict(best_model_wts)
    best_acc = 0
    best_epoch = 0

    display_num = 4
    display_iter = 100

    phase_iters = {"train": 0, "val": 0}

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for n, (labels, inputs) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                cat_loss = 0
                recon_loss = 0
                latent_loss = 0
                ir_loss = 0
                orth_loss = 0
                lr_cat_loss = 0
                lr_latent_loss = 0
                lr_loss = 0

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if mode == "Normal":
                        # forward
                        context = model(inputs)

                        cat_loss = cat_loss_function(context, labels)
                    else:
                        # IR
                        context, context_latent, addition, addition_latent, mu, logvar, recon = model(
                            inputs
                        )

                        # calculate losses
                        cat_loss = cat_loss_function(context, labels)
                        latent_loss = latent_loss_function(mu, logvar)
                        recon_loss = recon_loss_function(recon, inputs)
                        orth_loss = orth_loss_function(context_latent, addition_latent)

                    # calcualte forward loss
                    ir_loss = (
                        cat_rate * cat_loss
                        + recon_rate * recon_loss
                        + latent_rate * latent_loss
                    )

                    # optimize only if in training phase
                    if phase == "train":
                        # zero the parameter gradients
                        optimizer_I.zero_grad()

                        # backward
                        ir_loss.backward()

                        # update
                        optimizer_I.step()

                    if mode == "LR":
                        # random labels
                        random_labels = random_character(labels).to(device)

                        # random additions
                        random_additions = random_normal(
                            len(labels), model.addition_dim
                        ).to(device)

                        # LR
                        random_context, random_recon, random_mu, random_logvar, random_recon_addition = model.uturn(
                            random_labels, random_additions
                        )

                        # calculate losses
                        lr_cat_loss = cat_loss_function(random_context, random_labels)
                        lr_latent_loss = latent_loss_function(random_mu, random_logvar)

                        # calcualte uturn loss
                        lr_loss = (
                            lr_cat_rate * lr_cat_loss + lr_latent_rate * lr_latent_loss
                        )

                        # optimize only if in training phase
                        if phase == "train":
                            # zero the parameter gradients
                            optimizer_L.zero_grad()

                            # backward
                            lr_loss.backward()

                            # update
                            optimizer_L.step()

                # accuracy
                acc = get_accuracy(context, labels)

                # statistic
                running_loss += cat_loss.item()
                running_corrects += acc

                # logging
                writer.add_scalar(
                    "%s/IR_Loss" % phase, float(ir_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/Cat_Loss" % phase, float(cat_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/Recon_Loss" % phase, float(recon_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/Latent_Loss" % phase, float(latent_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/Orth_Loss" % phase, float(orth_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/LR_Loss" % phase, float(lr_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/LR_Cat_Loss" % phase, float(lr_cat_loss), phase_iters[phase] + n
                )
                writer.add_scalar(
                    "%s/LR_Latent_Loss" % phase,
                    float(lr_latent_loss),
                    phase_iters[phase] + n,
                )
                writer.add_scalar("%s/Accuracy" % phase, acc, phase_iters[phase] + n)
                if n % display_iter == 0:
                    if mode == "IR" or mode == "LR":
                        input_grid = make_grid(inputs[:display_num])
                        writer.add_image(
                            "%s/Input" % phase,
                            input_grid,
                            (phase_iters[phase] + n) // display_iter,
                        )
                        rcon_grid = make_grid(recon[:display_num])
                        writer.add_image(
                            "%s/Recon" % phase,
                            rcon_grid,
                            (phase_iters[phase] + n) // display_iter,
                        )
                        if mode == "LR":
                            random_grid = make_grid(random_recon[:display_num])
                            writer.add_image(
                                "%s/Random" % phase,
                                random_grid,
                                (phase_iters[phase] + n) // display_iter,
                            )

            dataset_len = len(dataloaders[phase])
            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects / dataset_len

            phase_iters[phase] += n

            print(
                "{} Epoch: {} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch, epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and best_acc < epoch_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = {k: v.to("cpu") for k, v in model.state_dict().items()}
                best_model_wts = OrderedDict(best_model_wts)

                # Save model
                torch.save(best_model_wts, checkpoint_path)

        print("Best val Accuracy: {:4f} in Epoch: {:.0f}".format(best_acc, best_epoch))


def main():
    torch.manual_seed(options["seed"])
    np.random.seed(options["seed"])
    device = torch.device("cuda")

    # Dataset
    train_transform = transforms.Compose([transforms.ToTensor()])
    val_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = PHD08(options["train_dir"], train_transform)
    val_dataset = PHD08(options["val_dir"], val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=options["batch_size"],
        shuffle=True,
        num_workers=options["workers"],
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=options["batch_size"],
        shuffle=True,
        num_workers=options["workers"],
        pin_memory=True,
    )

    dataloaders = {"train": train_loader, "val": val_loader}

    # Model
    if options["mode"] == "Normal":
        model = Normal(options["hidden_dim"], options["context_dim"])
    else:
        model = VAE(
            options["hidden_dim"], options["context_dim"], options["addition_dim"]
        )
    model = model.to(device)

    optimizer_I = optim.Adam(
        model.parameters(),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )

    optimizer_L = optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=options["learning_rate"],
        weight_decay=options["weight_decay"],
    )

    # Log
    logdir = "logs"
    explogdir = os.path.join(logdir, options["name"])
    if not os.path.exists(explogdir):
        os.makedirs(explogdir)
    writer = SummaryWriter(explogdir)

    checkpoint_path = "%s.pth" % options["name"]

    # Train
    train_model(
        options["mode"],
        device,
        model,
        dataloaders,
        optimizer_I,
        optimizer_L,
        writer,
        cat_rate=options["cat_rate"],
        recon_rate=options["recon_rate"],
        latent_rate=options["latent_rate"],
        lr_cat_rate=options["lr_cat_rate"],
        lr_latent_rate=options["lr_latent_rate"],
        checkpoint_path=checkpoint_path,
        num_epochs=options["num_epochs"],
    )


if __name__ == "__main__":
    main()
