from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from models import Normal, VAE
from dataset import PHD08
from utils import options, CHOSUNG_LIST, JUNGSUNG_LIST, JONGSUNG_LIST
from torchsummaryX import summary


def cat_loss_function(context, label):
    cat_loss = F.binary_cross_entropy(context, label)
    return cat_loss


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


def test_model(mode, device, model, dataloader):
    running_loss = 0.0
    running_corrects = 0.0

    # Iterate over data.
    for n, (labels, inputs) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if mode == "Normal":
            # forward
            context = model(inputs)

        else:
            # forward
            context, context_latent, addition, addition_latent, mu, logvar, recon = model(
                inputs
            )

        cat_loss = cat_loss_function(context, labels)

        # accuracy
        acc = get_accuracy(context, labels)

        # statistic
        running_loss += cat_loss.item()
        running_corrects += acc

    dataset_len = len(dataloader)
    avg_loss = running_loss / dataset_len
    avg_acc = running_corrects / dataset_len

    print("Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


def main():
    device = torch.device("cuda")

    # Dataset
    test_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = PHD08(options["test_dir"], test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=options["batch_size"],
        shuffle=False,
        num_workers=options["workers"],
        pin_memory=True,
    )

    checkpoint_path = "%s.pth" % options["name"]

    # Model
    if options["mode"] == "Normal":
        model = Normal(options["hidden_dim"], options["context_dim"])
    else:
        model = VAE(
            options["hidden_dim"], options["context_dim"], options["addition_dim"]
        )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    summary(model, torch.zeros((1, 1, 28, 28)))
    model = model.to(device)

    # Train
    test_model(options["mode"], device, model, test_loader)


if __name__ == "__main__":
    main()
