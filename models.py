import torch
from torch import nn
from torch.nn import functional as F


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.BatchNorm1d(128 * (self.input_height // 4) * (self.input_width // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_height // 4), (self.input_width // 4))
        x = self.deconv(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.input_height = 28
        self.input_width = 28

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_height // 4) * (self.input_width // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * (self.input_height // 4) * (self.input_width // 4))
        x = self.fc(x)
        return x


class Embedder(nn.Module):
    def __init__(self, hidden_dim, addition_dim):
        super(Embedder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, addition_dim)
        self.fc_logvar = nn.Linear(hidden_dim, addition_dim)

    def forward(self, x):
        latent = self.fc(x)
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)
        return latent, mu, logvar


class Classifier(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )
        self.fc = nn.Linear(hidden_dim, context_dim)

    def forward(self, x):
        latent = self.fc(x)
        output = F.softmax(self.fc(latent), dim=1)
        return latent, output


class Normal(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(Normal, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.encoder = Encoder()
        self.classifier = Classifier(self.hidden_dim, self.context_dim)

        initialize_weights(self)

    def forward(self, x):
        # feature
        feature = self.encoder(x)

        # class
        _, context = self.classifier(feature)

        return context


class VAE(nn.Module):
    def __init__(self, hidden_dim, context_dim, addition_dim):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.addition_dim = addition_dim

        self.encoder = Encoder()
        self.classifier = Classifier(self.hidden_dim, self.context_dim)
        self.embedder = Embedder(self.hidden_dim, self.addition_dim)
        self.decoder = Decoder(self.context_dim + self.addition_dim, self.hidden_dim)

        initialize_weights(self)

    def forward(self, x):
        # feature
        feature = self.encoder(x)

        # class
        context_latent, context = self.classifier(feature)

        # addition
        addition_latent, mu, logvar = self.embedder(feature)
        addition = self.reparameterize(mu, logvar)

        # concatenate
        concat = torch.cat((context, addition), dim=1).to(context.device)

        # reconstruct
        recon = self.decoder(concat)

        return (context, context_latent, addition, addition_latent, mu, logvar, recon)

    def uturn(self, random_label, random_addition):
        # random concatenate
        random_concat = torch.cat((random_label, random_addition), dim=1).to(
            random_label.device
        )

        # random reconstruct
        random_recon = self.decoder(random_concat)

        # random feature
        random_feature = self.encoder(random_recon)

        # random class
        random_context_latent, random_context = self.classifier(random_feature)

        # random recon addition
        random_addition_latent, random_mu, random_logvar = self.embedder(random_feature)
        random_recon_addition = self.reparameterize(random_mu, random_logvar)

        return random_context, random_recon, random_mu, random_logvar, random_recon_addition

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(logvar.device)
        eps = torch.randn_like(std).to(logvar.device)
        return mu + eps * std
