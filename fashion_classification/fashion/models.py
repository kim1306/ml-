"""Model classes: MLPClassifier, VGG16Scratch, PretrainedVGG16."""
import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h)) #linear transformation from prev to size h
            layers.append(nn.BatchNorm1d(h)) # batch activation to have a mean of 0 and variance of 1
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers) #chains those componenets above into a single module

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten
        return self.net(x)


class VGG16Scratch(nn.Module):
    def __init__(self, num_classes, in_size=224, use_batchnorm=True, dropout=0.5):
        # in_size is the h and w of the input image that model expects,
        # in other words, it squashes the origianl size (300X432) to (224X224)
        super().__init__()
        if in_size % 32 != 0:
            raise ValueError("in_size must be divisible by 32 ") # 5 halvings

        def conv_block(in_c, out_c, n_convs):
            # Builds one VGG "block": n_convs stacked 3x3 convs
            # then a single 2x2 max-pool that halves H and W.
            block_layers = []
            for i in range(n_convs):
                c_in = in_c if i == 0 else out_c
                block_layers.append(nn.Conv2d(c_in, out_c, kernel_size=3, padding=1))
                if use_batchnorm:
                    block_layers.append(nn.BatchNorm2d(out_c))
                block_layers.append(nn.ReLU())
            block_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            return nn.Sequential(*block_layers)

        # standard practce: 3(RGB) -> 64 channels ->256 ->512 to represent diverse visual patterns
        self.features = nn.Sequential(
            conv_block(3,   64,  2),
            conv_block(64,  128, 2),
            conv_block(128, 256, 3),
            conv_block(256, 512, 3),
            conv_block(512, 512, 3),
        )

        feat_dim = in_size // 32 #h and w of the final featuremap
        flat_dim = 512 * feat_dim * feat_dim #final total numbers before sending it to FC

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming init for conv layers (this is designed to work with ReLU);
        # normal init for FC layers, it is standard practice for VGG.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: # if the conv layer has a bias, set all to 0
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # it initializes the weights using a mean =0, and std 0.01
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout=0.5):
        super().__init__()
        self.backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Keep the pretrained 4096-unit hidden layers
        # swap the two dropout [2] and [5]
        # layers (tunable) and replace the final layer for our classes[6].
        self.backbone.classifier[2] = nn.Dropout(p=dropout)
        self.backbone.classifier[5] = nn.Dropout(p=dropout)
        self.backbone.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.backbone(x)
