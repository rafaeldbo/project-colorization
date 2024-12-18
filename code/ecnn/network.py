from torch import nn, Tensor, cat
from torch.nn.functional import relu


class NetworkBasic(nn.Module):
    def __init__(self, number_categories: int = 8, emb_size: int = 10) -> None:
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()

        # Embedding.
        self.emb_size = emb_size
        self.embb = nn.Embedding(number_categories, emb_size)

        # Encoder.
        self.conv1 = nn.Conv2d(1 + self.emb_size, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)

        # Decoder.
        self.tconv1 = nn.ConvTranspose2d(64 + self.emb_size, 32, kernel_size=4, stride=2, padding=1)
        self.tconv1_bn = nn.BatchNorm2d(32)
        self.tconv2 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)

    def forward(
        self,
        gray: Tensor,
        category: int,
    ) -> Tensor:

        # First Embeeding.
        embb_category_in = (
            self.embb(category)
            .view(-1, self.emb_size, 1, 1)
            .repeat(1, 1, gray.shape[2], gray.shape[3])
        )
        gray_embb_in = cat((gray, embb_category_in), 1)

        # Encoder.
        gray_conv1 = relu(self.conv1_bn(self.conv1(gray_embb_in)))
        gray_conv2 = relu(self.conv2_bn(self.conv2(gray_conv1)))

        # Second Embeeding.
        embb_category_out = (
            self.embb(category)
            .view(-1, self.emb_size, 1, 1)
            .repeat(1, 1, gray_conv2.shape[2], gray_conv2.shape[3])
        )
        gray_embb_out = cat((gray_conv2, embb_category_out), 1)

        # Decoder.
        gray_tconv1 = relu(self.tconv1_bn(self.tconv1(gray_embb_out)))
        gray_tconv2 = relu(self.tconv2(gray_tconv1))
        return gray_tconv2


class NetworkAdvanced(nn.Module):
    def __init__(
        self,
        number_categories: int = 8,
        emb_size: int = 10,
    ) -> None:
        """
        Initializes each part of the convolutional neural network.
        """
        super().__init__()

        # Embeddings
        self.emb_size = emb_size
        self.embd = nn.Embedding(number_categories, emb_size)

        # Encoder
        self.conv1 = nn.Conv2d(1 + self.emb_size, 32, kernel_size=4, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(128)
        
        # Transition
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.dilat1 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.dilat1_bn = nn.BatchNorm2d(256)
        self.dilat2 = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3, dilation=2)
        self.dilat2_bn = nn.BatchNorm2d(256)

        # Decoder
        self.tconv3 = nn.ConvTranspose2d(256 + self.emb_size, 128, kernel_size=4, stride=2, padding=1 )
        self.tconv3_bn = nn.BatchNorm2d(128)
        self.tconv2 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
        self.tconv2_bn = nn.BatchNorm2d(64)
        self.tconv1 = nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1)
        self.tconv1_bn = nn.BatchNorm2d(32)
        self.tconv0 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

        self.tconv_out = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        gray: Tensor,
        category: int,
    ) -> Tensor:

        # First Embeeding
        embd_decoder = self.embd(category)\
            .view(-1, self.emb_size, 1, 1)\
            .repeat(1, 1, gray.shape[2], gray.shape[3])
    
        gray_embd_encoder = cat((gray, embd_decoder), 1)

        # Encoder
        gray_conv1 = relu(self.conv1_bn(self.conv1(gray_embd_encoder)))
        gray_conv2 = relu(self.conv2_bn(self.conv2(gray_conv1)))
        gray_conv3 = relu(self.conv3_bn(self.conv3(gray_conv2)))

        # Transition
        gray_conv4 = relu(self.conv4_bn(self.conv4(gray_conv3)))
        gray_dilat1 = relu(self.dilat1_bn(self.dilat1(gray_conv4)))
        gray_dilat2 = relu(self.dilat2_bn(self.dilat2(gray_dilat1)))

        # Second Embeeding
        embd_decoder = self.embd(category)\
            .view(-1, self.emb_size, 1, 1)\
            .repeat(1, 1, gray_dilat2.shape[2], gray_dilat2.shape[3])
    
        gray_embd_decoder = cat((gray_dilat2, embd_decoder), 1)

        # Decoder
        gray_tconv3 = relu(self.tconv3_bn(self.tconv3(gray_embd_decoder)))
        gray_tconv3 = cat((gray_tconv3, gray_conv3), 1)
        gray_tconv2 = relu(self.tconv2_bn(self.tconv2(gray_tconv3)))
        gray_tconv2 = cat((gray_tconv2, gray_conv2), 1)
        gray_tconv1 = relu(self.tconv1_bn(self.tconv1(gray_tconv2)))
        gray_tconv1 = cat((gray_tconv1, gray_conv1), 1)
        gray_tconv0 = relu(self.tconv0(gray_tconv1))
        gray_tconv0 = cat((gray_tconv0, gray), 1)

        output = self.tconv_out(gray_tconv0)
        return output
