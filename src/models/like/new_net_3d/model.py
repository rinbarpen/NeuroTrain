import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

from ...conv.DWConv import get_dwconv_layer3d, get_dwconv_layer2d

# Helper function for dynamic padding to match sizes, common in U-Nets
def _calculate_padding(input_tensor, target_tensor):
    """
    Calculates padding for input_tensor to match target_tensor's spatial dimensions.
    Assumes input_tensor is smaller than target_tensor.
    Input tensors are expected to be (B, C, D, H, W).
    """
    input_size = input_tensor.size()
    target_size = target_tensor.size()

    diff_d = target_size[2] - input_size[2]
    diff_h = target_size[3] - input_size[3]
    diff_w = target_size[4] - input_size[4]

    # Calculate padding for each dimension
    # pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back
    padding = [diff_w // 2, diff_w - diff_w // 2,
               diff_h // 2, diff_h - diff_h // 2,
               diff_d // 2, diff_d - diff_d // 2]
    return padding

# --- Basic Building Blocks ---

class BasicDoubleConv(nn.Module):
    """
    Standard DoubleConv block: Conv3d -> BatchNorm3d -> ReLU -> Conv3d -> BatchNorm3d -> ReLU
    Used for most U-Net encoder/decoder stages.
    """
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1, activation=nn.ReLU):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_channels),
            activation(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm3d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # For 3D, kernel_size applies to H and W. Depth (D) is handled implicitly by pooling along it.
        # This is a common 2D CBAM spatial attention generalized to 3D.
        # It aggregates features across channels, then applies a convolution.
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size//2, kernel_size//2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat)) * x

class DepthAttention3D(nn.Module):
    """
    Attention mechanism focusing on the depth dimension.
    Pools over spatial dimensions (H, W), then applies a 1D convolution along depth.
    """
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1)) # Pool to (D, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool3d((None, 1, 1)) # Pool to (D, 1, 1)
        # Apply a 1D conv (1x1xk) along the depth dimension.
        # Input: B, C, D, 1, 1 -> Output: B, 1, D, 1, 1
        self.conv_depth = nn.Conv3d(in_channels, 1, kernel_size=(kernel_size, 1, 1),
                                     padding=(kernel_size//2, 0, 0), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: B, C, D, H, W
        avg_out = self.avg_pool(x) # B, C, D, 1, 1
        max_out = self.max_pool(x) # B, C, D, 1, 1
        combined_pooled = avg_out + max_out
        
        # Apply convolution along the depth dimension
        # The convolution combines channel information for each depth slice
        # and outputs a single value per depth slice.
        depth_attention_map = self.sigmoid(self.conv_depth(combined_pooled)) # B, 1, D, 1, 1
        return depth_attention_map * x # Broadcast attention map across H, W and channels


class CombinedAttention3D(nn.Module):
    """ Combines Depth, Channel, and Spatial Attention as DA -> CA -> SA. """
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7, depth_kernel_size=3):
        super().__init__()
        self.da = DepthAttention3D(in_channels, depth_kernel_size)
        self.ca = ChannelAttention3D(in_channels, reduction_ratio)
        self.sa = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x):
        x = self.da(x)
        x = self.ca(x)
        x = self.sa(x)
        return x

class DWConv3d(nn.Module):
    """ Depthwise Convolution 3D with BN and ReLU. """
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AsymmetricConv3d(nn.Module):
    """
    Approximation of AsymmetricConv3d for 3D data.
    Uses parallel 1x1xK, 1xKx1, Kx1x1 convolutions and sums their outputs.
    This interprets ACConv3d as capturing features along each axis independently.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 1x1xK convolution along depth axis
        self.conv_d = nn.Conv3d(in_channels, out_channels, (1, 1, kernel_size), padding=(0, 0, padding), bias=False)
        # 1xKx1 convolution along height axis
        self.conv_h = nn.Conv3d(in_channels, out_channels, (1, kernel_size, 1), padding=(0, padding, 0), bias=False)
        # Kx1x1 convolution along width axis
        self.conv_w = nn.Conv3d(in_channels, out_channels, (kernel_size, 1, 1), padding=(padding, 0, 0), bias=False)
        
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_d = self.conv_d(x)
        x_h = self.conv_h(x)
        x_w = self.conv_w(x)
        return self.relu(self.bn(x_d + x_h + x_w)) # Summing outputs from parallel branches

class InceptionLightBlock(nn.Module):
    """
    Represents an 'InceptionConv3d but Light' block as described for the Backbone.
    It comprises parallel standard, depthwise, and asymmetric convolutions,
    followed by a 1x1 convolution and then combined attention (DA+CBAM).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Distribute channels evenly among branches (or adjust as needed)
        mid_channels = out_channels // 3
        if mid_channels == 0: mid_channels = 1 # Ensure at least 1 channel

        # Branch 1: Standard Conv3d
        self.branch_std = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Depthwise Conv3d (followed by 1x1 pointwise conv)
        self.branch_dw = nn.Sequential(
            DWConv3d(in_channels, kernel_size=3, padding=1), # DWConv3d doesn't change channel count
            nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False), # Pointwise conv to adjust channels
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Branch 3: Asymmetric Conv3d
        self.branch_ac = AsymmetricConv3d(in_channels, mid_channels, kernel_size=3, padding=1)

        # Concatenate outputs of the branches and then map to final out_channels
        self.conv1x1_after_branches = nn.Sequential(
            nn.Conv3d(mid_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Attention block (DA+CBAM)
        self.attention = CombinedAttention3D(out_channels)

    def forward(self, x):
        x_std = self.branch_std(x)
        x_dw = self.branch_dw(x)
        x_ac = self.branch_ac(x)

        x_combined = torch.cat([x_std, x_dw, x_ac], dim=1) # Concatenate along channel dimension
        x_mapped = self.conv1x1_after_branches(x_combined)
        return x_mapped

class BackboneDoubleConv(nn.Module):
    """
    Special DoubleConv for the Backbone, using two InceptionLightBlocks.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block1 = InceptionLightBlock(in_channels, out_channels)
        self.block2 = InceptionLightBlock(out_channels, out_channels) # Second block processes output of first

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class DownBlock(nn.Module):
    """
    Encoder Downsampling Block: BasicDoubleConv -> MaxPool3d.
    Returns the feature map before pooling (for skip connection) and the pooled feature map.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = BasicDoubleConv(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2) # Standard downsampling

    def forward(self, x):
        features = self.double_conv(x)
        downsampled_features = self.downsample(features)
        return features, downsampled_features

class NeckBlock(nn.Module):
    """
    Bottleneck layer of the U-Net. Just a BasicDoubleConv.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = BasicDoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.double_conv(x)

class UpBlockDecoder(nn.Module):
    """
    Decoder Upsampling Block: ConvTranspose3d -> Add Skip Connection -> BasicDoubleConv.
    The addition of skip connection features is explicitly shown in the diagram.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ConvTranspose3d for upsampling and reducing channels by half
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.bn_up = nn.BatchNorm3d(out_channels) # Batch Norm after transpose conv
        self.relu_up = nn.ReLU(inplace=True)

        # After upsampling and adding skip connection, process with a DoubleConv
        self.double_conv = BasicDoubleConv(out_channels, out_channels) # Input channels are out_channels after addition

    def forward(self, x, skip_features):
        x = self.relu_up(self.bn_up(self.up(x))) # Upsample x

        # Handle potential size mismatches due to padding or odd dimensions
        # Crop skip_features if x is slightly larger, or pad x if smaller.
        # Given how ConvTranspose3d works, it might be slightly off. Padding is safer.
        x = F.pad(x, _calculate_padding(x, skip_features))

        x = x + skip_features # Element-wise addition as per diagram
        x = self.double_conv(x)
        return x

class MaskEncoder(nn.Module):
    """
    Separate Mask Encoder branch as depicted.
    Consists of two Conv3d blocks (k=2, s=2), each followed by BN and LeakyReLU.
    Then applies CombinedAttention3D, and a final 1x1 Conv to match target output channels.
    Initializes weights with kaiming_normal_ as specified.
    """
    def __init__(self, in_channels, final_out_channels):
        super().__init__()
        # First Conv3d: in_channels -> 32, k=2, s=2
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=2, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu1 = nn.LeakyReLU(inplace=True)

        # Second Conv3d: 32 -> 64, k=2, s=2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=2, stride=2, bias=False)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu2 = nn.LeakyReLU(inplace=True)

        # Attention block (DA+CBAM) applied after the convolutions
        self.attention = CombinedAttention3D(64)

        # Final 1x1 Conv to match the output channels of the Neck block for addition
        self.final_conv = nn.Conv3d(64, final_out_channels, kernel_size=1, bias=False)
        self.bn_final = nn.BatchNorm3d(final_out_channels)
        self.relu_final = nn.LeakyReLU(inplace=True) # Assuming LeakyReLU continues

        self._init_weights()

    def _init_weights(self):
        # Apply kaiming_normal_ initialization with leaky_relu nonlinearity
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.attention(x)
        x = self.relu_final(self.bn_final(self.final_conv(x)))
        return x

class CrossAttention(nn.Module):
    """
    Multi-head Cross-Attention mechanism for 3D feature maps.
    Flattens D*H*W, applies attention, then reshapes back.
    Assumes query and key_value features have the same number of channels (dim).
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias) # For Key and Value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_feature, key_value_feature):
        # Query feature (e.g., from a higher level, to be enhanced by lower level)
        # Key/Value feature (e.g., from a lower level, providing context)
        B, C, Dq, Hq, Wq = query_feature.shape
        _, _, Dkv, Hkv, Wkv = key_value_feature.shape

        # Flatten spatial dimensions for attention
        query = query_feature.flatten(2).transpose(1, 2) # B, Nq, C (Nq = Dq*Hq*Wq)
        kv = key_value_feature.flatten(2).transpose(1, 2) # B, Nkv, C (Nkv = Dkv*Hkv*Wkv)

        # Linear projections for Q, K, V
        q = self.q_proj(query).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, num_heads, Nq, head_dim
        k, v = self.kv_proj(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2, B, num_heads, Nkv, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale # B, num_heads, Nq, Nkv
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C) # B, Nq, C
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to 3D feature map, matching query's original dimensions
        x = x.transpose(1, 2).reshape(B, C, Dq, Hq, Wq)
        return x

# --- Full Model Integration ---

class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, mask_in_channels=1, base_features=32):
        super().__init__()

        # Define feature channel sizes for different levels
        # Based on typical U-Net progression (e.g., 32, 64, 128, 256)
        # The diagram doesn't specify exact channels, so we use a base and multiply.
        features = [base_features * (2**i) for i in range(4)] # [32, 64, 128, 256]

        # --- Encoder Path ---
        # The very first block from 'input' to '3D Encoder' (Level 0) uses the special BackboneDoubleConv.
        self.enc0 = BackboneDoubleConv(in_channels, features[0])

        # Downsampling blocks for Encoder (Level 1, 2, 3)
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])

        # --- Neck (Bottleneck) ---
        # The neck block processes the lowest resolution features.
        self.neck = NeckBlock(features[3], features[3] * 2) # Often doubles channels at bottleneck

        # --- Mask Encoder Branch ---
        # Takes a separate 'mask_input'. Its output is added to the neck output.
        # Its output channels must match neck_out channels for addition.
        self.mask_encoder = MaskEncoder(mask_in_channels, features[3] * 2) # e.g., mask_in_channels=1, output=512

        # --- Decoder Path ---
        # Upsampling blocks for Decoder (Level 3, 2, 1)
        # Each UpBlockDecoder takes previous level's output and a skip connection.
        self.up1 = UpBlockDecoder(features[3] * 2, features[3]) # From neck_out, skip from down3_features
        self.up2 = UpBlockDecoder(features[3], features[2])     # From up1_out, skip from down2_features
        self.up3 = UpBlockDecoder(features[2], features[1])     # From up2_out, skip from down1_features

        # Final Convolution for the main segmentation output
        self.main_output_conv = nn.Conv3d(features[1], num_classes, kernel_size=1)

        # --- Top-Right Processing Branch (Auxiliary Output) ---
        # This branch takes inputs from different encoder levels and processes them via CrossAttention.
        # To make CrossAttention work, all inputs must have the same channel dimension (dim).
        # We will project the higher-level features to match `features[0]` for consistency.
        # Inputs are from enc0_out (L0), skip1_features (L1), skip2_features (L2).
        self.proj_l1_to_ca_dim = nn.Conv3d(features[1], features[0], kernel_size=1, bias=False)
        self.proj_l2_to_ca_dim = nn.Conv3d(features[2], features[0], kernel_size=1, bias=False)

        # First CrossAttention block: Query from L1 features, Key/Value from L0 features.
        self.ca1 = CrossAttention(dim=features[0])
        self.double_conv_ca1 = BasicDoubleConv(features[0], features[0])

        # Second CrossAttention block: Query from L2 features, Key/Value from output of double_conv_ca1.
        self.ca2 = CrossAttention(dim=features[0])
        self.double_conv_ca2 = BasicDoubleConv(features[0], features[0])

        # Final convolution for this auxiliary output branch
        self.top_branch_output_conv = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x, mask_input):
        # --- Encoder Forward Pass ---
        enc0_out = self.enc0(x) # Input processed by BackboneDoubleConv

        skip1_features, down1_out = self.down1(enc0_out)   # L0 -> L1
        skip2_features, down2_out = self.down2(down1_out)  # L1 -> L2
        skip3_features, down3_out = self.down3(down2_out)  # L2 -> L3

        # --- Neck Forward Pass ---
        neck_out = self.neck(down3_out) # Lowest resolution features

        # --- Mask Encoder Branch Forward Pass ---
        mask_enc_out = self.mask_encoder(mask_input)
        # Ensure mask_enc_out matches neck_out spatial dimensions for addition
        mask_enc_out = F.pad(mask_enc_out, _calculate_padding(mask_enc_out, neck_out))
        
        neck_out_with_mask = neck_out + mask_enc_out # Add mask features to bottleneck features

        # --- Decoder Forward Pass ---
        up1_out = self.up1(neck_out_with_mask, skip3_features) # Upsample from neck, add L3 skip
        up2_out = self.up2(up1_out, skip2_features)           # Upsample from up1, add L2 skip
        up3_out = self.up3(up2_out, skip1_features)           # Upsample from up2, add L1 skip

        main_output = self.main_output_conv(up3_out)

        # --- Top-Right Processing Branch Forward Pass ---
        # Prepare inputs for CrossAttention by projecting to common channel dimension (features[0])
        ca_l0_in = enc0_out
        ca_l1_in = self.proj_l1_to_ca_dim(skip1_features)
        ca_l2_in = self.proj_l2_to_ca_dim(skip2_features)

        # First CrossAttention
        # Query: ca_l1_in (L1 features), Key/Value: ca_l0_in (L0 features)
        ca1_output = self.ca1(ca_l1_in, ca_l0_in)
        double_conv_ca1_output = self.double_conv_ca1(ca1_output)

        # Second CrossAttention
        # Query: ca_l2_in (L2 features), Key/Value: double_conv_ca1_output
        ca2_output = self.ca2(ca_l2_in, double_conv_ca1_output)
        double_conv_ca2_output = self.double_conv_ca2(ca2_output)

        top_branch_output = self.top_branch_output_conv(double_conv_ca2_output)

        return main_output, top_branch_output

if __name__ == '__main__':
    # Example Usage:
    # Input tensor: (Batch_size, Channels, Depth, Height, Width)
    # Mask input tensor: (Batch_size, Channels, Depth, Height, Width)

    # Let's assume input image size 64x64x64 for simplicity
    batch_size = 1
    input_channels = 1
    output_classes = 1 # e.g., for binary segmentation
    mask_input_channels = 1 # Assuming mask is also a single channel image

    # Create dummy input data
    input_data = torch.randn(batch_size, input_channels, 64, 64, 64)
    # For MaskEncoder, let's assume the mask input is a downsampled version
    # The MaskEncoder performs 2x downsampling twice, so 64 -> 32 -> 16
    # For addition with neck, neck is (64/8) = 8. So mask input should be large enough to result in 8x8x8 or similar
    # Mask encoder has 2x2 stride twice, so it reduces by factor of 4.
    # If neck_out is 8x8x8, then mask_enc_out before padding should be 8x8x8.
    # Input to mask encoder should be 8 * 4 = 32.
    mask_input_data = torch.randn(batch_size, mask_input_channels, 32, 32, 32)


    model = Model(in_channels=input_channels, num_classes=output_classes,
                  mask_in_channels=mask_input_channels, base_features=32)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()
        input_data = input_data.cuda()
        mask_input_data = mask_input_data.cuda()

    print(f"Input data shape: {input_data.shape}")
    print(f"Mask input data shape: {mask_input_data.shape}")

    main_output, top_branch_output = model(input_data, mask_input_data)

    print(f"\nMain Output shape: {main_output.shape}")
    print(f"Top Branch Output shape: {top_branch_output.shape}")

    # Verify model architecture and parameters
    # from torchsummary import summary # requires pip install torchsummary
    # print("\nModel Summary:")
    # if torch.cuda.is_available():
    #     summary(model, [(input_channels, 64, 64, 64), (mask_input_channels, 32, 32, 32)])
    # else:
    #     print("Torchsummary not run (requires GPU for this setup or careful CPU adaptation).")
    
    # A manual check for a few parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params}")

    # Test with different input sizes to confirm flexibility
    # input_data_larger = torch.randn(batch_size, input_channels, 96, 96, 96)
    # mask_input_data_larger = torch.randn(batch_size, mask_input_channels, 48, 48, 48) # Mask encoder input is half of original
    # if torch.cuda.is_available():
    #     input_data_larger = input_data_larger.cuda()
    #     mask_input_data_larger = mask_input_data_larger.cuda()
    # main_output_larger, top_branch_output_larger = model(input_data_larger, mask_input_data_larger)
    # print(f"\nMain Output (larger input) shape: {main_output_larger.shape}")
    # print(f"Top Branch Output (larger input) shape: {top_branch_output_larger.shape}")