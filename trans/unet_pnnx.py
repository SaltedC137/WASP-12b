import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.convbn2d_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.inc_double_conv_2 = nn.ReLU()
        self.convbn2d_1 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.inc_double_conv_5 = nn.ReLU()
        self.down1_maxpool_conv_0 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down1_maxpool_conv_1_double_conv_2 = nn.ReLU()
        self.convbn2d_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down1_maxpool_conv_1_double_conv_5 = nn.ReLU()
        self.down2_maxpool_conv_0 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_4 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down2_maxpool_conv_1_double_conv_2 = nn.ReLU()
        self.convbn2d_5 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down2_maxpool_conv_1_double_conv_5 = nn.ReLU()
        self.down3_maxpool_conv_0 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down3_maxpool_conv_1_double_conv_2 = nn.ReLU()
        self.convbn2d_7 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down3_maxpool_conv_1_double_conv_5 = nn.ReLU()
        self.down4_maxpool_conv_0 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.convbn2d_8 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=1024, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down4_maxpool_conv_1_double_conv_2 = nn.ReLU()
        self.convbn2d_9 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1024, kernel_size=(3,3), out_channels=1024, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.down4_maxpool_conv_1_double_conv_5 = nn.ReLU()
        self.up1_up = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=1024, kernel_size=(2,2), out_channels=512, output_padding=(0,0), padding=(0,0), stride=(2,2))
        self.convbn2d_10 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=1024, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up1_conv_double_conv_2 = nn.ReLU()
        self.convbn2d_11 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=512, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up1_conv_double_conv_5 = nn.ReLU()
        self.up2_up = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(2,2), out_channels=256, output_padding=(0,0), padding=(0,0), stride=(2,2))
        self.convbn2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=512, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up2_conv_double_conv_2 = nn.ReLU()
        self.convbn2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up2_conv_double_conv_5 = nn.ReLU()
        self.up3_up = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(2,2), out_channels=128, output_padding=(0,0), padding=(0,0), stride=(2,2))
        self.convbn2d_14 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up3_conv_double_conv_2 = nn.ReLU()
        self.convbn2d_15 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up3_conv_double_conv_5 = nn.ReLU()
        self.up4_up = nn.ConvTranspose2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(2,2), out_channels=64, output_padding=(0,0), padding=(0,0), stride=(2,2))
        self.convbn2d_16 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up4_conv_double_conv_2 = nn.ReLU()
        self.convbn2d_17 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.up4_conv_double_conv_5 = nn.ReLU()
        self.outc_conv = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(1,1), out_channels=2, padding=(0,0), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('.\unet.pnnx.bin', 'r')
        self.convbn2d_0.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.bias', (64), 'float32')
        self.convbn2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_0.weight', (64,3,3,3), 'float32')
        self.convbn2d_1.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.bias', (64), 'float32')
        self.convbn2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_1.weight', (64,64,3,3), 'float32')
        self.convbn2d_2.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.bias', (128), 'float32')
        self.convbn2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_2.weight', (128,64,3,3), 'float32')
        self.convbn2d_3.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.bias', (128), 'float32')
        self.convbn2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_3.weight', (128,128,3,3), 'float32')
        self.convbn2d_4.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.bias', (256), 'float32')
        self.convbn2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_4.weight', (256,128,3,3), 'float32')
        self.convbn2d_5.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.bias', (256), 'float32')
        self.convbn2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_5.weight', (256,256,3,3), 'float32')
        self.convbn2d_6.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.bias', (512), 'float32')
        self.convbn2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_6.weight', (512,256,3,3), 'float32')
        self.convbn2d_7.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.bias', (512), 'float32')
        self.convbn2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_7.weight', (512,512,3,3), 'float32')
        self.convbn2d_8.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.bias', (1024), 'float32')
        self.convbn2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_8.weight', (1024,512,3,3), 'float32')
        self.convbn2d_9.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.bias', (1024), 'float32')
        self.convbn2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_9.weight', (1024,1024,3,3), 'float32')
        self.up1_up.bias = self.load_pnnx_bin_as_parameter(archive, 'up1.up.bias', (512), 'float32')
        self.up1_up.weight = self.load_pnnx_bin_as_parameter(archive, 'up1.up.weight', (1024,512,2,2), 'float32')
        self.convbn2d_10.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_10.bias', (512), 'float32')
        self.convbn2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_10.weight', (512,1024,3,3), 'float32')
        self.convbn2d_11.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_11.bias', (512), 'float32')
        self.convbn2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_11.weight', (512,512,3,3), 'float32')
        self.up2_up.bias = self.load_pnnx_bin_as_parameter(archive, 'up2.up.bias', (256), 'float32')
        self.up2_up.weight = self.load_pnnx_bin_as_parameter(archive, 'up2.up.weight', (512,256,2,2), 'float32')
        self.convbn2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_12.bias', (256), 'float32')
        self.convbn2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_12.weight', (256,512,3,3), 'float32')
        self.convbn2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_13.bias', (256), 'float32')
        self.convbn2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_13.weight', (256,256,3,3), 'float32')
        self.up3_up.bias = self.load_pnnx_bin_as_parameter(archive, 'up3.up.bias', (128), 'float32')
        self.up3_up.weight = self.load_pnnx_bin_as_parameter(archive, 'up3.up.weight', (256,128,2,2), 'float32')
        self.convbn2d_14.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_14.bias', (128), 'float32')
        self.convbn2d_14.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_14.weight', (128,256,3,3), 'float32')
        self.convbn2d_15.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_15.bias', (128), 'float32')
        self.convbn2d_15.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_15.weight', (128,128,3,3), 'float32')
        self.up4_up.bias = self.load_pnnx_bin_as_parameter(archive, 'up4.up.bias', (64), 'float32')
        self.up4_up.weight = self.load_pnnx_bin_as_parameter(archive, 'up4.up.weight', (128,64,2,2), 'float32')
        self.convbn2d_16.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_16.bias', (64), 'float32')
        self.convbn2d_16.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_16.weight', (64,128,3,3), 'float32')
        self.convbn2d_17.bias = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_17.bias', (64), 'float32')
        self.convbn2d_17.weight = self.load_pnnx_bin_as_parameter(archive, 'convbn2d_17.weight', (64,64,3,3), 'float32')
        self.outc_conv.bias = self.load_pnnx_bin_as_parameter(archive, 'outc.conv.bias', (2), 'float32')
        self.outc_conv.weight = self.load_pnnx_bin_as_parameter(archive, 'outc.conv.weight', (2,64,1,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.convbn2d_0(v_0)
        v_2 = self.inc_double_conv_2(v_1)
        v_3 = self.convbn2d_1(v_2)
        v_4 = self.inc_double_conv_5(v_3)
        v_5 = self.down1_maxpool_conv_0(v_4)
        v_6 = self.convbn2d_2(v_5)
        v_7 = self.down1_maxpool_conv_1_double_conv_2(v_6)
        v_8 = self.convbn2d_3(v_7)
        v_9 = self.down1_maxpool_conv_1_double_conv_5(v_8)
        v_10 = self.down2_maxpool_conv_0(v_9)
        v_11 = self.convbn2d_4(v_10)
        v_12 = self.down2_maxpool_conv_1_double_conv_2(v_11)
        v_13 = self.convbn2d_5(v_12)
        v_14 = self.down2_maxpool_conv_1_double_conv_5(v_13)
        v_15 = self.down3_maxpool_conv_0(v_14)
        v_16 = self.convbn2d_6(v_15)
        v_17 = self.down3_maxpool_conv_1_double_conv_2(v_16)
        v_18 = self.convbn2d_7(v_17)
        v_19 = self.down3_maxpool_conv_1_double_conv_5(v_18)
        v_20 = self.down4_maxpool_conv_0(v_19)
        v_21 = self.convbn2d_8(v_20)
        v_22 = self.down4_maxpool_conv_1_double_conv_2(v_21)
        v_23 = self.convbn2d_9(v_22)
        v_24 = self.down4_maxpool_conv_1_double_conv_5(v_23)
        v_25 = self.up1_up(v_24)
        v_26 = F.pad(v_25, mode='constant', pad=(0,1,0,1), value=None)
        v_27 = torch.cat((v_19, v_26), dim=1)
        v_28 = self.convbn2d_10(v_27)
        v_29 = self.up1_conv_double_conv_2(v_28)
        v_30 = self.convbn2d_11(v_29)
        v_31 = self.up1_conv_double_conv_5(v_30)
        v_32 = self.up2_up(v_31)
        v_33 = F.pad(v_32, mode='constant', pad=(0,1,0,1), value=None)
        v_34 = torch.cat((v_14, v_33), dim=1)
        v_35 = self.convbn2d_12(v_34)
        v_36 = self.up2_conv_double_conv_2(v_35)
        v_37 = self.convbn2d_13(v_36)
        v_38 = self.up2_conv_double_conv_5(v_37)
        v_39 = self.up3_up(v_38)
        v_40 = torch.cat((v_9, v_39), dim=1)
        v_41 = self.convbn2d_14(v_40)
        v_42 = self.up3_conv_double_conv_2(v_41)
        v_43 = self.convbn2d_15(v_42)
        v_44 = self.up3_conv_double_conv_5(v_43)
        v_45 = self.up4_up(v_44)
        v_46 = torch.cat((v_4, v_45), dim=1)
        v_47 = self.convbn2d_16(v_46)
        v_48 = self.up4_conv_double_conv_2(v_47)
        v_49 = self.convbn2d_17(v_48)
        v_50 = self.up4_conv_double_conv_5(v_49)
        v_51 = self.outc_conv(v_50)
        return v_51

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 572, 572, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save(".\unet_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 572, 572, dtype=torch.float)

    torch.onnx.export(net, v_0, ".\unet_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 572, 572, dtype=torch.float)

    import pnnx
    pnnx.export(net, ".\unet_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 572, 572, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
