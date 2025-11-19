import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.CDEM import *
from net.MAFM import MAFM

class ICLR(nn.Module):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(ICLR, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, scale=0.5, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, scale=0.5, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, scale=0.5, use_norm = norm)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, scale=2.0, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, scale=2.0, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, scale=2.0, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, scale=0.5, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, scale=0.5, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, scale=0.5, use_norm = norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, scale=2.0,use_norm = norm)
        self.ID_block2 = NormUpsample(ch3, ch2, scale=2.0,use_norm = norm)
        self.ID_block1 = NormUpsample(ch2, ch1, scale=2.0,use_norm = norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )
        
        self.CDEM_HV1 = CDEM_HV(ch2, head2)
        self.CDEM_HV2 = CDEM_HV(ch3, head3)
        self.CDEM_HV3 = CDEM_HV(ch4, head4)
        self.CDEM_HV4 = CDEM_HV(ch4, head4)
        self.CDEM_HV5 = CDEM_HV(ch3, head3)
        self.CDEM_HV6 = CDEM_HV(ch2, head2)
        
        self.CDEM_I1 = CDEM_I(ch2, head2)
        self.CDEM_I2 = CDEM_I(ch3, head3)
        self.CDEM_I3 = CDEM_I(ch4, head4)
        self.CDEM_I4 = CDEM_I(ch4, head4)
        self.CDEM_I5 = CDEM_I(ch3, head3)
        self.CDEM_I6 = CDEM_I(ch2, head2)
        
        self.trans = RGB_HVI().cuda()
        self.fusion1 = MAFM(ch1)
        self.fusion2 = MAFM(ch2)
        self.fusion3 = MAFM(ch3)
        self.fusion4 = MAFM(ch4)


    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        hv = hvi[:,0:2,:,:].to(dtypes)

        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hv)
        hv_0 = self.fusion1(hv_0, i_enc0)
        # low
        i_enc1 = self.IE_block1(i_enc0)
        hv_1 = self.HVE_block1(hv_0)
        hv_1 = self.fusion2(hv_1, i_enc1)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.CDEM_I1(i_enc1, hv_1)
        hv_2 = self.CDEM_HV1(hv_1, i_enc1)
        hv_2 = self.fusion2(hv_2, i_enc2)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        hv_2 = self.fusion3(hv_2, i_enc2)
        
        i_enc3 = self.CDEM_I2(i_enc2, hv_2)
        hv_3 = self.CDEM_HV2(hv_2, i_enc2)
        hv_3 = self.fusion3(hv_3, i_enc3)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        hv_3 = self.fusion4(hv_3, i_enc3)
        
        i_enc4 = self.CDEM_I3(i_enc3, hv_3)
        hv_4 = self.CDEM_HV3(hv_3, i_enc3)
        hv_4 = self.fusion4(hv_4, i_enc4)
        
        i_dec4 = self.CDEM_I4(i_enc4,hv_4)
        hv_4 = self.CDEM_HV4(hv_4, i_enc4)
        hv_4 = self.fusion4(hv_4, i_dec4)
        
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        hv_3 = self.fusion3(hv_3, i_dec3)
        i_dec2 = self.CDEM_I5(i_dec3, hv_3)
        hv_2 = self.CDEM_HV5(hv_3, i_dec3)
        hv_2 = self.fusion3(hv_2, i_dec2)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        hv_2 = self.fusion2(hv_2, i_dec2)
        
        i_dec1 = self.CDEM_I6(i_dec2, hv_2)
        hv_1 = self.CDEM_HV6(hv_2, i_dec2)
        hv_1 = self.fusion2(hv_1, i_dec1)
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_1 = self.fusion1(hv_1, i_dec1)
        i_dec0 = self.ID_block0(i_dec1)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    

    

