import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as t


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nin, eps=1e-5)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        x = self.pointwise(x)
        x = self.bn_pw(x)
        x = F.relu(x, inplace=True)
        return x

class depthwise_separable_conv_stride2(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv_stride2, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=3, padding=1, stride=2, groups=nin)
        self.bn_dw = nn.BatchNorm2d(nout, eps=1e-5)
        #self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
        #self.bn_pw = nn.BatchNorm2d(nout, eps=1e-5)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x, inplace=True) 
        #x = self.pointwise(x)
        #x = self.bn_pw(x)
        #x = F.relu(x, inplace=True)
        return x


class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 43, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 42, kernel_size=3, padding=1)
    #self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    #self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=5, padding=2)
    #self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
  
  def forward(self, x):
    branch1x1 = self.branch1x1(x)
    
    #branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)
    
    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)
    
    #branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    #branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    #branch3x3_3 = self.branch3x3_3(branch3x3_2)
    
    #outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    outputs = [branch1x1, branch1x1_2, branch3x3]
    return torch.cat(outputs, 1)


class CRelu(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)
  
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x

class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)

    
class Shuffle(nn.Module):

    def __init__(self, in_planes, out_planes, stride, groups):
        super(Shuffle, self).__init__()
        self.stride = stride

        mid_planes = out_planes//4
        g = 1 if in_planes==16 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        #self.shortcut = nn.Sequential()
        #if stride == 2:
        #    self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        #res = self.shortcut(x)
        #out = F.relu(torch.cat([out,res], 1))
        return out


class Shufflev2(nn.Module):
    def __init__(self, in_c, out_c, downsample=False):
        super(Shufflev2, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
            # 3*3 dw conv, stride = 2
            nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            # 1*1 pw conv
            nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(True)
            )
      
            self.branch2 = nn.Sequential(
            # 1*1 pw conv
            nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(True),
            # 3*3 dw conv, stride = 2
            nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
            nn.BatchNorm2d(half_c),
            # 1*1 pw conv
            nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(True)
            )
        else:
            #in_c = out_c
            #assert in_c == out_c
        
            self.branch2 = nn.Sequential(
            # 1*1 pw conv
            nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(True),
            # 3*3 dw conv, stride = 1
            nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
            nn.BatchNorm2d(half_c),
            # 1*1 pw conv
            nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half_c),
            nn.ReLU(True)
            )
      
      
    def forward(self, x):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)
    

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        #self.inplanes = inplanes

        self.squeeze = BasicConv2d(inplanes, squeeze_planes, kernel_size=1, padding=0)
        self.expand1x1 = BasicConv2d(squeeze_planes, expand1x1_planes, kernel_size=1, padding=0)
        self.expand3x3 = BasicConv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)


    def forward(self, x):
        identity = x
        x = self.squeeze(x)
        out1 = self.expand1x1(x)
        out2 = self.expand3x3(x)
 
        x = torch.cat((out1, out2), 1)
        #return x

        #x = identity + x
        return x

def channel_shuffle(x, groups=2):
  bat_size, channels, w, h = x.shape
  group_c = channels // groups
  x = x.view(bat_size, groups, group_c, w, h)
  x = t.transpose(x, 1, 2).contiguous()
  x = x.view(bat_size, -1, w, h)
  return x


class Att(nn.Module):
    def __init__(self, channel):
        super(Att, self).__init__()
        self.c1 = nn.Conv2d(channel, 1, kernel_size=1, padding=0)
        #self.c1 = nn.AvgPool2d(x, x.size()[2:])
        #.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        #b, c, _, _ = x.size()
        #y = self.avg_pool(x).view(b, c)
        #y = self.fc(y).view(b, c, 1, 1)
        
        y = self.c1(x)
        y = self.sigmoid(y)
        y = x * y       

        return x + y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        y = x * y       

        return x + y

class SpatialAttention_eff(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_eff, self).__init__()
        self.c1 = nn.Conv2d(1, 1, kernel_size=1, padding=0, bias=False)   
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        #max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = self.c1(avg_out)
        y = self.sigmoid(y)
        y = x * y       

        return x


class stem(nn.Module):

    #def __init__(self, in_c, out_c):
    def __init__(self):
        super(stem, self).__init__()
        
        self.c3x3_1 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.c3x3_2 = BasicConv2d(64, 64, kernel_size=3, padding=1)       

        #self.att = Att(64)
        self.sa = SpatialAttention_eff()

    def forward(self, x):
        split = torch.chunk(x, 2, dim=1)
 
        x = self.c3x3_1(split[1])        
        x = self.c3x3_2(x)

        #y = self.att(split[0])
        y = self.sa(split[0])
        x = torch.cat((y, x), 1)


        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            #nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel // reduction, eps=1e-5),
            nn.ReLU(inplace=True),
            #nn.Linear(channel // reduction, channel, bias=False),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.conv(y).view(b, c, 1, 1)
        return x * y.expand_as(x)      






class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(4, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.bn1 = nn.BatchNorm2d(channel)
        self.act = h_swish()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        #print(y.shape)
        # Two different branches of ECA module
        y_ = y.squeeze(-1).transpose(-1, -2)

        #print(y_.shape)        
        y = self.conv(y_).transpose(-1, -2).unsqueeze(-1)
        y = self.bn1(y)
        y = self.act(y) 
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        #print(y.shape)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)












class Face(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(Face, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size
    
    self.conv1 = BasicConv2d(3, 32, kernel_size=7, stride=8, padding=1)
    self.conv2_1 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv2_2 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
    #self.conv2_3 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

    self.stem_1 = stem()
    self.stem_2 = stem()
    self.stem_3 = stem()
    #self.stem_4 = stem()
    self.se1 = eca_layer(128, k_size=3)

    #self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    #self.conv_dw_tr1 = depthwise_separable_conv(128, 128)
    self.conv_dw_std1 = depthwise_separable_conv_stride2(128, 256)
    #self.se2 = eca_layer(256, k_size=3)
    #self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    #self.se2 = SELayer(256)

    #self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)  
    #self.conv_dw_tr2 = depthwise_separable_conv(256, 128)  
    self.conv_dw_std2 = depthwise_separable_conv_stride2(256, 256)
    #self.se3 = eca_layer(256, k_size=3)
    #self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    #self.se3 = SELayer(256)
    
    self.loc, self.conf = self.multibox(self.num_classes)
    
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 4 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 4 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 2 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 2 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
    
  def forward(self, x):
  
    sources = list()
    loc = list()
    conf = list()
    detection_dimension = list()

    x = self.conv1(x)
    x = self.conv2_1(x)
    x = self.conv2_2(x)
    #x = self.conv2_3(x)

    x = self.stem_1(x)
    x = self.stem_2(x)
    x = self.stem_3(x)
    #x = self.stem_4(x)
    x = self.se1(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    #x = self.conv3_1(x)    
    #x = self.conv_dw_tr1(x)
    x = self.conv_dw_std1(x)
    #x = self.conv3_2(x)
    #x = self.se2(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)

    #x = self.conv4_1(x)    
    #x = self.conv_dw_tr2(x)
    x = self.conv_dw_std2(x)
    #x = self.conv4_2(x)
    #x = self.se3(x)
    detection_dimension.append(x.shape[2:])
    sources.append(x)
    
    detection_dimension = torch.tensor(detection_dimension, device=x.device)

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)),
                detection_dimension)
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                detection_dimension)
  
    return output
