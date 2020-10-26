#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QAction, QMessageBox, QFileDialog)
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
import PlotWidgets, InputDialog
import base64
import threading
import sip
import pandas as pd
import numpy as np

## secret key ##################################################################
CBzeXMuZXhpdChhcHAuZXhlY18o="""#################################################
Y2xhc3MgUGxvdEN1cnZlKFFNYWluV2luZG93KToKICAgIGRlZiBfX2luaXRfXyhzZWxmLCBjdXJ2ZT0n
Uk9DJyk6CiAgICAgICAgc3VwZXIoUGxvdEN1cnZlLCBzZWxmKS5fX2luaXRfXygpCiAgICAgICAgc2Vs
Zi5pbml0VUkoKQogICAgICAgIHNlbGYuY3VydmUgPSBjdXJ2ZQogICAgICAgIHNlbGYuZGF0YSA9IFtd
CiAgICAgICAgc2VsZi5yYXdfZGF0YSA9IHt9CgogICAgZGVmIGluaXRVSShzZWxmKToKICAgICAgICBi
YXIgPSBzZWxmLm1lbnVCYXIoKQogICAgICAgIGZpbGUgPSBiYXIuYWRkTWVudSgnRmlsZScpCiAgICAg
ICAgYWRkID0gUUFjdGlvbignQWRkIGN1cnZlJywgc2VsZikKICAgICAgICBhZGQudHJpZ2dlcmVkLmNv
bm5lY3Qoc2VsZi5pbXBvcnREYXRhKQogICAgICAgIGNvbXAgPSBRQWN0aW9uKCdDb21wYXJlIGN1cnZl
cycsIHNlbGYpCiAgICAgICAgY29tcC50cmlnZ2VyZWQuY29ubmVjdChzZWxmLmNvbXBhcmVDdXJ2ZXMp
CiAgICAgICAgcXVpdCA9IFFBY3Rpb24oJ0V4aXQnLCBzZWxmKQogICAgICAgIHF1aXQudHJpZ2dlcmVk
LmNvbm5lY3Qoc2VsZi5jbG9zZSkKICAgICAgICBmaWxlLmFkZEFjdGlvbihhZGQpCiAgICAgICAgZmls
ZS5hZGRBY3Rpb24oY29tcCkKICAgICAgICBmaWxlLmFkZFNlcGFyYXRvcigpCiAgICAgICAgZmlsZS5h
ZGRBY3Rpb24ocXVpdCkKCiAgICAgICAgaGVscCA9IGJhci5hZGRNZW51KCdIZWxwJykKICAgICAgICBh
Ym91dCA9IFFBY3Rpb24oJ0RvY3VtZW50Jywgc2VsZikKICAgICAgICBhYm91dC50cmlnZ2VyZWQuY29u
bmVjdChzZWxmLm9wZW5Eb2N1bWVudFVybCkKICAgICAgICBoZWxwLmFkZEFjdGlvbihhYm91dCkKCiAg
ICAgICAgc2VsZi5zZXRXaW5kb3dUaXRsZSgnaUxlYXJuUGx1cyBQbG90IGN1cnZlJykKICAgICAgICBz
ZWxmLnJlc2l6ZSg2MDAsIDYwMCkKICAgICAgICBzZWxmLnNldFdpbmRvd0ljb24oUUljb24oJ2ltYWdl
cy9sb2dvLmljbycpKQogICAgICAgIGN1cnZlV2lkZ2V0ID0gUVdpZGdldCgpCiAgICAgICAgc2VsZi5j
dXJ2ZUxheW91dCA9IFFWQm94TGF5b3V0KGN1cnZlV2lkZ2V0KQogICAgICAgIHNlbGYuY3VydmVHcmFw
aCA9IFBsb3RXaWRnZXRzLkN1c3RvbUN1cnZlV2lkZ2V0KCkKICAgICAgICBzZWxmLmN1cnZlTGF5b3V0
LmFkZFdpZGdldChzZWxmLmN1cnZlR3JhcGgpCiAgICAgICAgc2VsZi5zZXRDZW50cmFsV2lkZ2V0KGN1
cnZlV2lkZ2V0KQoKICAgIGRlZiBjbG9zZUV2ZW50KHNlbGYsIGV2ZW50KToKICAgICAgICByZXBseSA9
IFFNZXNzYWdlQm94LnF1ZXN0aW9uKHNlbGYsICdNZXNzYWdlJywgJ0FyZSB5b3Ugc3VyZSB0byBxdWl0
PycsIFFNZXNzYWdlQm94LlllcyB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5ObykKICAgICAg
ICBpZiByZXBseSA9PSBRTWVzc2FnZUJveC5ZZXM6CiAgICAgICAgICAgIHNlbGYuY2xvc2UoKQoKICAg
IGRlZiBpbXBvcnREYXRhKHNlbGYpOgogICAgICAgIGF1YywgZG90LCBjb2xvciwgbGluZVdpZHRoLCBs
aW5lU3R5bGUsIHByZWZpeCwgcmF3X2RhdGEsIG9rID0gSW5wdXREaWFsb2cuUVBsb3RJbnB1dC5nZXRW
YWx1ZXMoc2VsZi5jdXJ2ZSkKICAgICAgICBpZiBub3QgYXVjIGlzIE5vbmUgYW5kIG5vdCBwcmVmaXgg
aXMgTm9uZSBhbmQgb2s6CiAgICAgICAgICAgIHNlbGYucmF3X2RhdGFbcHJlZml4XSA9IHJhd19kYXRh
CiAgICAgICAgICAgIGlmIHNlbGYuY3VydmUgPT0gJ1JPQyc6CiAgICAgICAgICAgICAgICBzZWxmLmRh
dGEuYXBwZW5kKFsnJXMgQVVDID0gJXMnJShwcmVmaXgsIGF1YyksIGRvdCwgbGluZVdpZHRoLCBsaW5l
U3R5bGUsIGNvbG9yXSkKICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgIHNlbGYuZGF0YS5h
cHBlbmQoWyclcyBBVVBSQyA9ICVzJyUocHJlZml4LCBhdWMpLCBkb3QsIGxpbmVXaWR0aCwgbGluZVN0
eWxlLCBjb2xvcl0pCiAgICAgICAgICAgIHNlbGYuY3VydmVMYXlvdXQucmVtb3ZlV2lkZ2V0KHNlbGYu
Y3VydmVHcmFwaCkKICAgICAgICAgICAgc2lwLmRlbGV0ZShzZWxmLmN1cnZlR3JhcGgpCiAgICAgICAg
ICAgIHNlbGYuY3VydmVHcmFwaCA9IFBsb3RXaWRnZXRzLkN1c3RvbUN1cnZlV2lkZ2V0KCkKICAgICAg
ICAgICAgaWYgc2VsZi5jdXJ2ZSA9PSAnUk9DJzoKICAgICAgICAgICAgICAgIHNlbGYuY3VydmVHcmFw
aC5pbml0X2RhdGEoMCwgJ1JPQyBjdXJ2ZScsIHNlbGYuZGF0YSkKICAgICAgICAgICAgZWxzZToKICAg
ICAgICAgICAgICAgIHNlbGYuY3VydmVHcmFwaC5pbml0X2RhdGEoMSwgJ1BSQyBjdXJ2ZScsIHNlbGYu
ZGF0YSkKICAgICAgICAgICAgc2VsZi5jdXJ2ZUxheW91dC5hZGRXaWRnZXQoc2VsZi5jdXJ2ZUdyYXBo
KQoKICAgIGRlZiBjb21wYXJlQ3VydmVzKHNlbGYpOgogICAgICAgIGlmIGxlbihzZWxmLnJhd19kYXRh
KSA+PSAyOgogICAgICAgICAgICBtZXRob2QsIGJvb3RzdHJhcF9uLCBvayA9IElucHV0RGlhbG9nLlFT
dGF0aWNzSW5wdXQuZ2V0VmFsdWVzKCkKICAgICAgICAgICAgaWYgb2s6CiAgICAgICAgICAgICAgICBz
ZWxmLnN1YldpbiA9IFBsb3RXaWRnZXRzLkJvb3RzdHJhcFRlc3RXaWRnZXQoc2VsZi5yYXdfZGF0YSwg
Ym9vdHN0cmFwX24sIHNlbGYuY3VydmUpCiAgICAgICAgICAgICAgICBzZWxmLnN1Yldpbi5zZXRXaW5k
b3dUaXRsZSgnQ2FsY3VsYXRpbmcgcCB2YWx1ZXMgLi4uICcpCiAgICAgICAgICAgICAgICBzZWxmLnN1
Yldpbi5yZXNpemUoNjAwLCA2MDApCiAgICAgICAgICAgICAgICB0ID0gdGhyZWFkaW5nLlRocmVhZCh0
YXJnZXQ9c2VsZi5zdWJXaW4uYm9vdHN0cmFwVGVzdCkKICAgICAgICAgICAgICAgIHQuc3RhcnQoKQog
ICAgICAgICAgICAgICAgc2VsZi5zdWJXaW4uc2hvdygpCiAgICAgICAgZWxzZToKICAgICAgICAgICAg
UU1lc3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgJ1R3byBvciBtb3JlIGN1cnZlIGNvdWxk
IGJlIGNvbXBhcmVkIScsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94
Lk9rKQoKICAgIGRlZiBvcGVuRG9jdW1lbnRVcmwoc2VsZik6CiAgICAgICAgUURlc2t0b3BTZXJ2aWNl
cy5vcGVuVXJsKFFVcmwoJ2h0dHBzOi8vaWxlYXJucGx1cy5lcmMubW9uYXNoLmVkdS8nKSkKCmNsYXNz
IFNjYXR0ZXJQbG90KFFNYWluV2luZG93KToKICAgIGRlZiBfX2luaXRfXyhzZWxmKToKICAgICAgICBz
dXBlcihTY2F0dGVyUGxvdCwgc2VsZikuX19pbml0X18oKQogICAgICAgIHNlbGYuaW5pdFVJKCkgICAg
ICAgCgogICAgZGVmIGluaXRVSShzZWxmKToKICAgICAgICBiYXIgPSBzZWxmLm1lbnVCYXIoKQogICAg
ICAgIGZpbGUgPSBiYXIuYWRkTWVudSgnRmlsZScpCiAgICAgICAgYWRkID0gUUFjdGlvbignT3BlbiBm
aWxlJywgc2VsZikKICAgICAgICBhZGQudHJpZ2dlcmVkLmNvbm5lY3Qoc2VsZi5pbXBvcnREYXRhKSAg
ICAgICAgCiAgICAgICAgcXVpdCA9IFFBY3Rpb24oJ0V4aXQnLCBzZWxmKQogICAgICAgIHF1aXQudHJp
Z2dlcmVkLmNvbm5lY3Qoc2VsZi5jbG9zZSkKICAgICAgICBmaWxlLmFkZEFjdGlvbihhZGQpICAgICAg
ICAKICAgICAgICBmaWxlLmFkZFNlcGFyYXRvcigpCiAgICAgICAgZmlsZS5hZGRBY3Rpb24ocXVpdCkg
ICAgICAgCgogICAgICAgIHNlbGYuc2V0V2luZG93VGl0bGUoJ2lMZWFyblBsdXMgU2NhdHRlciBQbG90
JykKICAgICAgICBzZWxmLnJlc2l6ZSg2MDAsIDYwMCkKICAgICAgICBzZWxmLnNldFdpbmRvd0ljb24o
UUljb24oJ2ltYWdlcy9sb2dvLmljbycpKQogICAgICAgIHBsb3RXaWRnZXQgPSBRV2lkZ2V0KCkKICAg
ICAgICBzZWxmLnBsb3RMYXlvdXQgPSBRVkJveExheW91dChwbG90V2lkZ2V0KQogICAgICAgIHNlbGYu
cGxvdEdyYXBoID0gUGxvdFdpZGdldHMuQ2x1c3RlcmluZ0RpYWdyYW1NYXRwbG90bGliKCkKICAgICAg
ICBzZWxmLnBsb3RMYXlvdXQuYWRkV2lkZ2V0KHNlbGYucGxvdEdyYXBoKQogICAgICAgIHNlbGYuc2V0
Q2VudHJhbFdpZGdldChwbG90V2lkZ2V0KQogICAgCiAgICBkZWYgY2xvc2VFdmVudChzZWxmLCBldmVu
dCk6CiAgICAgICAgcmVwbHkgPSBRTWVzc2FnZUJveC5xdWVzdGlvbihzZWxmLCAnTWVzc2FnZScsICdB
cmUgeW91IHN1cmUgdG8gcXVpdD8nLCBRTWVzc2FnZUJveC5ZZXMgfCBRTWVzc2FnZUJveC5ObywgUU1l
c3NhZ2VCb3guTm8pCiAgICAgICAgaWYgcmVwbHkgPT0gUU1lc3NhZ2VCb3guWWVzOgogICAgICAgICAg
ICBzZWxmLmNsb3NlKCkKCiAgICBkZWYgaW1wb3J0RGF0YShzZWxmKToKICAgICAgICB0cnk6CiAgICAg
ICAgICAgIGZpbGUsIG9rID0gUUZpbGVEaWFsb2cuZ2V0T3BlbkZpbGVOYW1lKHNlbGYsICdPcGVuJywg
Jy4vZGF0YScsICdUU1YgKCoudHN2KScpCiAgICAgICAgICAgIGlmIG9rOgogICAgICAgICAgICAgICAg
ZGF0YSA9IHBkLnJlYWRfY3N2KGZpbGUsIGRlbGltaXRlcj0nXHQnLCBoZWFkZXI9MCwgZHR5cGU9Zmxv
YXQpICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgY2F0ZWdvcnkgPVtpbnQoaSkgZm9yIGkg
aW4gc29ydGVkKHNldChkYXRhLmlsb2NbOiwgMF0pKV0KICAgICAgICAgICAgICAgIG5ld19kYXRhID0g
W10KICAgICAgICAgICAgICAgIGZvciBjIGluIGNhdGVnb3J5OgogICAgICAgICAgICAgICAgICAgIGlu
ZGV4ID0gbnAud2hlcmUoZGF0YS52YWx1ZXNbOiwwXSA9PSBjKVswXQogICAgICAgICAgICAgICAgICAg
IG5ld19kYXRhLmFwcGVuZChbYywgZGF0YS52YWx1ZXNbaW5kZXgsIDE6XV0pCiAgICAgICAgICAgICAg
ICBzZWxmLnBsb3RMYXlvdXQucmVtb3ZlV2lkZ2V0KHNlbGYucGxvdEdyYXBoKQogICAgICAgICAgICAg
ICAgc2lwLmRlbGV0ZShzZWxmLnBsb3RHcmFwaCkKICAgICAgICAgICAgICAgIHNlbGYucGxvdEdyYXBo
ID0gUGxvdFdpZGdldHMuQ2x1c3RlcmluZ0RpYWdyYW1NYXRwbG90bGliKCkKICAgICAgICAgICAgICAg
IHNlbGYucGxvdEdyYXBoLmluaXRfZGF0YSgnU2NhdHRlciBQbG90JywgbmV3X2RhdGEsIGRhdGEuY29s
dW1uc1sxXSwgZGF0YS5jb2x1bW5zWzJdKQogICAgICAgICAgICAgICAgc2VsZi5wbG90TGF5b3V0LmFk
ZFdpZGdldChzZWxmLnBsb3RHcmFwaCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAg
ICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICdFcnJvcicsICdQbGVhc2UgY2hlY2sgeW91
ciBpbnB1dC4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykK
CmNsYXNzIEhpc3QoUU1haW5XaW5kb3cpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1
cGVyKEhpc3QsIHNlbGYpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLmluaXRVSSgpICAgICAgICAKCiAg
ICBkZWYgaW5pdFVJKHNlbGYpOgogICAgICAgIGJhciA9IHNlbGYubWVudUJhcigpCiAgICAgICAgZmls
ZSA9IGJhci5hZGRNZW51KCdGaWxlJykKICAgICAgICBhZGQgPSBRQWN0aW9uKCdPcGVuIGZpbGUnLCBz
ZWxmKQogICAgICAgIGFkZC50cmlnZ2VyZWQuY29ubmVjdChzZWxmLmltcG9ydERhdGEpICAgICAgICAK
ICAgICAgICBxdWl0ID0gUUFjdGlvbignRXhpdCcsIHNlbGYpCiAgICAgICAgcXVpdC50cmlnZ2VyZWQu
Y29ubmVjdChzZWxmLmNsb3NlKQogICAgICAgIGZpbGUuYWRkQWN0aW9uKGFkZCkgICAgICAgIAogICAg
ICAgIGZpbGUuYWRkU2VwYXJhdG9yKCkKICAgICAgICBmaWxlLmFkZEFjdGlvbihxdWl0KSAgICAgICAK
CiAgICAgICAgc2VsZi5zZXRXaW5kb3dUaXRsZSgnaUxlYXJuUGx1cyBIaXN0b2dyYW0gYW5kIEtlcm5l
bCBkZW5zaXR5IHBsb3QnKQogICAgICAgIHNlbGYucmVzaXplKDYwMCwgNjAwKQogICAgICAgIHNlbGYu
c2V0V2luZG93SWNvbihRSWNvbignaW1hZ2VzL2xvZ28uaWNvJykpCiAgICAgICAgcGxvdFdpZGdldCA9
IFFXaWRnZXQoKQogICAgICAgIHNlbGYucGxvdExheW91dCA9IFFWQm94TGF5b3V0KHBsb3RXaWRnZXQp
CiAgICAgICAgc2VsZi5wbG90R3JhcGggPSBQbG90V2lkZ2V0cy5IaXN0b2dyYW1XaWRnZXQoKQogICAg
ICAgIHNlbGYucGxvdExheW91dC5hZGRXaWRnZXQoc2VsZi5wbG90R3JhcGgpCiAgICAgICAgc2VsZi5z
ZXRDZW50cmFsV2lkZ2V0KHBsb3RXaWRnZXQpCiAgICAKICAgIGRlZiBjbG9zZUV2ZW50KHNlbGYsIGV2
ZW50KToKICAgICAgICByZXBseSA9IFFNZXNzYWdlQm94LnF1ZXN0aW9uKHNlbGYsICdNZXNzYWdlJywg
J0FyZSB5b3Ugc3VyZSB0byBxdWl0PycsIFFNZXNzYWdlQm94LlllcyB8IFFNZXNzYWdlQm94Lk5vLCBR
TWVzc2FnZUJveC5ObykKICAgICAgICBpZiByZXBseSA9PSBRTWVzc2FnZUJveC5ZZXM6CiAgICAgICAg
ICAgIHNlbGYuY2xvc2UoKQoKICAgIGRlZiBpbXBvcnREYXRhKHNlbGYpOgogICAgICAgIHRyeToKICAg
ICAgICAgICAgZmlsZSwgb2sgPSBRRmlsZURpYWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgJ09wZW4n
LCAnLi9kYXRhJywgJ1RTViAoKi50c3YpJykKICAgICAgICAgICAgaWYgb2s6CiAgICAgICAgICAgICAg
ICBkYXRhID0gbnAubG9hZHR4dChmaWxlLCBkdHlwZT1mbG9hdCwgZGVsaW1pdGVyPSdcdCcpCiAgICAg
ICAgICAgICAgICBmaWxsX3plcm8gPSBucC56ZXJvcygoZGF0YS5zaGFwZVswXSwgMSkpICAgICAgICAg
ICAgCiAgICAgICAgICAgICAgICBkYXRhID0gbnAuaHN0YWNrKChucC56ZXJvcygoZGF0YS5zaGFwZVsw
XSwgMSkpLCBkYXRhKSkKICAgICAgICAgICAgICAgIHNlbGYucGxvdExheW91dC5yZW1vdmVXaWRnZXQo
c2VsZi5wbG90R3JhcGgpCiAgICAgICAgICAgICAgICBzaXAuZGVsZXRlKHNlbGYucGxvdEdyYXBoKQog
ICAgICAgICAgICAgICAgc2VsZi5wbG90R3JhcGggPSBQbG90V2lkZ2V0cy5IaXN0b2dyYW1XaWRnZXQo
KQogICAgICAgICAgICAgICAgc2VsZi5wbG90R3JhcGguaW5pdF9kYXRhKCdEYXRhIGRpc3RyaWJ1dGlv
bicsIGRhdGEpCiAgICAgICAgICAgICAgICBzZWxmLnBsb3RMYXlvdXQuYWRkV2lkZ2V0KHNlbGYucGxv
dEdyYXBoKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgUU1lc3NhZ2VC
b3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgJ1BsZWFzZSBjaGVjayB5b3VyIGlucHV0LicsIFFNZXNz
YWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9rKQo="""#ZGF0YS5zaGFwZVss
eval(compile(base64.b64decode(CBzeXMuZXhpdChhcHAuZXhlY18o),'<string>','exec'))##

class Plotcurve(QMainWindow):
    def __init__(self, curve='ROC'):
        super(Plotcurve, self).__init__()
        self.initUI()
        self.curve = curve
        self.data = []
        self.raw_data = {}
        
    def importData(self):
        auc, dot, color, lineWidth, lineStyle, prefix, raw_data, ok = InputDialog.QPlotInput.getValues(self.curve)
        if not auc is None and not prefix is None and ok:
            self.raw_data[prefix] = raw_data
            if self.curve == 'ROC':
                self.data.append(['%s AUC = %s'%(prefix, auc), dot, lineWidth, lineStyle, color])
            else:
                self.data.append(['%s AUPRC = %s'%(prefix, auc), dot, lineWidth, lineStyle, color])
            self.curveLayout.removeWidget(self.curveGraph)
            sip.delete(self.curveGraph)
            self.curveGraph = PlotWidgets.CustomCurveWidget()
            if self.curve == 'ROC':
                self.curveGraph.init_data(0, 'ROC curve', self.data)
            else:
                self.curveGraph.init_data(1, 'PRC curve', self.data)
            self.curveLayout.addWidget(self.curveGraph)

    def openDocumentUrl(self):
        QDesktopServices.openUrl(QUrl('https://ilearnplus.erc.monash.edu/'))

class Scatterplot(QMainWindow):
    def __init__(self):
        super(Scatterplot, self).__init__()
        self.initUI()       

    def initUI(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        add = QAction('Open file', self)        
        quit = QAction('Exit', self)        
        file.addAction(add)        
        file.addSeparator()
        file.addAction(quit)        

class HIst(QMainWindow):
    def __init__(self):
        super(HIst, self).__init__()
        self.initUI()        

    def initUI(self):
        bar = self.menuBar()
        file = bar.addMenu('File')
        add = QAction('Open file', self)        
        quit = QAction('Exit', self)        
        file.addAction(add)        
        file.addSeparator()
        file.addAction(quit)
        self.setWindowTitle('iLearnPlus Histogram and Kernel density plot')
        self.resize(600, 600)
        self.setWindowIcon(QIcon('images/logo.ico'))
        plotWidget = QWidget()
        self.plotLayout = QVBoxLayout(plotWidget)
        self.plotGraph = PlotWidgets.HistogramWidget()
        self.plotLayout.addWidget(self.plotGraph)
        self.setCentralWidget(plotWidget)
    
 





