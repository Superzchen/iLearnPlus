#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os, re
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
                             QVBoxLayout, QSplitter, QTableWidget, QTabWidget,
                             QTableWidgetItem, QMessageBox, QFormLayout, QRadioButton,
                             QHeaderView,
                             QAbstractItemView)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, pyqtSignal
from util import PlotWidgets
import numpy as np
import pandas as pd
import torch
from util.EvaluationMetrics import Metrics
from torch.utils.data import DataLoader
from Nets import (DealDataset, Net_CNN_1, Net_CNN_11, Net_RNN_2, Net_ABCNN_4, Net_ResNet_5, Net_AutoEncoder_6)
import base64
import qdarkstyle
import sip
import joblib

## secret key ##################################################################
CBzeXMuZXhpdChhcHAuZXhlY18o="""#################################################
Y2xhc3MgaUxlYXJuUGx1c0xvYWRNb2RlbChRV2lkZ2V0KToKICAgIGNsb3NlX3NpZ25hbCA9IHB5cXRT
aWduYWwoc3RyKQogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAgIHN1cGVyKGlMZWFyblBsdXNM
b2FkTW9kZWwsIHNlbGYpLl9faW5pdF9fKCkKCiAgICAgICAgIiIiIE1hY2hpbmUgTGVhcm5pbmcgVmFy
aWFibGUgIiIiCiAgICAgICAgc2VsZi5kYXRhX2luZGV4ID0gewogICAgICAgICAgICAnVHJhaW5pbmdf
ZGF0YSc6IE5vbmUsCiAgICAgICAgICAgICdUZXN0aW5nX2RhdGEnOiBOb25lLAogICAgICAgICAgICAn
VHJhaW5pbmdfc2NvcmUnOiBOb25lLAogICAgICAgICAgICAnVGVzdGluZ19zY29yZSc6IE5vbmUsCiAg
ICAgICAgICAgICdNZXRyaWNzJzogTm9uZSwKICAgICAgICAgICAgJ1JPQyc6IE5vbmUsCiAgICAgICAg
ICAgICdQUkMnOiBOb25lLAogICAgICAgICAgICAnTW9kZWwnOiBOb25lLAogICAgICAgIH0KICAgICAg
ICBzZWxmLmN1cnJlbnRfZGF0YV9pbmRleCA9IDAKICAgICAgICBzZWxmLm1sX3J1bm5pbmdfc3RhdHVz
ID0gRmFsc2UKCiAgICAgICAgc2VsZi5tb2RlbF9saXN0ID0gW10KICAgICAgICBzZWxmLmRhdGFmcmFt
ZSA9IE5vbmUKICAgICAgICBzZWxmLmRhdGFsYWJlbCA9IE5vbmUKICAgICAgICBzZWxmLnNjb3JlID0g
Tm9uZQogICAgICAgIHNlbGYubWV0cmljcyA9IE5vbmUKICAgICAgICBzZWxmLmF1Y0RhdGEgPSBOb25l
CiAgICAgICAgc2VsZi5wcmNEYXRhID0gTm9uZQoKICAgICAgICAjIGluaXRpYWxpemUgVUkKICAgICAg
ICBzZWxmLmluaXRVSSgpCgogICAgZGVmIGluaXRVSShzZWxmKToKICAgICAgICBzZWxmLnNldFdpbmRv
d1RpdGxlKCdpTGVhcm5QbHVzIExvYWRNb2RlbCcpCiAgICAgICAgc2VsZi5yZXNpemUoODAwLCA2MDAp
CiAgICAgICAgc2VsZi5zZXRXaW5kb3dTdGF0ZShRdC5XaW5kb3dNYXhpbWl6ZWQpCiAgICAgICAgc2Vs
Zi5zZXRXaW5kb3dJY29uKFFJY29uKCdpbWFnZXMvbG9nby5pY28nKSkKCiAgICAgICAgIyBmaWxlCiAg
ICAgICAgdG9wR3JvdXBCb3ggPSBRR3JvdXBCb3goJ0xvYWQgZGF0YScsIHNlbGYpCiAgICAgICAgdG9w
R3JvdXBCb3guc2V0Rm9udChRRm9udCgnQXJpYWwnLCAxMCkpCiAgICAgICAgdG9wR3JvdXBCb3guc2V0
TWluaW11bUhlaWdodCgxMDApCiAgICAgICAgdG9wR3JvdXBCb3hMYXlvdXQgPSBRRm9ybUxheW91dCgp
CiAgICAgICAgbW9kZWxGaWxlQnV0dG9uID0gUVB1c2hCdXR0b24oJ0xvYWQnKQogICAgICAgIG1vZGVs
RmlsZUJ1dHRvbi5zZXRUb29sVGlwKCdPbmUgb3IgbW9yZSBtb2RlbHMgY291bGQgYmUgbG9hZGVkLicp
CiAgICAgICAgbW9kZWxGaWxlQnV0dG9uLmNsaWNrZWQuY29ubmVjdChzZWxmLmxvYWRNb2RlbCkKICAg
ICAgICB0ZXN0RmlsZUJ1dHRvbiA9IFFQdXNoQnV0dG9uKCdPcGVuJykKICAgICAgICB0ZXN0RmlsZUJ1
dHRvbi5jbGlja2VkLmNvbm5lY3Qoc2VsZi5sb2FkRGF0YUZpbGUpCiAgICAgICAgdG9wR3JvdXBCb3hM
YXlvdXQuYWRkUm93KCdPcGVuIG1vZGVsIGZpbGUocyk6JywgbW9kZWxGaWxlQnV0dG9uKQogICAgICAg
IHRvcEdyb3VwQm94TGF5b3V0LmFkZFJvdygnT3BlbiB0ZXN0aW5nIGZpbGU6JywgdGVzdEZpbGVCdXR0
b24pCiAgICAgICAgdG9wR3JvdXBCb3guc2V0TGF5b3V0KHRvcEdyb3VwQm94TGF5b3V0KQoKICAgICAg
ICAjIHN0YXJ0IGJ1dHRvbgogICAgICAgIHN0YXJ0R3JvdXBCb3ggPSBRR3JvdXBCb3goJ09wZXJhdG9y
Jywgc2VsZikKICAgICAgICBzdGFydEdyb3VwQm94LnNldEZvbnQoUUZvbnQoJ0FyaWFsJywgMTApKQog
ICAgICAgIHN0YXJ0TGF5b3V0ID0gUUhCb3hMYXlvdXQoc3RhcnRHcm91cEJveCkKICAgICAgICBzZWxm
Lm1sX3N0YXJ0X2J1dHRvbiA9IFFQdXNoQnV0dG9uKCdTdGFydCcpCiAgICAgICAgc2VsZi5tbF9zdGFy
dF9idXR0b24uY2xpY2tlZC5jb25uZWN0KHNlbGYucnVuX21vZGVsKQogICAgICAgIHNlbGYubWxfc3Rh
cnRfYnV0dG9uLnNldEZvbnQoUUZvbnQoJ0FyaWFsJywgMTApKQogICAgICAgIHNlbGYubWxfc2F2ZV9i
dXR0b24gPSBRUHVzaEJ1dHRvbignU2F2ZScpCiAgICAgICAgc2VsZi5tbF9zYXZlX2J1dHRvbi5zZXRG
b250KFFGb250KCdBcmlhbCcsIDEwKSkKICAgICAgICBzZWxmLm1sX3NhdmVfYnV0dG9uLmNsaWNrZWQu
Y29ubmVjdChzZWxmLnNhdmVfbWxfZmlsZXMpCiAgICAgICAgc3RhcnRMYXlvdXQuYWRkV2lkZ2V0KHNl
bGYubWxfc3RhcnRfYnV0dG9uKQogICAgICAgIHN0YXJ0TGF5b3V0LmFkZFdpZGdldChzZWxmLm1sX3Nh
dmVfYnV0dG9uKQoKICAgICAgICAjIGxvZwogICAgICAgIGxvZ0dyb3VwQm94ID0gUUdyb3VwQm94KCdM
b2cnLCBzZWxmKQogICAgICAgIGxvZ0dyb3VwQm94LnNldEZvbnQoUUZvbnQoJ0FyaWFsJywgMTApKQog
ICAgICAgIGxvZ0xheW91dCA9IFFIQm94TGF5b3V0KGxvZ0dyb3VwQm94KQogICAgICAgIHNlbGYubG9n
VGV4dEVkaXQgPSBRVGV4dEVkaXQoKQogICAgICAgIHNlbGYubG9nVGV4dEVkaXQuc2V0Rm9udChRRm9u
dCgnQXJpYWwnLCA4KSkKICAgICAgICBsb2dMYXlvdXQuYWRkV2lkZ2V0KHNlbGYubG9nVGV4dEVkaXQp
CgoKICAgICAgICAjIyMgbGF5b3V0CiAgICAgICAgbGVmdF92ZXJ0aWNhbF9sYXlvdXQgPSBRVkJveExh
eW91dCgpCiAgICAgICAgbGVmdF92ZXJ0aWNhbF9sYXlvdXQuYWRkV2lkZ2V0KHRvcEdyb3VwQm94KQog
ICAgICAgIGxlZnRfdmVydGljYWxfbGF5b3V0LmFkZFdpZGdldChsb2dHcm91cEJveCkKICAgICAgICBs
ZWZ0X3ZlcnRpY2FsX2xheW91dC5hZGRXaWRnZXQoc3RhcnRHcm91cEJveCkKCiAgICAgICAgIyMjIyB3
aWRnZXQKICAgICAgICBsZWZ0V2lkZ2V0ID0gUVdpZGdldCgpCiAgICAgICAgbGVmdFdpZGdldC5zZXRM
YXlvdXQobGVmdF92ZXJ0aWNhbF9sYXlvdXQpCgogICAgICAgICMjIyMgdmlldyByZWdpb24KICAgICAg
ICBzY29yZVRhYldpZGdldCA9IFFUYWJXaWRnZXQoKQogICAgICAgIHRyYWluU2NvcmVXaWRnZXQgPSBR
V2lkZ2V0KCkKICAgICAgICBzY29yZVRhYldpZGdldC5zZXRGb250KFFGb250KCdBcmlhbCcsIDgpKQog
ICAgICAgIHNjb3JlVGFiV2lkZ2V0LmFkZFRhYih0cmFpblNjb3JlV2lkZ2V0LCAnUHJlZGljdGlvbiBz
Y29yZSBhbmQgZXZhbHVhdGlvbiBtZXRyaWNzJykKICAgICAgICB0cmFpbl9zY29yZV9sYXlvdXQgPSBR
VkJveExheW91dCh0cmFpblNjb3JlV2lkZ2V0KQogICAgICAgIHNlbGYudHJhaW5fc2NvcmVfdGFibGVX
aWRnZXQgPSBRVGFibGVXaWRnZXQoKQogICAgICAgIHNlbGYudHJhaW5fc2NvcmVfdGFibGVXaWRnZXQu
c2V0Rm9udChRRm9udCgnQXJpYWwnLCA4KSkKICAgICAgICBzZWxmLnRyYWluX3Njb3JlX3RhYmxlV2lk
Z2V0LnNldEVkaXRUcmlnZ2VycyhRQWJzdHJhY3RJdGVtVmlldy5Ob0VkaXRUcmlnZ2VycykKICAgICAg
ICBzZWxmLnRyYWluX3Njb3JlX3RhYmxlV2lkZ2V0Lmhvcml6b250YWxIZWFkZXIoKS5zZXRTZWN0aW9u
UmVzaXplTW9kZShRSGVhZGVyVmlldy5TdHJldGNoKQogICAgICAgIHRyYWluX3Njb3JlX2xheW91dC5h
ZGRXaWRnZXQoc2VsZi50cmFpbl9zY29yZV90YWJsZVdpZGdldCkKCiAgICAgICAgc2VsZi5tZXRyaWNz
VGFibGVXaWRnZXQgPSBRVGFibGVXaWRnZXQoKQogICAgICAgIHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0
LnNldEZvbnQoUUZvbnQoJ0FyaWFsJywgOCkpCiAgICAgICAgc2VsZi5tZXRyaWNzVGFibGVXaWRnZXQu
aG9yaXpvbnRhbEhlYWRlcigpLnNldFNlY3Rpb25SZXNpemVNb2RlKFFIZWFkZXJWaWV3LlN0cmV0Y2gp
CiAgICAgICAgc2VsZi5tZXRyaWNzVGFibGVXaWRnZXQuc2V0RWRpdFRyaWdnZXJzKFFBYnN0cmFjdEl0
ZW1WaWV3Lk5vRWRpdFRyaWdnZXJzKQogICAgICAgIHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0LnJlc2l6
ZVJvd3NUb0NvbnRlbnRzKCkKICAgICAgICBzcGxpdHRlcl9taWRkbGUgPSBRU3BsaXR0ZXIoUXQuVmVy
dGljYWwpCiAgICAgICAgc3BsaXR0ZXJfbWlkZGxlLmFkZFdpZGdldChzY29yZVRhYldpZGdldCkKICAg
ICAgICBzcGxpdHRlcl9taWRkbGUuYWRkV2lkZ2V0KHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0KQoKICAg
ICAgICBzZWxmLmRhdGFUYWJsZVdpZGdldCA9IFFUYWJsZVdpZGdldCgyLCA0KQogICAgICAgIHNlbGYu
ZGF0YVRhYmxlV2lkZ2V0LnNldEZvbnQoUUZvbnQoJ0FyaWFsJywgOCkpCiAgICAgICAgc2VsZi5kYXRh
VGFibGVXaWRnZXQuc2V0RWRpdFRyaWdnZXJzKFFBYnN0cmFjdEl0ZW1WaWV3Lk5vRWRpdFRyaWdnZXJz
KQogICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0LnNldFNob3dHcmlkKEZhbHNlKQogICAgICAgIHNl
bGYuZGF0YVRhYmxlV2lkZ2V0Lmhvcml6b250YWxIZWFkZXIoKS5zZXRTZWN0aW9uUmVzaXplTW9kZShR
SGVhZGVyVmlldy5TdHJldGNoKQogICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0Lmhvcml6b250YWxI
ZWFkZXIoKS5zZXRTZWN0aW9uUmVzaXplTW9kZSgwLCBRSGVhZGVyVmlldy5SZXNpemVUb0NvbnRlbnRz
KQogICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0Lmhvcml6b250YWxIZWFkZXIoKS5zZXRTZWN0aW9u
UmVzaXplTW9kZSgzLCBRSGVhZGVyVmlldy5SZXNpemVUb0NvbnRlbnRzKQogICAgICAgIHNlbGYuZGF0
YVRhYmxlV2lkZ2V0LnNldEhvcml6b250YWxIZWFkZXJMYWJlbHMoWydTZWxlY3QnLCAnRGF0YScsICdT
aGFwZScsICdTb3VyY2UnXSkKICAgICAgICBzZWxmLmRhdGFUYWJsZVdpZGdldC52ZXJ0aWNhbEhlYWRl
cigpLnNldFZpc2libGUoRmFsc2UpCgogICAgICAgIHNlbGYucm9jX2N1cnZlX3dpZGdldCA9IFBsb3RX
aWRnZXRzLkN1cnZlV2lkZ2V0KCkKICAgICAgICBzZWxmLnByY19jdXJ2ZV93aWRnZXQgPSBQbG90V2lk
Z2V0cy5DdXJ2ZVdpZGdldCgpCiAgICAgICAgcGxvdFRhYldpZGdldCA9IFFUYWJXaWRnZXQoKQogICAg
ICAgIHBsb3RUYWJXaWRnZXQuc2V0Rm9udChRRm9udCgnQXJpYWwnLCA4KSkKICAgICAgICByb2NXaWRn
ZXQgPSBRV2lkZ2V0KCkKICAgICAgICBzZWxmLnJvY0xheW91dCA9IFFWQm94TGF5b3V0KHJvY1dpZGdl
dCkKICAgICAgICBzZWxmLnJvY0xheW91dC5hZGRXaWRnZXQoc2VsZi5yb2NfY3VydmVfd2lkZ2V0KQog
ICAgICAgIHByY1dpZGdldCA9IFFXaWRnZXQoKQogICAgICAgIHNlbGYucHJjTGF5b3V0ID0gUUhCb3hM
YXlvdXQocHJjV2lkZ2V0KQogICAgICAgIHNlbGYucHJjTGF5b3V0LmFkZFdpZGdldChzZWxmLnByY19j
dXJ2ZV93aWRnZXQpCiAgICAgICAgcGxvdFRhYldpZGdldC5hZGRUYWIocm9jV2lkZ2V0LCAnUk9DIGN1
cnZlJykKICAgICAgICBwbG90VGFiV2lkZ2V0LmFkZFRhYihwcmNXaWRnZXQsICdQUkMgY3VydmUnKQog
ICAgICAgIHNwbGl0dGVyX3JpZ2h0ID0gUVNwbGl0dGVyKFF0LlZlcnRpY2FsKQogICAgICAgIHNwbGl0
dGVyX3JpZ2h0LmFkZFdpZGdldChzZWxmLmRhdGFUYWJsZVdpZGdldCkKICAgICAgICBzcGxpdHRlcl9y
aWdodC5hZGRXaWRnZXQocGxvdFRhYldpZGdldCkKICAgICAgICBzcGxpdHRlcl9yaWdodC5zZXRTaXpl
cyhbMTAwLCAzMDBdKQoKICAgICAgICBzcGxpdHRlcl92aWV3ID0gUVNwbGl0dGVyKFF0Lkhvcml6b250
YWwpCiAgICAgICAgc3BsaXR0ZXJfdmlldy5hZGRXaWRnZXQoc3BsaXR0ZXJfbWlkZGxlKQogICAgICAg
IHNwbGl0dGVyX3ZpZXcuYWRkV2lkZ2V0KHNwbGl0dGVyX3JpZ2h0KQogICAgICAgIHNwbGl0dGVyX3Zp
ZXcuc2V0U2l6ZXMoWzEwMCwgMjAwXSkKCiAgICAgICAgIyMjIyMgc3BsaXR0ZXIKICAgICAgICBzcGxp
dHRlcl8xID0gUVNwbGl0dGVyKFF0Lkhvcml6b250YWwpCiAgICAgICAgc3BsaXR0ZXJfMS5hZGRXaWRn
ZXQobGVmdFdpZGdldCkKICAgICAgICBzcGxpdHRlcl8xLmFkZFdpZGdldChzcGxpdHRlcl92aWV3KQog
ICAgICAgIHNwbGl0dGVyXzEuc2V0U2l6ZXMoWzEwMCwgMTIwMF0pCgogICAgICAgICMjIyMjIyB2ZXJ0
aWNhbCBsYXlvdXQKICAgICAgICB2TGF5b3V0ID0gUVZCb3hMYXlvdXQoKQoKICAgICAgICAjIyBzdGF0
dXMgYmFyCiAgICAgICAgc3RhdHVzR3JvdXBCb3ggPSBRR3JvdXBCb3goJ1N0YXR1cycsIHNlbGYpCiAg
ICAgICAgc3RhdHVzR3JvdXBCb3guc2V0Rm9udChRRm9udCgnQXJpYWwnLCAxMCkpCiAgICAgICAgc3Rh
dHVzTGF5b3V0ID0gUUhCb3hMYXlvdXQoc3RhdHVzR3JvdXBCb3gpCiAgICAgICAgc2VsZi5tbF9zdGF0
dXNfbGFiZWwgPSBRTGFiZWwoJ1dlbGNvbWUgdG8gaUxlYXJuUGx1cyBMb2FkTW9kZWwnKQogICAgICAg
IHNlbGYubWxfcHJvZ3Jlc3NfYmFyID0gUUxhYmVsKCkKICAgICAgICBzZWxmLm1sX3Byb2dyZXNzX2Jh
ci5zZXRNYXhpbXVtV2lkdGgoMjMwKQogICAgICAgIHN0YXR1c0xheW91dC5hZGRXaWRnZXQoc2VsZi5t
bF9zdGF0dXNfbGFiZWwpCiAgICAgICAgc3RhdHVzTGF5b3V0LmFkZFdpZGdldChzZWxmLm1sX3Byb2dy
ZXNzX2JhcikKCiAgICAgICAgc3BsaXR0ZXJfMiA9IFFTcGxpdHRlcihRdC5WZXJ0aWNhbCkKICAgICAg
ICBzcGxpdHRlcl8yLmFkZFdpZGdldChzcGxpdHRlcl8xKQogICAgICAgIHNwbGl0dGVyXzIuYWRkV2lk
Z2V0KHN0YXR1c0dyb3VwQm94KQogICAgICAgIHNwbGl0dGVyXzIuc2V0U2l6ZXMoWzEwMDAsIDEwMF0p
CiAgICAgICAgdkxheW91dC5hZGRXaWRnZXQoc3BsaXR0ZXJfMikKICAgICAgICBzZWxmLnNldExheW91
dCh2TGF5b3V0KQoKICAgIGRlZiBsb2FkTW9kZWwoc2VsZik6CiAgICAgICAgbW9kZWxfZmlsZXMsIG9r
ID0gUUZpbGVEaWFsb2cuZ2V0T3BlbkZpbGVOYW1lcyhzZWxmLCAnT3BlbicsICcuL2RhdGEnLCAnUEtM
IEZpbGVzICgqLnBrbCknKQogICAgICAgIGlmIGxlbihtb2RlbF9maWxlcykgPiAwOgogICAgICAgICAg
ICBzZWxmLm1vZGVsX2xpc3QgPSBbXQogICAgICAgICAgICBmb3IgZmlsZSBpbiBtb2RlbF9maWxlczoK
ICAgICAgICAgICAgICAgIHRyeToKICAgICAgICAgICAgICAgICAgICBtb2RlbCA9IGpvYmxpYi5sb2Fk
KGZpbGUpCiAgICAgICAgICAgICAgICAgICAgaWYgJ3ByZWRpY3RfcHJvYmEnIG5vdCBpbiBkaXIobW9k
ZWwpOgogICAgICAgICAgICAgICAgICAgICAgICBtb2RlbCA9IHRvcmNoLmxvYWQoZmlsZSkKICAgICAg
ICAgICAgICAgICAgICBzZWxmLm1vZGVsX2xpc3QuYXBwZW5kKG1vZGVsKQogICAgICAgICAgICAgICAg
ZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRp
Y2FsKHNlbGYsICdFcnJvcicsICdMb2FkIG1vZGVsIGZhaWxlZC4nLCBRTWVzc2FnZUJveC5PayB8IFFN
ZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykKICAgICAgICAgICAgICAgICAgICBzZWxmLm1vZGVs
X2xpc3QgPSBbXQogICAgICAgICAgICAgICAgICAgIHJldHVybiBGYWxzZQogICAgICAgICAgICBzZWxm
LmxvZ1RleHRFZGl0LmFwcGVuZCgnTG9hZCBtb2RlbCBzdWNjZXNzZnVsbHkuJykKICAgICAgICAgICAg
c2VsZi5sb2dUZXh0RWRpdC5hcHBlbmQoJ01vZGVsIG51bWJlcjogJXMnICVsZW4obW9kZWxfZmlsZXMp
KQogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIHJldHVybiBG
YWxzZQoKICAgIGRlZiBsb2FkRGF0YUZpbGUoc2VsZik6CiAgICAgICAgZmlsZSwgb2sgPSBRRmlsZURp
YWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgJ09wZW4nLCAnLi9kYXRhJywgJ0NTViBGaWxlcyAoKi5j
c3YpOztUU1YgRmlsZXMgKCoudHN2KTs7U1ZNIEZpbGVzKCouc3ZtKTs7V2VrYSBGaWxlcyAoKi5hcmZm
KScpCiAgICAgICAgaWYgb2s6CiAgICAgICAgICAgIGlmIG5vdCBvcy5wYXRoLmV4aXN0cyhmaWxlKToK
ICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICdFcnJvcicsICdEYXRhIGZp
bGUgZG9lcyBub3QgZXhpc3QuJywgUU1lc3NhZ2VCb3guT2sgfCBRTWVzc2FnZUJveC5ObywgUU1lc3Nh
Z2VCb3guT2spCiAgICAgICAgICAgICAgICByZXR1cm4gRmFsc2UKICAgICAgICAgICAgc2VsZi5kYXRh
ZnJhbWUsIHNlbGYuZGF0YWxhYmVsID0gTm9uZSwgTm9uZQogICAgICAgICAgICB0cnk6CiAgICAgICAg
ICAgICAgICBpZiBmaWxlLmVuZHN3aXRoKCcudHN2Jyk6CiAgICAgICAgICAgICAgICAgICAgZGYgPSBw
ZC5yZWFkX2NzdihmaWxlLCBzZXA9J1x0JywgaGVhZGVyPU5vbmUpCiAgICAgICAgICAgICAgICAgICAg
c2VsZi5kYXRhZnJhbWUgPSBkZi5pbG9jWzosIDE6XQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0
YWZyYW1lLmluZGV4ID0gWydTYW1wbGVfJXMnICUgaSBmb3IgaSBpbiByYW5nZShkYXRhZnJhbWUudmFs
dWVzLnNoYXBlWzBdKV0KICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFmcmFtZS5jb2x1bW5zID0g
WydGXyVzJyAlIGkgZm9yIGkgaW4gcmFuZ2UoZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVsxXSldCiAgICAg
ICAgICAgICAgICAgICAgc2VsZi5kYXRhbGFiZWwgPSBucC5hcnJheShkZi5pbG9jWzosIDBdKS5hc3R5
cGUoaW50KQogICAgICAgICAgICAgICAgZWxpZiBmaWxlLmVuZHN3aXRoKCcuY3N2Jyk6CiAgICAgICAg
ICAgICAgICAgICAgZGYgPSBwZC5yZWFkX2NzdihmaWxlLCBzZXA9JywnLCBoZWFkZXI9Tm9uZSkKICAg
ICAgICAgICAgICAgICAgICBzZWxmLmRhdGFmcmFtZSA9IGRmLmlsb2NbOiwgMTpdCiAgICAgICAgICAg
ICAgICAgICAgc2VsZi5kYXRhZnJhbWUuaW5kZXggPSBbJ1NhbXBsZV8lcycgJSBpIGZvciBpIGluIHJh
bmdlKHNlbGYuZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVswXSldCiAgICAgICAgICAgICAgICAgICAgc2Vs
Zi5kYXRhZnJhbWUuY29sdW1ucyA9IFsnRl8lcycgJSBpIGZvciBpIGluIHJhbmdlKHNlbGYuZGF0YWZy
YW1lLnZhbHVlcy5zaGFwZVsxXSldCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhbGFiZWwgPSBu
cC5hcnJheShkZi5pbG9jWzosIDBdKS5hc3R5cGUoaW50KQogICAgICAgICAgICAgICAgZWxpZiBmaWxl
LmVuZHN3aXRoKCcuc3ZtJyk6CiAgICAgICAgICAgICAgICAgICAgd2l0aCBvcGVuKGZpbGUpIGFzIGY6
CiAgICAgICAgICAgICAgICAgICAgICAgIHJlY29yZCA9IGYucmVhZCgpLnN0cmlwKCkKICAgICAgICAg
ICAgICAgICAgICByZWNvcmQgPSByZS5zdWIoJ1xkKzonLCAnJywgcmVjb3JkKQogICAgICAgICAgICAg
ICAgICAgIGFycmF5ID0gbnAuYXJyYXkoW1tpIGZvciBpIGluIGl0ZW0uc3BsaXQoKV0gZm9yIGl0ZW0g
aW4gcmVjb3JkLnNwbGl0KCdcbicpXSkKICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFmcmFtZSA9
IHBkLkRhdGFGcmFtZShhcnJheVs6LCAxOl0sIGR0eXBlPWZsb2F0KQogICAgICAgICAgICAgICAgICAg
IHNlbGYuZGF0YWZyYW1lLmluZGV4ID0gWydTYW1wbGVfJXMnICUgaSBmb3IgaSBpbiByYW5nZShzZWxm
LmRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMF0pXQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWZy
YW1lLmNvbHVtbnMgPSBbJ0ZfJXMnICUgaSBmb3IgaSBpbiByYW5nZShzZWxmLmRhdGFmcmFtZS52YWx1
ZXMuc2hhcGVbMV0pXQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWxhYmVsID0gYXJyYXlbOiwg
MF0uYXN0eXBlKGludCkKICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgd2l0
aCBvcGVuKGZpbGUpIGFzIGY6CiAgICAgICAgICAgICAgICAgICAgICAgIHJlY29yZCA9IGYucmVhZCgp
LnN0cmlwKCkuc3BsaXQoJ0AnKVstMV0uc3BsaXQoJ1xuJylbMTpdCiAgICAgICAgICAgICAgICAgICAg
YXJyYXkgPSBucC5hcnJheShbaXRlbS5zcGxpdCgnLCcpIGZvciBpdGVtIGluIHJlY29yZF0pCiAgICAg
ICAgICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUgPSBwZC5EYXRhRnJhbWUoYXJyYXlbOiwgMDotMV0s
IGR0eXBlPWZsb2F0KQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmluZGV4ID0gWydT
YW1wbGVfJXMnICUgaSBmb3IgaSBpbiByYW5nZShzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMF0p
XQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmNvbHVtbnMgPSBbJ0ZfJXMnICUgaSBm
b3IgaSBpbiByYW5nZShzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMV0pXQogICAgICAgICAgICAg
ICAgICAgIGxhYmVsID0gW10KICAgICAgICAgICAgICAgICAgICBmb3IgaSBpbiBhcnJheVs6LCAtMV06
CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIGkgPT0gJ3llcyc6CiAgICAgICAgICAgICAgICAgICAg
ICAgICAgICBsYWJlbC5hcHBlbmQoMSkKICAgICAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAg
ICAgICAgICAgICAgICAgICAgICAgIGxhYmVsLmFwcGVuZCgwKQogICAgICAgICAgICAgICAgICAgIHNl
bGYuZGF0YWxhYmVsID0gbnAuYXJyYXkobGFiZWwpCgogICAgICAgICAgICBleGNlcHQgRXhjZXB0aW9u
IGFzIGU6CiAgICAgICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAn
T3BlbiBkYXRhIGZpbGUgZmFpbGVkLicsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFN
ZXNzYWdlQm94Lk9rKQogICAgICAgICAgICAgICAgcmV0dXJuIEZhbHNlCiAgICAgICAgICAgIHNlbGYu
bG9nVGV4dEVkaXQuYXBwZW5kKCdMb2FkIGRhdGEgZmlsZSBzdWNjZXNzZnVsbHkuJykKICAgICAgICAg
ICAgc2VsZi5sb2dUZXh0RWRpdC5hcHBlbmQoJ0RhdGEgc2hhcGU6ICVzJyAlKHN0cihzZWxmLmRhdGFm
cmFtZS52YWx1ZXMuc2hhcGUpKSkKICAgICAgICAgICAgcmV0dXJuIFRydWUKICAgICAgICBlbHNlOgog
ICAgICAgICAgICByZXR1cm4gRmFsc2UKCiAgICBkZWYgcnVuX21vZGVsKHNlbGYpOgogICAgICAgICMg
cmVzZXQKICAgICAgICBzZWxmLnNjb3JlID0gTm9uZQogICAgICAgIHNlbGYubWV0cmljcyA9IE5vbmUK
CiAgICAgICAgaWYgbGVuKHNlbGYubW9kZWxfbGlzdCkgPiAwIGFuZCBub3Qgc2VsZi5kYXRhZnJhbWUg
aXMgTm9uZToKICAgICAgICAgICAgdHJ5OgogICAgICAgICAgICAgICAgcHJlZGljdGlvbl9zY29yZSA9
IE5vbmUKICAgICAgICAgICAgICAgIGZvciBtb2RlbCBpbiBzZWxmLm1vZGVsX2xpc3Q6CiAgICAgICAg
ICAgICAgICAgICAgaWYgJ3ByZWRpY3RfcHJvYmEnIG5vdCBpbiBkaXIobW9kZWwpOgogICAgICAgICAg
ICAgICAgICAgICAgICB2YWxpZF9zZXQgPSBEZWFsRGF0YXNldChzZWxmLmRhdGFmcmFtZS52YWx1ZXMs
IHNlbGYuZGF0YWxhYmVsLnJlc2hhcGUoKC0xLCAxKSkpCiAgICAgICAgICAgICAgICAgICAgICAgIHZh
bGlkX2xvYWRlciA9IERhdGFMb2FkZXIodmFsaWRfc2V0LCBiYXRjaF9zaXplPTUxMiwgc2h1ZmZsZT1G
YWxzZSkKICAgICAgICAgICAgICAgICAgICAgICAgdG1wX3ByZWRpY3Rpb25fc2NvcmUgPSBtb2RlbC5w
cmVkaWN0KHZhbGlkX2xvYWRlcikKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAg
ICAgICAgICAgICB0bXBfcHJlZGljdGlvbl9zY29yZSA9IG1vZGVsLnByZWRpY3RfcHJvYmEoc2VsZi5k
YXRhZnJhbWUudmFsdWVzKQoKICAgICAgICAgICAgICAgICAgICBpZiBwcmVkaWN0aW9uX3Njb3JlIGlz
IE5vbmU6CiAgICAgICAgICAgICAgICAgICAgICAgIHByZWRpY3Rpb25fc2NvcmUgPSB0bXBfcHJlZGlj
dGlvbl9zY29yZQogICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAg
IHByZWRpY3Rpb25fc2NvcmUgKz0gdG1wX3ByZWRpY3Rpb25fc2NvcmUKICAgICAgICAgICAgICAgIHBy
ZWRpY3Rpb25fc2NvcmUgLz0gbGVuKHNlbGYubW9kZWxfbGlzdCkKICAgICAgICAgICAgICAgIHNlbGYu
c2NvcmUgPSBwcmVkaWN0aW9uX3Njb3JlCgogICAgICAgICAgICAgICAgIyBkaXNwbGF5IHByZWRpY3Rp
b24gc2NvcmUKICAgICAgICAgICAgICAgIGlmIG5vdCBzZWxmLnNjb3JlIGlzIE5vbmU6CiAgICAgICAg
ICAgICAgICAgICAgZGF0YSA9IHNlbGYuc2NvcmUKICAgICAgICAgICAgICAgICAgICBzZWxmLnRyYWlu
X3Njb3JlX3RhYmxlV2lkZ2V0LnNldFJvd0NvdW50KGRhdGEuc2hhcGVbMF0pCiAgICAgICAgICAgICAg
ICAgICAgc2VsZi50cmFpbl9zY29yZV90YWJsZVdpZGdldC5zZXRDb2x1bW5Db3VudChkYXRhLnNoYXBl
WzFdKQogICAgICAgICAgICAgICAgICAgIHNlbGYudHJhaW5fc2NvcmVfdGFibGVXaWRnZXQuc2V0SG9y
aXpvbnRhbEhlYWRlckxhYmVscyhbJ1Njb3JlIGZvciBjYXRlZ29yeSAlcycgJWkgZm9yIGkgaW4gcmFu
Z2UoZGF0YS5zaGFwZVsxXSldKQogICAgICAgICAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKGRhdGEu
c2hhcGVbMF0pOgogICAgICAgICAgICAgICAgICAgICAgICBmb3IgaiBpbiByYW5nZShkYXRhLnNoYXBl
WzFdKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNlbGwgPSBRVGFibGVXaWRnZXRJdGVtKHN0
cihyb3VuZChkYXRhW2ldW2pdLCA0KSkpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZWxmLnRy
YWluX3Njb3JlX3RhYmxlV2lkZ2V0LnNldEl0ZW0oaSwgaiwgY2VsbCkKICAgICAgICAgICAgICAgICAg
ICBpZiBzZWxmLmRhdGFfaW5kZXhbJ1RyYWluaW5nX3Njb3JlJ10gaXMgTm9uZToKICAgICAgICAgICAg
ICAgICAgICAgICAgIyBpbmRleCA9IHNlbGYuY3VycmVudF9kYXRhX2luZGV4CiAgICAgICAgICAgICAg
ICAgICAgICAgIGluZGV4ID0gMAogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFfaW5kZXhb
J1RyYWluaW5nX3Njb3JlJ10gPSBpbmRleAogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFU
YWJsZVdpZGdldC5pbnNlcnRSb3coaW5kZXgpCiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYuY3Vy
cmVudF9kYXRhX2luZGV4ICs9IDEKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAg
ICAgICAgICAgICAjIGluZGV4ID0gc2VsZi5kYXRhX2luZGV4WydUcmFpbmluZ19zY29yZSddCiAgICAg
ICAgICAgICAgICAgICAgICAgIGluZGV4ID0gMAogICAgICAgICAgICAgICAgICAgIHNlbGYudHJhaW5p
bmdfc2NvcmVfcmFkaW8gPSBRUmFkaW9CdXR0b24oKQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0
YVRhYmxlV2lkZ2V0LnNldENlbGxXaWRnZXQoaW5kZXgsIDAsIHNlbGYudHJhaW5pbmdfc2NvcmVfcmFk
aW8pCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuc2V0SXRlbShpbmRleCwg
MSwgUVRhYmxlV2lkZ2V0SXRlbSgnVHJhaW5pbmcgc2NvcmUnKSkKICAgICAgICAgICAgICAgICAgICBz
ZWxmLmRhdGFUYWJsZVdpZGdldC5zZXRJdGVtKGluZGV4LCAyLCBRVGFibGVXaWRnZXRJdGVtKHN0cihk
YXRhLnNoYXBlKSkpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuc2V0SXRl
bShpbmRleCwgMywgUVRhYmxlV2lkZ2V0SXRlbSgnTkEnKSkKCiAgICAgICAgICAgICAgICAjIGNhbGN1
bGF0ZSBhbmQgZGlzcGxheSBldmFsdWF0aW9uIG1ldHJpY3MKICAgICAgICAgICAgICAgIGNvbHVtbl9u
YW1lID0gWydTbicsICdTcCcsICdQcmUnLCAnQWNjJywgJ01DQycsICdGMScsICdBVVJPQycsICdBVVBS
QyddCiAgICAgICAgICAgICAgICBpZiBub3Qgc2VsZi5zY29yZSBpcyBOb25lIGFuZCBzZWxmLnNjb3Jl
LnNoYXBlWzFdID09IDI6CiAgICAgICAgICAgICAgICAgICAgIyBjYWxjdWxhdGUgbWV0cmljcwogICAg
ICAgICAgICAgICAgICAgIGRhdGEgPSBzZWxmLnNjb3JlCiAgICAgICAgICAgICAgICAgICAgbWV0cmlj
cyA9IE1ldHJpY3MoZGF0YVs6LCAtMV0sIHNlbGYuZGF0YWxhYmVsKQogICAgICAgICAgICAgICAgICAg
IG1ldHJpY3NfaW5kID0gbnAuYXJyYXkoCiAgICAgICAgICAgICAgICAgICAgICAgIFttZXRyaWNzLnNl
bnNpdGl2aXR5LCBtZXRyaWNzLnNwZWNpZmljaXR5LCBtZXRyaWNzLnByZWNpc2lvbiwgbWV0cmljcy5h
Y2N1cmFjeSwgbWV0cmljcy5tY2MsCiAgICAgICAgICAgICAgICAgICAgICAgICBtZXRyaWNzLmYxLCBt
ZXRyaWNzLmF1YywgbWV0cmljcy5wcmNdKS5yZXNoYXBlKCgxLCAtMSkpCiAgICAgICAgICAgICAgICAg
ICAgaW5kZXhfbmFtZSA9IFsnTWV0cmljcyB2YWx1ZSddCiAgICAgICAgICAgICAgICAgICAgc2VsZi5h
dWNEYXRhID0gWydBVVJPQyA9ICVzJyAlIG1ldHJpY3MuYXVjLCBtZXRyaWNzLmF1Y0RvdF0KICAgICAg
ICAgICAgICAgICAgICBzZWxmLnByY0RhdGEgPSBbJ0FVUFJDID0gJXMnICUgbWV0cmljcy5wcmMsIG1l
dHJpY3MucHJjRG90XQogICAgICAgICAgICAgICAgICAgIGRlbCBtZXRyaWNzCiAgICAgICAgICAgICAg
ICAgICAgc2VsZi5tZXRyaWNzID0gcGQuRGF0YUZyYW1lKG1ldHJpY3NfaW5kLCBpbmRleD1pbmRleF9u
YW1lLCBjb2x1bW5zPWNvbHVtbl9uYW1lKQoKICAgICAgICAgICAgICAgICAgICAjIGRpc3BsYXkgbWV0
cmljcwogICAgICAgICAgICAgICAgICAgIGRhdGEgPSBzZWxmLm1ldHJpY3MudmFsdWVzCiAgICAgICAg
ICAgICAgICAgICAgc2VsZi5tZXRyaWNzVGFibGVXaWRnZXQuc2V0Um93Q291bnQoZGF0YS5zaGFwZVsw
XSkKICAgICAgICAgICAgICAgICAgICBzZWxmLm1ldHJpY3NUYWJsZVdpZGdldC5zZXRDb2x1bW5Db3Vu
dChkYXRhLnNoYXBlWzFdKQogICAgICAgICAgICAgICAgICAgIHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0
LnNldEhvcml6b250YWxIZWFkZXJMYWJlbHMoCiAgICAgICAgICAgICAgICAgICAgICAgIFsnU24gKCUp
JywgJ1NwICglKScsICdQcmUgKCUpJywgJ0FjYyAoJSknLCAnTUNDJywgJ0YxJywgJ0FVUk9DJywgJ0FV
UFJDJ10pCiAgICAgICAgICAgICAgICAgICAgc2VsZi5tZXRyaWNzVGFibGVXaWRnZXQuc2V0VmVydGlj
YWxIZWFkZXJMYWJlbHMoc2VsZi5tZXRyaWNzLmluZGV4KQogICAgICAgICAgICAgICAgICAgIGZvciBp
IGluIHJhbmdlKGRhdGEuc2hhcGVbMF0pOgogICAgICAgICAgICAgICAgICAgICAgICBmb3IgaiBpbiBy
YW5nZShkYXRhLnNoYXBlWzFdKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNlbGwgPSBRVGFi
bGVXaWRnZXRJdGVtKHN0cihkYXRhW2ldW2pdKSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNl
bGYubWV0cmljc1RhYmxlV2lkZ2V0LnNldEl0ZW0oaSwgaiwgY2VsbCkKICAgICAgICAgICAgICAgICAg
ICBpZiBzZWxmLmRhdGFfaW5kZXhbJ01ldHJpY3MnXSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAg
ICAgICAjIGluZGV4ID0gc2VsZi5jdXJyZW50X2RhdGFfaW5kZXgKICAgICAgICAgICAgICAgICAgICAg
ICAgaW5kZXggPSAxCiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YV9pbmRleFsnTWV0cmlj
cyddID0gaW5kZXgKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuaW5z
ZXJ0Um93KGluZGV4KQogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmN1cnJlbnRfZGF0YV9pbmRl
eCArPSAxCiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgIyBp
bmRleCA9IHNlbGYuZGF0YV9pbmRleFsnTWV0cmljcyddCiAgICAgICAgICAgICAgICAgICAgICAgIGlu
ZGV4ID0gMQogICAgICAgICAgICAgICAgICAgIHNlbGYubWV0cmljc19yYWRpbyA9IFFSYWRpb0J1dHRv
bigpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuc2V0Q2VsbFdpZGdldChp
bmRleCwgMCwgc2VsZi5tZXRyaWNzX3JhZGlvKQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YVRh
YmxlV2lkZ2V0LnNldEl0ZW0oaW5kZXgsIDEsIFFUYWJsZVdpZGdldEl0ZW0oJ0V2YWx1YXRpb24gbWV0
cmljcycpKQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0LnNldEl0ZW0oaW5k
ZXgsIDIsIFFUYWJsZVdpZGdldEl0ZW0oc3RyKGRhdGEuc2hhcGUpKSkKICAgICAgICAgICAgICAgICAg
ICBzZWxmLmRhdGFUYWJsZVdpZGdldC5zZXRJdGVtKGluZGV4LCAzLCBRVGFibGVXaWRnZXRJdGVtKCdO
QScpKQoKICAgICAgICAgICAgICAgICNwbG90IFJPQwogICAgICAgICAgICAgICAgaWYgbm90IHNlbGYu
YXVjRGF0YSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgIHNlbGYucm9jTGF5b3V0LnJlbW92ZVdp
ZGdldChzZWxmLnJvY19jdXJ2ZV93aWRnZXQpCiAgICAgICAgICAgICAgICAgICAgc2lwLmRlbGV0ZShz
ZWxmLnJvY19jdXJ2ZV93aWRnZXQpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5yb2NfY3VydmVfd2lk
Z2V0ID0gUGxvdFdpZGdldHMuQ3VydmVXaWRnZXQoKQogICAgICAgICAgICAgICAgICAgIHNlbGYucm9j
X2N1cnZlX3dpZGdldC5pbml0X2RhdGEoMCwgJ1JPQyBjdXJ2ZScsIGluZF9kYXRhPXNlbGYuYXVjRGF0
YSkKICAgICAgICAgICAgICAgICAgICBzZWxmLnJvY0xheW91dC5hZGRXaWRnZXQoc2VsZi5yb2NfY3Vy
dmVfd2lkZ2V0KQoKICAgICAgICAgICAgICAgICMgcGxvdCBQUkMKICAgICAgICAgICAgICAgIGlmIG5v
dCBzZWxmLnByY0RhdGEgaXMgTm9uZToKICAgICAgICAgICAgICAgICAgICBzZWxmLnByY0xheW91dC5y
ZW1vdmVXaWRnZXQoc2VsZi5wcmNfY3VydmVfd2lkZ2V0KQogICAgICAgICAgICAgICAgICAgIHNpcC5k
ZWxldGUoc2VsZi5wcmNfY3VydmVfd2lkZ2V0KQogICAgICAgICAgICAgICAgICAgIHNlbGYucHJjX2N1
cnZlX3dpZGdldCA9IFBsb3RXaWRnZXRzLkN1cnZlV2lkZ2V0KCkKICAgICAgICAgICAgICAgICAgICBz
ZWxmLnByY19jdXJ2ZV93aWRnZXQuaW5pdF9kYXRhKDEsICdQUkMgY3VydmUnLCBpbmRfZGF0YT1zZWxm
LnByY0RhdGEpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5wcmNMYXlvdXQuYWRkV2lkZ2V0KHNlbGYu
cHJjX2N1cnZlX3dpZGdldCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAg
ICAgICAgICAgUU1lc3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgc3RyKGUpLCBRTWVzc2Fn
ZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykKICAgICAgICBlbHNlOgogICAg
ICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAnUGxlYXNlIGxvYWQgdGhl
IG1vZGVsIGZpbGUocykgb3IgZGF0YSBmaWxlLicsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3gu
Tm8sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94Lk9rKQoKICAgIGRl
ZiBzYXZlX21sX2ZpbGVzKHNlbGYpOgogICAgICAgIHRhZyA9IDAKICAgICAgICB0cnk6CiAgICAgICAg
ICAgIGlmIHNlbGYudHJhaW5pbmdfc2NvcmVfcmFkaW8uaXNDaGVja2VkKCk6CiAgICAgICAgICAgICAg
ICB0YWcgPSAxCiAgICAgICAgICAgICAgICBzYXZlX2ZpbGUsIG9rID0gUUZpbGVEaWFsb2cuZ2V0U2F2
ZUZpbGVOYW1lKHNlbGYsICdTYXZlJywgJy4vZGF0YScsICdUU1YgRmlsZXMgKCoudHN2KScpCiAgICAg
ICAgICAgICAgICBpZiBvazoKICAgICAgICAgICAgICAgICAgICBvazEgPSBzZWxmLnNhdmVfcHJlZGlj
dGlvbl9zY29yZShzYXZlX2ZpbGUpICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAg
ICBpZiBub3Qgb2sxOgogICAgICAgICAgICAgICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChz
ZWxmLCAnRXJyb3InLCAnU2F2ZSBmaWxlIGZhaWxlZC4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdl
Qm94Lk5vLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFFNZXNzYWdl
Qm94Lk9rKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgcGFzcwoKICAg
ICAgICB0cnk6CiAgICAgICAgICAgIGlmIHNlbGYubWV0cmljc19yYWRpby5pc0NoZWNrZWQoKToKICAg
ICAgICAgICAgICAgIHRhZyA9IDEKICAgICAgICAgICAgICAgIHNhdmVfZmlsZSwgb2sgPSBRRmlsZURp
YWxvZy5nZXRTYXZlRmlsZU5hbWUoc2VsZiwgJ1NhdmUnLCAnLi9kYXRhJywgJ1RTViBGaWxlcyAoKi50
c3YpJykKICAgICAgICAgICAgICAgIGlmIG9rOgogICAgICAgICAgICAgICAgICAgIG9rMSA9IHNlbGYu
c2F2ZV9tZXRyaWNzKHNhdmVfZmlsZSkgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAg
ICAgIGlmIG5vdCBvazE6CiAgICAgICAgICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2Fs
KHNlbGYsICdFcnJvcicsICdTYXZlIGZpbGUgZmFpbGVkLicsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3Nh
Z2VCb3guTm8sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgUU1lc3Nh
Z2VCb3guT2spCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXNzCgog
ICAgICAgIGlmIHRhZyA9PSAwOgogICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAn
RXJyb3InLCAnUGxlYXNlIHNlbGVjdCB3aGljaCBkYXRhIHRvIHNhdmUuJywgUU1lc3NhZ2VCb3guT2sg
fCBRTWVzc2FnZUJveC5ObywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgUU1lc3NhZ2VC
b3guT2spCgogICAgZGVmIHNhdmVfcHJlZGljdGlvbl9zY29yZShzZWxmLCBmaWxlKToKICAgICAgICB0
cnk6CiAgICAgICAgICAgIGRmID0gcGQuRGF0YUZyYW1lKHNlbGYuc2NvcmUsIGNvbHVtbnM9WydTY29y
ZV8lcycgJWkgZm9yIGkgaW4gcmFuZ2Uoc2VsZi5zY29yZS5zaGFwZVsxXSldKQogICAgICAgICAgICBk
Zi50b19jc3YoZmlsZSwgc2VwPSdcdCcsIGhlYWRlcj1UcnVlLCBpbmRleD1GYWxzZSkKICAgICAgICAg
ICAgcmV0dXJuIFRydWUKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIHBy
aW50KGUpCiAgICAgICAgICAgIHJldHVybiBGYWxzZQoKICAgIGRlZiBzYXZlX21ldHJpY3Moc2VsZiwg
ZmlsZSk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBzZWxmLm1ldHJpY3MudG9fY3N2KGZpbGUsIHNl
cD0nXHQnLCBoZWFkZXI9VHJ1ZSwgaW5kZXg9VHJ1ZSkKICAgICAgICAgICAgcmV0dXJuIFRydWUKICAg
ICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIHJldHVybiBGYWxzZQoKICAgIGRl
ZiBjbG9zZUV2ZW50KHNlbGYsIGV2ZW50KToKICAgICAgICByZXBseSA9IFFNZXNzYWdlQm94LnF1ZXN0
aW9uKHNlbGYsICdDb25maXJtIEV4aXQnLCAnQXJlIHlvdSBzdXJlIHdhbnQgdG8gcXVpdCBpTGVhcm5Q
bHVzPycsIFFNZXNzYWdlQm94LlllcyB8IFFNZXNzYWdlQm94Lk5vLAogICAgICAgICAgICAgICAgICAg
ICAgICAgICAgICAgICAgICAgUU1lc3NhZ2VCb3guTm8pCiAgICAgICAgaWYgcmVwbHkgPT0gUU1lc3Nh
Z2VCb3guWWVzOgogICAgICAgICAgICBzZWxmLmNsb3NlX3NpZ25hbC5lbWl0KCdMb2FkTW9kZWwnKQog
ICAgICAgICAgICBzZWxmLmNsb3NlKCkKICAgICAgICBlbHNlOgogICAgICAgICAgICBpZiBldmVudDoK
ICAgICAgICAgICAgICAgIGV2ZW50Lmlnbm9yZSgpCg=="""#PVRydWUsIGluZGV4PUZhbHNlKQogICAg
eval(compile(base64.b64decode(CBzeXMuZXhpdChhcHAuZXhlY18o),'<string>','exec'))##
###################### end of secret key #######################################
#  ____ ____ ____ ____ ____ ____ ____ _________ ____ ____ ____ ____            #
# ||A |||u |||t |||h |||o |||r |||: |||       |||Z |||h |||e |||n ||           #
# ||__|||__|||__|||__|||__|||__|||__|||_______|||__|||__|||__|||__||           #
# |/__\|/__\|/__\|/__\|/__\|/__\|/__\|/_______\|/__\|/__\|/__\|/__\|           #
#                                                                              #
#  ____                               ___                                      #
# /\  _`\                         __ /\_ \                                     #
# \ \ \L\_\    ___ ___      __   /\_\\//\ \    __                              #
#  \ \  _\L  /' __` __`\  /'__`\ \/\ \ \ \ \  /\_\                             #
#   \ \ \L\ \/\ \/\ \/\ \/\ \L\.\_\ \ \ \_\ \_\/_/_                            #
#    \ \____/\ \_\ \_\ \_\ \__/.\_\\ \_\/\____\ /\_\                           #
#     \/___/  \/_/\/_/\/_/\/__/\/_/ \/_/\/____/ \/_/  chenzhen-win2009@163.com #
#                                                                              #
#                t#####f,                                                      #
#              .##i..L### .                                                    #
#              ##.     #   :                                                   #
#             D##j    ,##E##                                                   #
#             ####f.:G######E                                                  # 
#            .###############                                                  #
#            .W##############.                                                 #
#             E##############                                                  #
#             iW#############                                                  #
#      .iLt    EW###########;                                                  #
#     fDDEEDEjEEEE#############L.                                              #
#    tDE.   tLLE.;EEK############,                                             #
#    EE        D    ##############                                             #
#    .j       LE    ##############                                             #
#            :DL   i#############W                                             #
#          ,DDE    D###t##########                                             #
#        tDDDt     ###W ;#########                                             #
#         LE.      ###E  i########                                             #
#          .      D###;   L#######       .##,.                                 #
#                 ####:    #######W      ######f                               #
#                ;####     #########i    #######.                              #
#        ###:    ####W    .###########i  K######:                              #
#       #####i .#####i    ##############W#######:                              #
#       #####t.######.   t########W#############                               #
#    :#######D######D    #########;G############                               #
#    ###############.   W#########i j##########:                               #
#   :##############G   ;#####.####t  j########j                                #
#   .##############    #####G.####W   .i####E:                                 #
#    #############.   K#####  #####                                            #
#    .###########.   j#####i .#####                                            #
#      f#######,     #####D  .#####                                            #
#        :;Li.      t#####.  .#####                                            #
#                   #####L   :#####                                            #
#                  L#####    i#####:                                           #
#                 ,#####,    f#####:                                           #
#                 #####K     L#####:                                           #
#                G#####.     L#####:                                           #
#               ,#####f      L#####,                                           #
#               ######       K#####i                                           #
#              ######;       ######i                                           #
#             f######        ######;                                           #
#            ,######i        ######i                                           #
#            #######        .######i                                           #
#           #######,        t######i                                           #
#          E#######         G######i                                           #
################################################################################
# main class 
class iLearnPLusLoadModel(QWidget):
    def __init__(self):
        super(iLearnPLusLoadModel, self).__init__()
        self.data_index = {
            'Training_data': None,
            'Testing_data': None,
            'Training_score': None,
            'Testing_score': None,
            'Metrics': None,
            'ROC': None,
            'PRC': None,
            'Model': None,
        }
        self.current_data_index = 0
        self.ml_running_status = False

        self.model_list = []
        self.dataframe = None
        self.datalabel = None
        self.score = None
        self.metrics = None
        self.aucData = None
        self.prcData = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus LoadModel')
        self.resize(800, 600)
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowIcon(QIcon('images/logo.ico'))

        topGroupBox = QGroupBox('Load data', self)
        topGroupBox.setFont(QFont('Arial', 10))
        topGroupBox.setMinimumHeight(100)
        topGroupBoxLayout = QFormLayout()
        modelFileButton = QPushButton('Load')
        modelFileButton.setToolTip('One or more models could be loaded.')        
        testFileButton = QPushButton('Open')        
        topGroupBoxLayout.addRow('Open model file(s):', modelFileButton)
        topGroupBoxLayout.addRow('Open testing file:', testFileButton)
        topGroupBox.setLayout(topGroupBoxLayout)

        startGroupBox = QGroupBox('Operator', self)
        startGroupBox.setFont(QFont('Arial', 10))
        startLayout = QHBoxLayout(startGroupBox)
        self.ml_start_button = QPushButton('Start')        
        self.ml_start_button.setFont(QFont('Arial', 10))
        self.ml_save_button = QPushButton('Save')
        self.ml_save_button.setFont(QFont('Arial', 10))        
        startLayout.addWidget(self.ml_start_button)
        startLayout.addWidget(self.ml_save_button)

        logGroupBox = QGroupBox('Log', self)
        logGroupBox.setFont(QFont('Arial', 10))
        logLayout = QHBoxLayout(logGroupBox)
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setFont(QFont('Arial', 8))
        logLayout.addWidget(self.logTextEdit)


        left_vertical_layout = QVBoxLayout()
        left_vertical_layout.addWidget(topGroupBox)
        left_vertical_layout.addWidget(logGroupBox)
        left_vertical_layout.addWidget(startGroupBox)

        leftWidget = QWidget()
        leftWidget.setLayout(left_vertical_layout)

        scoreTabWidget = QTabWidget()
        trainScoreWidget = QWidget()
        scoreTabWidget.setFont(QFont('Arial', 8))
        scoreTabWidget.addTab(trainScoreWidget, 'Prediction score and evaluation metrics')
        train_score_layout = QVBoxLayout(trainScoreWidget)
        self.train_score_tableWidget = QTableWidget()
        self.train_score_tableWidget.setFont(QFont('Arial', 8))
        self.train_score_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.train_score_tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        train_score_layout.addWidget(self.train_score_tableWidget)

        self.metricsTableWidget = QTableWidget()
        self.metricsTableWidget.setFont(QFont('Arial', 8))
        self.metricsTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metricsTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metricsTableWidget.resizeRowsToContents()
        splitter_middle = QSplitter(Qt.Vertical)
        splitter_middle.addWidget(scoreTabWidget)
        splitter_middle.addWidget(self.metricsTableWidget)

        self.dataTableWidget = QTableWidget(0, 4)
        self.dataTableWidget.setFont(QFont('Arial', 8))
        self.dataTableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dataTableWidget.setShowGrid(False)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.dataTableWidget.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.dataTableWidget.setHorizontalHeaderLabels(['Select', 'Data', 'Shape', 'Source'])
        self.dataTableWidget.verticalHeader().setVisible(False)

        self.roc_curve_widget = PlotWidgets.CurveWidget()
        self.prc_curve_widget = PlotWidgets.CurveWidget()
        plotTabWidget = QTabWidget()
        plotTabWidget.setFont(QFont('Arial', 8))
        rocWidget = QWidget()
        self.rocLayout = QVBoxLayout(rocWidget)
        self.rocLayout.addWidget(self.roc_curve_widget)
        prcWidget = QWidget()
        self.prcLayout = QHBoxLayout(prcWidget)
        self.prcLayout.addWidget(self.prc_curve_widget)
        plotTabWidget.addTab(rocWidget, 'ROC curve')
        plotTabWidget.addTab(prcWidget, 'PRC curve')
        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = iLearnPlusLoadModel()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())