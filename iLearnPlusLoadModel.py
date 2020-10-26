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
ICAgICBzZWxmLmRhdGFUYWJsZVdpZGdldCA9IFFUYWJsZVdpZGdldCgwLCA0KQogICAgICAgIHNlbGYu
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
KQogICAgICAgICAgICByZXR1cm4gVHJ1ZQoKICAgIGRlZiBsb2FkRGF0YUZpbGUoc2VsZik6CiAgICAg
ICAgZmlsZSwgb2sgPSBRRmlsZURpYWxvZy5nZXRPcGVuRmlsZU5hbWUoc2VsZiwgJ09wZW4nLCAnLi9k
YXRhJywgJ0NTViBGaWxlcyAoKi5jc3YpOztUU1YgRmlsZXMgKCoudHN2KTs7U1ZNIEZpbGVzKCouc3Zt
KTs7V2VrYSBGaWxlcyAoKi5hcmZmKScpCiAgICAgICAgaWYgb2s6CiAgICAgICAgICAgIGlmIG5vdCBv
cy5wYXRoLmV4aXN0cyhmaWxlKToKICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNl
bGYsICdFcnJvcicsICdEYXRhIGZpbGUgZG9lcyBub3QgZXhpc3QuJywgUU1lc3NhZ2VCb3guT2sgfCBR
TWVzc2FnZUJveC5ObywgUU1lc3NhZ2VCb3guT2spCiAgICAgICAgICAgICAgICByZXR1cm4gRmFsc2UK
ICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUsIHNlbGYuZGF0YWxhYmVsID0gTm9uZSwgTm9uZQogICAg
ICAgICAgICB0cnk6CiAgICAgICAgICAgICAgICBpZiBmaWxlLmVuZHN3aXRoKCcudHN2Jyk6CiAgICAg
ICAgICAgICAgICAgICAgZGYgPSBwZC5yZWFkX2NzdihmaWxlLCBzZXA9J1x0JywgaGVhZGVyPU5vbmUp
CiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUgPSBkZi5pbG9jWzosIDE6XQogICAgICAg
ICAgICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmluZGV4ID0gWydTYW1wbGVfJXMnICUgaSBmb3IgaSBp
biByYW5nZShkYXRhZnJhbWUudmFsdWVzLnNoYXBlWzBdKV0KICAgICAgICAgICAgICAgICAgICBzZWxm
LmRhdGFmcmFtZS5jb2x1bW5zID0gWydGXyVzJyAlIGkgZm9yIGkgaW4gcmFuZ2UoZGF0YWZyYW1lLnZh
bHVlcy5zaGFwZVsxXSldCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhbGFiZWwgPSBucC5hcnJh
eShkZi5pbG9jWzosIDBdKS5hc3R5cGUoaW50KQogICAgICAgICAgICAgICAgZWxpZiBmaWxlLmVuZHN3
aXRoKCcuY3N2Jyk6CiAgICAgICAgICAgICAgICAgICAgZGYgPSBwZC5yZWFkX2NzdihmaWxlLCBzZXA9
JywnLCBoZWFkZXI9Tm9uZSkKICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFmcmFtZSA9IGRmLmls
b2NbOiwgMTpdCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUuaW5kZXggPSBbJ1NhbXBs
ZV8lcycgJSBpIGZvciBpIGluIHJhbmdlKHNlbGYuZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVswXSldCiAg
ICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUuY29sdW1ucyA9IFsnRl8lcycgJSBpIGZvciBp
IGluIHJhbmdlKHNlbGYuZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVsxXSldCiAgICAgICAgICAgICAgICAg
ICAgc2VsZi5kYXRhbGFiZWwgPSBucC5hcnJheShkZi5pbG9jWzosIDBdKS5hc3R5cGUoaW50KQogICAg
ICAgICAgICAgICAgZWxpZiBmaWxlLmVuZHN3aXRoKCcuc3ZtJyk6CiAgICAgICAgICAgICAgICAgICAg
d2l0aCBvcGVuKGZpbGUpIGFzIGY6CiAgICAgICAgICAgICAgICAgICAgICAgIHJlY29yZCA9IGYucmVh
ZCgpLnN0cmlwKCkKICAgICAgICAgICAgICAgICAgICByZWNvcmQgPSByZS5zdWIoJ1xkKzonLCAnJywg
cmVjb3JkKQogICAgICAgICAgICAgICAgICAgIGFycmF5ID0gbnAuYXJyYXkoW1tpIGZvciBpIGluIGl0
ZW0uc3BsaXQoKV0gZm9yIGl0ZW0gaW4gcmVjb3JkLnNwbGl0KCdcbicpXSkKICAgICAgICAgICAgICAg
ICAgICBzZWxmLmRhdGFmcmFtZSA9IHBkLkRhdGFGcmFtZShhcnJheVs6LCAxOl0sIGR0eXBlPWZsb2F0
KQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmluZGV4ID0gWydTYW1wbGVfJXMnICUg
aSBmb3IgaSBpbiByYW5nZShzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMF0pXQogICAgICAgICAg
ICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmNvbHVtbnMgPSBbJ0ZfJXMnICUgaSBmb3IgaSBpbiByYW5n
ZShzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMV0pXQogICAgICAgICAgICAgICAgICAgIHNlbGYu
ZGF0YWxhYmVsID0gYXJyYXlbOiwgMF0uYXN0eXBlKGludCkKICAgICAgICAgICAgICAgIGVsc2U6CiAg
ICAgICAgICAgICAgICAgICAgd2l0aCBvcGVuKGZpbGUpIGFzIGY6CiAgICAgICAgICAgICAgICAgICAg
ICAgIHJlY29yZCA9IGYucmVhZCgpLnN0cmlwKCkuc3BsaXQoJ0AnKVstMV0uc3BsaXQoJ1xuJylbMTpd
CiAgICAgICAgICAgICAgICAgICAgYXJyYXkgPSBucC5hcnJheShbaXRlbS5zcGxpdCgnLCcpIGZvciBp
dGVtIGluIHJlY29yZF0pCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhZnJhbWUgPSBwZC5EYXRh
RnJhbWUoYXJyYXlbOiwgMDotMV0sIGR0eXBlPWZsb2F0KQogICAgICAgICAgICAgICAgICAgIHNlbGYu
ZGF0YWZyYW1lLmluZGV4ID0gWydTYW1wbGVfJXMnICUgaSBmb3IgaSBpbiByYW5nZShzZWxmLmRhdGFm
cmFtZS52YWx1ZXMuc2hhcGVbMF0pXQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWZyYW1lLmNv
bHVtbnMgPSBbJ0ZfJXMnICUgaSBmb3IgaSBpbiByYW5nZShzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hh
cGVbMV0pXQogICAgICAgICAgICAgICAgICAgIGxhYmVsID0gW10KICAgICAgICAgICAgICAgICAgICBm
b3IgaSBpbiBhcnJheVs6LCAtMV06CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIGkgPT0gJ3llcyc6
CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYWJlbC5hcHBlbmQoMSkKICAgICAgICAgICAgICAg
ICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxhYmVsLmFwcGVuZCgwKQog
ICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YWxhYmVsID0gbnAuYXJyYXkobGFiZWwpCgogICAgICAg
ICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0
aWNhbChzZWxmLCAnRXJyb3InLCAnT3BlbiBkYXRhIGZpbGUgZmFpbGVkLicsIFFNZXNzYWdlQm94Lk9r
IHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9rKQogICAgICAgICAgICAgICAgcmV0dXJuIEZh
bHNlCiAgICAgICAgICAgIHNlbGYubG9nVGV4dEVkaXQuYXBwZW5kKCdMb2FkIGRhdGEgZmlsZSBzdWNj
ZXNzZnVsbHkuJykKICAgICAgICAgICAgc2VsZi5sb2dUZXh0RWRpdC5hcHBlbmQoJ0RhdGEgc2hhcGU6
ICVzJyAlKHN0cihzZWxmLmRhdGFmcmFtZS52YWx1ZXMuc2hhcGUpKSkKICAgICAgICAgICAgcmV0dXJu
IFRydWUKCiAgICBkZWYgcnVuX21vZGVsKHNlbGYpOgogICAgICAgICMgcmVzZXQKICAgICAgICBzZWxm
LnNjb3JlID0gTm9uZQogICAgICAgIHNlbGYubWV0cmljcyA9IE5vbmUKCiAgICAgICAgaWYgbGVuKHNl
bGYubW9kZWxfbGlzdCkgPiAwIGFuZCBub3Qgc2VsZi5kYXRhZnJhbWUgaXMgTm9uZToKICAgICAgICAg
ICAgdHJ5OgogICAgICAgICAgICAgICAgcHJlZGljdGlvbl9zY29yZSA9IE5vbmUKICAgICAgICAgICAg
ICAgIGZvciBtb2RlbCBpbiBzZWxmLm1vZGVsX2xpc3Q6CiAgICAgICAgICAgICAgICAgICAgaWYgJ3By
ZWRpY3RfcHJvYmEnIG5vdCBpbiBkaXIobW9kZWwpOgogICAgICAgICAgICAgICAgICAgICAgICB2YWxp
ZF9zZXQgPSBEZWFsRGF0YXNldChzZWxmLmRhdGFmcmFtZS52YWx1ZXMsIHNlbGYuZGF0YWxhYmVsLnJl
c2hhcGUoKC0xLCAxKSkpCiAgICAgICAgICAgICAgICAgICAgICAgIHZhbGlkX2xvYWRlciA9IERhdGFM
b2FkZXIodmFsaWRfc2V0LCBiYXRjaF9zaXplPTUxMiwgc2h1ZmZsZT1GYWxzZSkKICAgICAgICAgICAg
ICAgICAgICAgICAgdG1wX3ByZWRpY3Rpb25fc2NvcmUgPSBtb2RlbC5wcmVkaWN0KHZhbGlkX2xvYWRl
cikKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAgICB0bXBfcHJl
ZGljdGlvbl9zY29yZSA9IG1vZGVsLnByZWRpY3RfcHJvYmEoc2VsZi5kYXRhZnJhbWUudmFsdWVzKQoK
ICAgICAgICAgICAgICAgICAgICBpZiBwcmVkaWN0aW9uX3Njb3JlIGlzIE5vbmU6CiAgICAgICAgICAg
ICAgICAgICAgICAgIHByZWRpY3Rpb25fc2NvcmUgPSB0bXBfcHJlZGljdGlvbl9zY29yZQogICAgICAg
ICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgIHByZWRpY3Rpb25fc2NvcmUg
Kz0gdG1wX3ByZWRpY3Rpb25fc2NvcmUKICAgICAgICAgICAgICAgIHByZWRpY3Rpb25fc2NvcmUgLz0g
bGVuKHNlbGYubW9kZWxfbGlzdCkKICAgICAgICAgICAgICAgIHNlbGYuc2NvcmUgPSBwcmVkaWN0aW9u
X3Njb3JlCgogICAgICAgICAgICAgICAgIyBkaXNwbGF5IHByZWRpY3Rpb24gc2NvcmUKICAgICAgICAg
ICAgICAgIGlmIG5vdCBzZWxmLnNjb3JlIGlzIE5vbmU6CiAgICAgICAgICAgICAgICAgICAgZGF0YSA9
IHNlbGYuc2NvcmUKICAgICAgICAgICAgICAgICAgICBzZWxmLnRyYWluX3Njb3JlX3RhYmxlV2lkZ2V0
LnNldFJvd0NvdW50KGRhdGEuc2hhcGVbMF0pCiAgICAgICAgICAgICAgICAgICAgc2VsZi50cmFpbl9z
Y29yZV90YWJsZVdpZGdldC5zZXRDb2x1bW5Db3VudChkYXRhLnNoYXBlWzFdKQogICAgICAgICAgICAg
ICAgICAgIHNlbGYudHJhaW5fc2NvcmVfdGFibGVXaWRnZXQuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVs
cyhbJ1Njb3JlIGZvciBjYXRlZ29yeSAlcycgJWkgZm9yIGkgaW4gcmFuZ2UoZGF0YS5zaGFwZVsxXSld
KQogICAgICAgICAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKGRhdGEuc2hhcGVbMF0pOgogICAgICAg
ICAgICAgICAgICAgICAgICBmb3IgaiBpbiByYW5nZShkYXRhLnNoYXBlWzFdKToKICAgICAgICAgICAg
ICAgICAgICAgICAgICAgIGNlbGwgPSBRVGFibGVXaWRnZXRJdGVtKHN0cihyb3VuZChkYXRhW2ldW2pd
LCA0KSkpCiAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZWxmLnRyYWluX3Njb3JlX3RhYmxlV2lk
Z2V0LnNldEl0ZW0oaSwgaiwgY2VsbCkKICAgICAgICAgICAgICAgICAgICBpZiBzZWxmLmRhdGFfaW5k
ZXhbJ1RyYWluaW5nX3Njb3JlJ10gaXMgTm9uZToKICAgICAgICAgICAgICAgICAgICAgICAgaW5kZXgg
PSBzZWxmLmN1cnJlbnRfZGF0YV9pbmRleAogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFf
aW5kZXhbJ1RyYWluaW5nX3Njb3JlJ10gPSBpbmRleAogICAgICAgICAgICAgICAgICAgICAgICBzZWxm
LmRhdGFUYWJsZVdpZGdldC5pbnNlcnRSb3coaW5kZXgpCiAgICAgICAgICAgICAgICAgICAgICAgIHNl
bGYuY3VycmVudF9kYXRhX2luZGV4ICs9IDEKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAg
ICAgICAgICAgICAgICAgICBpbmRleCA9IHNlbGYuZGF0YV9pbmRleFsnVHJhaW5pbmdfc2NvcmUnXQog
ICAgICAgICAgICAgICAgICAgIHNlbGYudHJhaW5pbmdfc2NvcmVfcmFkaW8gPSBRUmFkaW9CdXR0b24o
KQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0LnNldENlbGxXaWRnZXQoaW5k
ZXgsIDAsIHNlbGYudHJhaW5pbmdfc2NvcmVfcmFkaW8pCiAgICAgICAgICAgICAgICAgICAgc2VsZi5k
YXRhVGFibGVXaWRnZXQuc2V0SXRlbShpbmRleCwgMSwgUVRhYmxlV2lkZ2V0SXRlbSgnVHJhaW5pbmcg
c2NvcmUnKSkKICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFUYWJsZVdpZGdldC5zZXRJdGVtKGlu
ZGV4LCAyLCBRVGFibGVXaWRnZXRJdGVtKHN0cihkYXRhLnNoYXBlKSkpCiAgICAgICAgICAgICAgICAg
ICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuc2V0SXRlbShpbmRleCwgMywgUVRhYmxlV2lkZ2V0SXRlbSgn
TkEnKSkKCiAgICAgICAgICAgICAgICAjIGNhbGN1bGF0ZSBhbmQgZGlzcGxheSBldmFsdWF0aW9uIG1l
dHJpY3MKICAgICAgICAgICAgICAgIGNvbHVtbl9uYW1lID0gWydTbicsICdTcCcsICdQcmUnLCAnQWNj
JywgJ01DQycsICdGMScsICdBVVJPQycsICdBVVBSQyddCiAgICAgICAgICAgICAgICBpZiBub3Qgc2Vs
Zi5zY29yZSBpcyBOb25lIGFuZCBzZWxmLnNjb3JlLnNoYXBlWzFdID09IDI6CiAgICAgICAgICAgICAg
ICAgICAgIyBjYWxjdWxhdGUgbWV0cmljcwogICAgICAgICAgICAgICAgICAgIGRhdGEgPSBzZWxmLnNj
b3JlCiAgICAgICAgICAgICAgICAgICAgbWV0cmljcyA9IE1ldHJpY3MoZGF0YVs6LCAtMV0sIHNlbGYu
ZGF0YWxhYmVsKQogICAgICAgICAgICAgICAgICAgIG1ldHJpY3NfaW5kID0gbnAuYXJyYXkoCiAgICAg
ICAgICAgICAgICAgICAgICAgIFttZXRyaWNzLnNlbnNpdGl2aXR5LCBtZXRyaWNzLnNwZWNpZmljaXR5
LCBtZXRyaWNzLnByZWNpc2lvbiwgbWV0cmljcy5hY2N1cmFjeSwgbWV0cmljcy5tY2MsCiAgICAgICAg
ICAgICAgICAgICAgICAgICBtZXRyaWNzLmYxLCBtZXRyaWNzLmF1YywgbWV0cmljcy5wcmNdKS5yZXNo
YXBlKCgxLCAtMSkpCiAgICAgICAgICAgICAgICAgICAgaW5kZXhfbmFtZSA9IFsnTWV0cmljcyB2YWx1
ZSddCiAgICAgICAgICAgICAgICAgICAgc2VsZi5hdWNEYXRhID0gWydBVVJPQyA9ICVzJyAlIG1ldHJp
Y3MuYXVjLCBtZXRyaWNzLmF1Y0RvdF0KICAgICAgICAgICAgICAgICAgICBzZWxmLnByY0RhdGEgPSBb
J0FVUFJDID0gJXMnICUgbWV0cmljcy5wcmMsIG1ldHJpY3MucHJjRG90XQogICAgICAgICAgICAgICAg
ICAgIGRlbCBtZXRyaWNzCiAgICAgICAgICAgICAgICAgICAgc2VsZi5tZXRyaWNzID0gcGQuRGF0YUZy
YW1lKG1ldHJpY3NfaW5kLCBpbmRleD1pbmRleF9uYW1lLCBjb2x1bW5zPWNvbHVtbl9uYW1lKQoKICAg
ICAgICAgICAgICAgICAgICAjIGRpc3BsYXkgbWV0cmljcwogICAgICAgICAgICAgICAgICAgIGRhdGEg
PSBzZWxmLm1ldHJpY3MudmFsdWVzCiAgICAgICAgICAgICAgICAgICAgc2VsZi5tZXRyaWNzVGFibGVX
aWRnZXQuc2V0Um93Q291bnQoZGF0YS5zaGFwZVswXSkKICAgICAgICAgICAgICAgICAgICBzZWxmLm1l
dHJpY3NUYWJsZVdpZGdldC5zZXRDb2x1bW5Db3VudChkYXRhLnNoYXBlWzFdKQogICAgICAgICAgICAg
ICAgICAgIHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0LnNldEhvcml6b250YWxIZWFkZXJMYWJlbHMoCiAg
ICAgICAgICAgICAgICAgICAgICAgIFsnU24gKCUpJywgJ1NwICglKScsICdQcmUgKCUpJywgJ0FjYyAo
JSknLCAnTUNDJywgJ0YxJywgJ0FVUk9DJywgJ0FVUFJDJ10pCiAgICAgICAgICAgICAgICAgICAgc2Vs
Zi5tZXRyaWNzVGFibGVXaWRnZXQuc2V0VmVydGljYWxIZWFkZXJMYWJlbHMoc2VsZi5tZXRyaWNzLmlu
ZGV4KQogICAgICAgICAgICAgICAgICAgIGZvciBpIGluIHJhbmdlKGRhdGEuc2hhcGVbMF0pOgogICAg
ICAgICAgICAgICAgICAgICAgICBmb3IgaiBpbiByYW5nZShkYXRhLnNoYXBlWzFdKToKICAgICAgICAg
ICAgICAgICAgICAgICAgICAgIGNlbGwgPSBRVGFibGVXaWRnZXRJdGVtKHN0cihkYXRhW2ldW2pdKSkK
ICAgICAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYubWV0cmljc1RhYmxlV2lkZ2V0LnNldEl0ZW0o
aSwgaiwgY2VsbCkKICAgICAgICAgICAgICAgICAgICBpZiBzZWxmLmRhdGFfaW5kZXhbJ01ldHJpY3Mn
XSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgICAgICBpbmRleCA9IHNlbGYuY3VycmVudF9kYXRh
X2luZGV4CiAgICAgICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YV9pbmRleFsnTWV0cmljcyddID0g
aW5kZXgKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRnZXQuaW5zZXJ0Um93
KGluZGV4KQogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmN1cnJlbnRfZGF0YV9pbmRleCArPSAx
CiAgICAgICAgICAgICAgICAgICAgZWxzZToKICAgICAgICAgICAgICAgICAgICAgICAgaW5kZXggPSBz
ZWxmLmRhdGFfaW5kZXhbJ01ldHJpY3MnXQogICAgICAgICAgICAgICAgICAgIHNlbGYubWV0cmljc19y
YWRpbyA9IFFSYWRpb0J1dHRvbigpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5kYXRhVGFibGVXaWRn
ZXQuc2V0Q2VsbFdpZGdldChpbmRleCwgMCwgc2VsZi5tZXRyaWNzX3JhZGlvKQogICAgICAgICAgICAg
ICAgICAgIHNlbGYuZGF0YVRhYmxlV2lkZ2V0LnNldEl0ZW0oaW5kZXgsIDEsIFFUYWJsZVdpZGdldEl0
ZW0oJ0V2YWx1YXRpb24gbWV0cmljcycpKQogICAgICAgICAgICAgICAgICAgIHNlbGYuZGF0YVRhYmxl
V2lkZ2V0LnNldEl0ZW0oaW5kZXgsIDIsIFFUYWJsZVdpZGdldEl0ZW0oc3RyKGRhdGEuc2hhcGUpKSkK
ICAgICAgICAgICAgICAgICAgICBzZWxmLmRhdGFUYWJsZVdpZGdldC5zZXRJdGVtKGluZGV4LCAzLCBR
VGFibGVXaWRnZXRJdGVtKCdOQScpKQoKICAgICAgICAgICAgICAgICNwbG90IFJPQwogICAgICAgICAg
ICAgICAgaWYgbm90IHNlbGYuYXVjRGF0YSBpcyBOb25lOgogICAgICAgICAgICAgICAgICAgIHNlbGYu
cm9jTGF5b3V0LnJlbW92ZVdpZGdldChzZWxmLnJvY19jdXJ2ZV93aWRnZXQpCiAgICAgICAgICAgICAg
ICAgICAgc2lwLmRlbGV0ZShzZWxmLnJvY19jdXJ2ZV93aWRnZXQpCiAgICAgICAgICAgICAgICAgICAg
c2VsZi5yb2NfY3VydmVfd2lkZ2V0ID0gUGxvdFdpZGdldHMuQ3VydmVXaWRnZXQoKQogICAgICAgICAg
ICAgICAgICAgIHNlbGYucm9jX2N1cnZlX3dpZGdldC5pbml0X2RhdGEoMCwgJ1JPQyBjdXJ2ZScsIGlu
ZF9kYXRhPXNlbGYuYXVjRGF0YSkKICAgICAgICAgICAgICAgICAgICBzZWxmLnJvY0xheW91dC5hZGRX
aWRnZXQoc2VsZi5yb2NfY3VydmVfd2lkZ2V0KQoKICAgICAgICAgICAgICAgICMgcGxvdCBQUkMKICAg
ICAgICAgICAgICAgIGlmIG5vdCBzZWxmLnByY0RhdGEgaXMgTm9uZToKICAgICAgICAgICAgICAgICAg
ICBzZWxmLnByY0xheW91dC5yZW1vdmVXaWRnZXQoc2VsZi5wcmNfY3VydmVfd2lkZ2V0KQogICAgICAg
ICAgICAgICAgICAgIHNpcC5kZWxldGUoc2VsZi5wcmNfY3VydmVfd2lkZ2V0KQogICAgICAgICAgICAg
ICAgICAgIHNlbGYucHJjX2N1cnZlX3dpZGdldCA9IFBsb3RXaWRnZXRzLkN1cnZlV2lkZ2V0KCkKICAg
ICAgICAgICAgICAgICAgICBzZWxmLnByY19jdXJ2ZV93aWRnZXQuaW5pdF9kYXRhKDEsICdQUkMgY3Vy
dmUnLCBpbmRfZGF0YT1zZWxmLnByY0RhdGEpCiAgICAgICAgICAgICAgICAgICAgc2VsZi5wcmNMYXlv
dXQuYWRkV2lkZ2V0KHNlbGYucHJjX2N1cnZlX3dpZGdldCkKICAgICAgICAgICAgZXhjZXB0IEV4Y2Vw
dGlvbiBhcyBlOgogICAgICAgICAgICAgICAgUU1lc3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9y
Jywgc3RyKGUpLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykK
ICAgICAgICBlbHNlOgogICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3In
LCAnUGxlYXNlIGxvYWQgdGhlIG1vZGVsIGZpbGUocykgb3IgZGF0YSBmaWxlLicsIFFNZXNzYWdlQm94
Lk9rIHwgUU1lc3NhZ2VCb3guTm8sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFFNZXNz
YWdlQm94Lk9rKQoKICAgIGRlZiBzYXZlX21sX2ZpbGVzKHNlbGYpOgogICAgICAgIGlmICd0cmFpbmlu
Z19zY29yZV9yYWRpbycgaW4gZGlyKHNlbGYpIGFuZCBzZWxmLnRyYWluaW5nX3Njb3JlX3JhZGlvLmlz
Q2hlY2tlZCgpOgogICAgICAgICAgICBzYXZlX2ZpbGUsIG9rID0gUUZpbGVEaWFsb2cuZ2V0U2F2ZUZp
bGVOYW1lKHNlbGYsICdTYXZlJywgJy4vZGF0YScsICdUU1YgRmlsZXMgKCoudHN2KScpCiAgICAgICAg
ICAgIGlmIG9rOgogICAgICAgICAgICAgICAgb2sxID0gc2VsZi5zYXZlX3ByZWRpY3Rpb25fc2NvcmUo
c2F2ZV9maWxlKQogICAgICAgICAgICAgICAgaWYgbm90IG9rMToKICAgICAgICAgICAgICAgICAgICBR
TWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAnU2F2ZSBmaWxlIGZhaWxlZC4nLCBRTWVz
c2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICAgICAgICAgIFFNZXNzYWdlQm94Lk9rKQogICAgICAgIGVsaWYgJ21ldHJpY3NfcmFkaW8nIGluIGRp
cihzZWxmKSBhbmQgc2VsZi5tZXRyaWNzX3JhZGlvLmlzQ2hlY2tlZCgpOgogICAgICAgICAgICBzYXZl
X2ZpbGUsIG9rID0gUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsICdTYXZlJywgJy4vZGF0
YScsICdUU1YgRmlsZXMgKCoudHN2KScpCiAgICAgICAgICAgIGlmIG9rOgogICAgICAgICAgICAgICAg
b2sxID0gc2VsZi5zYXZlX21ldHJpY3Moc2F2ZV9maWxlKQogICAgICAgICAgICAgICAgaWYgbm90IG9r
MToKICAgICAgICAgICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAn
U2F2ZSBmaWxlIGZhaWxlZC4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLAogICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIFFNZXNzYWdlQm94Lk9rKQogICAgICAgIGVs
c2U6CiAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICdFcnJvcicsICdQbGVhc2Ug
c2VsZWN0IHdoaWNoIGRhdGEgdG8gc2F2ZS4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5v
LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBRTWVzc2FnZUJveC5PaykKCiAgICBkZWYg
c2F2ZV9wcmVkaWN0aW9uX3Njb3JlKHNlbGYsIGZpbGUpOgogICAgICAgIHRyeToKICAgICAgICAgICAg
ZGYgPSBwZC5EYXRhRnJhbWUoc2VsZi5zY29yZSwgY29sdW1ucz1bJ1Njb3JlXyVzJyAlaSBmb3IgaSBp
biByYW5nZShzZWxmLnNjb3JlLnNoYXBlWzFdKV0pCiAgICAgICAgICAgIGRmLnRvX2NzdihmaWxlLCBz
ZXA9J1x0JywgaGVhZGVyPVRydWUsIGluZGV4PUZhbHNlKQogICAgICAgICAgICByZXR1cm4gVHJ1ZQog
ICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgcHJpbnQoZSkKICAgICAgICAg
ICAgcmV0dXJuIEZhbHNlCgogICAgZGVmIHNhdmVfbWV0cmljcyhzZWxmLCBmaWxlKToKICAgICAgICB0
cnk6CiAgICAgICAgICAgIHNlbGYubWV0cmljcy50b19jc3YoZmlsZSwgc2VwPSdcdCcsIGhlYWRlcj1U
cnVlLCBpbmRleD1UcnVlKQogICAgICAgICAgICByZXR1cm4gVHJ1ZQogICAgICAgIGV4Y2VwdCBFeGNl
cHRpb24gYXMgZToKICAgICAgICAgICAgcmV0dXJuIEZhbHNlCgogICAgZGVmIGNsb3NlRXZlbnQoc2Vs
ZiwgZXZlbnQpOgogICAgICAgIHJlcGx5ID0gUU1lc3NhZ2VCb3gucXVlc3Rpb24oc2VsZiwgJ0NvbmZp
cm0gRXhpdCcsICdBcmUgeW91IHN1cmUgd2FudCB0byBxdWl0IGlMZWFyblBsdXM/JywgUU1lc3NhZ2VC
b3guWWVzIHwgUU1lc3NhZ2VCb3guTm8sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICBRTWVzc2FnZUJveC5ObykKICAgICAgICBpZiByZXBseSA9PSBRTWVzc2FnZUJveC5ZZXM6CiAgICAg
ICAgICAgIHNlbGYuY2xvc2Vfc2lnbmFsLmVtaXQoJ0xvYWRNb2RlbCcpCiAgICAgICAgICAgIHNlbGYu
Y2xvc2UoKQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGlmIGV2ZW50OgogICAgICAgICAgICAgICAg
ZXZlbnQuaWdub3JlKCkKCg=="""#PVRydWUsIGluZGV4PUZhbHNlKQogICAgICAgICAgICByZKTjdyjD
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