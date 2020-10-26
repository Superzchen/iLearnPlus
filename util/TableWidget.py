#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
from PyQt5.QtWidgets import (QApplication, QTableWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QLineEdit, QTableWidgetItem, QMessageBox, QComboBox, QSpacerItem, QSizePolicy, QAbstractItemView)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
import pandas as pd
import numpy as np
import base64
from PlotWidgets import HistogramWidget
import qdarkstyle

## secret key ##################################################################
CBzeXMuZXhpdChhcHAuZXhlY18o="""#################################################
Y2xhc3MgVGFibGVXaWRnZXQoUVdpZGdldCk6CiAgICBjb250cm9sX3NpZ25hbCA9IHB5cXRTaWduYWwo
bGlzdCkKCiAgICBkZWYgX19pbml0X18oc2VsZik6CiAgICAgICAgc3VwZXIoVGFibGVXaWRnZXQsIHNl
bGYpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLmRhdGEgPSBOb25lCiAgICAgICAgc2VsZi5yb3cgPSA1
MAogICAgICAgIHNlbGYuY29sdW1uID0gMAogICAgICAgIHNlbGYucGFnZV9kaWN0ID0ge30KICAgICAg
ICBzZWxmLmZpbmFsX3BhZ2UgPSAnJwogICAgICAgIHNlbGYuaGVhZGVyID0gW10KICAgICAgICBzZWxm
LnNlbGVjdGVkX2NvbHVtbiA9IE5vbmUKICAgICAgICBzZWxmLmluaXRVSSgpCgogICAgZGVmIGluaXRV
SShzZWxmKToKICAgICAgICBzZWxmLmNvbnRyb2xfc2lnbmFsLmNvbm5lY3Qoc2VsZi5wYWdlX2NvbnRy
b2xsZXIpCiAgICAgICAgbGF5b3V0ID0gUVZCb3hMYXlvdXQoc2VsZikKICAgICAgICBzZWxmLnRhYmxl
V2lkZ2V0ID0gUVRhYmxlV2lkZ2V0KCkKICAgICAgICBzZWxmLnRhYmxlV2lkZ2V0LnNldEVkaXRUcmln
Z2VycyhRQWJzdHJhY3RJdGVtVmlldy5Ob0VkaXRUcmlnZ2VycykKICAgICAgICBzZWxmLnRhYmxlV2lk
Z2V0Lmhvcml6b250YWxIZWFkZXIoKS5zZWN0aW9uQ2xpY2tlZC5jb25uZWN0KHNlbGYuaG9yaXpvbnRh
bF9oZWFkZXJfY2xpY2tlZCkKICAgICAgICBzZWxmLnRhYmxlV2lkZ2V0LnNldEZvbnQoUUZvbnQoJ0Fy
aWFsJywgOCkpCiAgICAgICAgbGF5b3V0LmFkZFdpZGdldChzZWxmLnRhYmxlV2lkZ2V0KQoKICAgICAg
ICBjb250cm9sX2xheW91dCA9IFFIQm94TGF5b3V0KCkKICAgICAgICBob21lUGFnZSA9IFFQdXNoQnV0
dG9uKCJIb21lIikKICAgICAgICBwcmVQYWdlID0gUVB1c2hCdXR0b24oIjwgUHJldmlvdXMiKQogICAg
ICAgIHNlbGYuY3VyUGFnZSA9IFFMYWJlbCgiMSIpCiAgICAgICAgbmV4dFBhZ2UgPSBRUHVzaEJ1dHRv
bigiTmV4dCA+IikKICAgICAgICBmaW5hbFBhZ2UgPSBRUHVzaEJ1dHRvbigiRmluYWwiKQogICAgICAg
IHNlbGYudG90YWxQYWdlID0gUUxhYmVsKCJUb3RhbCIpCiAgICAgICAgc2tpcExhYmxlXzAgPSBRTGFi
ZWwoIkp1bXAgdG8iKQogICAgICAgIHNlbGYuc2tpcFBhZ2UgPSBRTGluZUVkaXQoKQogICAgICAgIHNl
bGYuc2tpcFBhZ2Uuc2V0TWF4aW11bVdpZHRoKDUwKQogICAgICAgIGNvbmZpcm1Ta2lwID0gUVB1c2hC
dXR0b24oIk9LIikKICAgICAgICBzcGFjZXIgPSBRU3BhY2VySXRlbSgyMCwgMTAsIFFTaXplUG9saWN5
LkV4cGFuZGluZywgUVNpemVQb2xpY3kuTWluaW11bSkKICAgICAgICBzZWxmLmxhYmVsX2NvbWJvQm94
ID0gUUNvbWJvQm94KCkKICAgICAgICBzZWxmLmxhYmVsX2NvbWJvQm94LmFkZEl0ZW0oJ0FsbCcpCiAg
ICAgICAgc2VsZi5sYWJlbF9jb21ib0JveC5zZXRDdXJyZW50SW5kZXgoMCkKICAgICAgICBzZWxmLmxh
YmVsX2NvbWJvQm94LnNldFRvb2xUaXAoJ1NlbGVjdCB0aGUgc2FtcGxlIGNhdGVnb3J5IHVzZWQgdG8g
ZHJhdyB0aGUgaGlzdG9ncmFtLicpCiAgICAgICAgc2VsZi5kaXNwbGF5X2hpc3QgPSBRUHVzaEJ1dHRv
bignIEhpc3RvZ3JhbSAnKQogICAgICAgIHNlbGYuZGlzcGxheV9oaXN0LmNsaWNrZWQuY29ubmVjdChz
ZWxmLmRpc3BsYXlfaGlzdG9ncmFtKQoKICAgICAgICBjb250cm9sX2xheW91dC5hZGRTdHJldGNoKDEp
CiAgICAgICAgaG9tZVBhZ2UuY2xpY2tlZC5jb25uZWN0KHNlbGYuX19ob21lX3BhZ2UpCiAgICAgICAg
cHJlUGFnZS5jbGlja2VkLmNvbm5lY3Qoc2VsZi5fX3ByZV9wYWdlKQogICAgICAgIG5leHRQYWdlLmNs
aWNrZWQuY29ubmVjdChzZWxmLl9fbmV4dF9wYWdlKQogICAgICAgIGZpbmFsUGFnZS5jbGlja2VkLmNv
bm5lY3Qoc2VsZi5fX2ZpbmFsX3BhZ2UpCiAgICAgICAgY29uZmlybVNraXAuY2xpY2tlZC5jb25uZWN0
KHNlbGYuX19jb25maXJtX3NraXApCiAgICAgICAgY29udHJvbF9sYXlvdXQuYWRkV2lkZ2V0KGhvbWVQ
YWdlKQogICAgICAgIGNvbnRyb2xfbGF5b3V0LmFkZFdpZGdldChwcmVQYWdlKQogICAgICAgIGNvbnRy
b2xfbGF5b3V0LmFkZFdpZGdldChzZWxmLmN1clBhZ2UpCiAgICAgICAgY29udHJvbF9sYXlvdXQuYWRk
V2lkZ2V0KG5leHRQYWdlKQogICAgICAgIGNvbnRyb2xfbGF5b3V0LmFkZFdpZGdldChmaW5hbFBhZ2Up
CiAgICAgICAgY29udHJvbF9sYXlvdXQuYWRkV2lkZ2V0KHNlbGYudG90YWxQYWdlKQogICAgICAgIGNv
bnRyb2xfbGF5b3V0LmFkZFdpZGdldChza2lwTGFibGVfMCkKICAgICAgICBjb250cm9sX2xheW91dC5h
ZGRXaWRnZXQoc2VsZi5za2lwUGFnZSkKICAgICAgICBjb250cm9sX2xheW91dC5hZGRXaWRnZXQoY29u
ZmlybVNraXApCiAgICAgICAgY29udHJvbF9sYXlvdXQuYWRkSXRlbShzcGFjZXIpCiAgICAgICAgY29u
dHJvbF9sYXlvdXQuYWRkV2lkZ2V0KHNlbGYuZGlzcGxheV9oaXN0KQogICAgICAgIGNvbnRyb2xfbGF5
b3V0LmFkZFN0cmV0Y2goMSkKICAgICAgICBsYXlvdXQuYWRkTGF5b3V0KGNvbnRyb2xfbGF5b3V0KQoK
ICAgIGRlZiBzZXRSb3dBbmRDb2x1bW5zKHNlbGYsIHJvdywgY29sdW1uKToKICAgICAgICBzZWxmLnJv
dyA9IHJvdwogICAgICAgIHNlbGYuY29sdW1uID0gY29sdW1uCiAgICAgICAgc2VsZi50YWJsZVdpZGdl
dC5zZXRSb3dDb3VudChyb3cpCiAgICAgICAgc2VsZi50YWJsZVdpZGdldC5zZXRDb2x1bW5Db3VudChj
b2x1bW4pCgogICAgZGVmIGluaXRfZGF0YShzZWxmLCBoZWFkZXIsIGRhdGEpOgogICAgICAgIHNlbGYu
dGFibGVXaWRnZXQuc2V0SG9yaXpvbnRhbEhlYWRlckxhYmVscyhoZWFkZXIpCiAgICAgICAgc2VsZi5o
ZWFkZXIgPSBoZWFkZXIKICAgICAgICBzZWxmLmRhdGEgPSBkYXRhCiAgICAgICAgcGFnZSA9IDEKICAg
ICAgICBmb3IgaSBpbiByYW5nZSgwLCBzZWxmLmRhdGEuc2hhcGVbMF0sIHNlbGYucm93KToKICAgICAg
ICAgICAgZW5kID0gaSArIHNlbGYucm93IGlmIGkgKyBzZWxmLnJvdyA8IHNlbGYuZGF0YS5zaGFwZVsw
XSBlbHNlIHNlbGYuZGF0YS5zaGFwZVswXQogICAgICAgICAgICBzZWxmLnBhZ2VfZGljdFtzdHIocGFn
ZSldID0gKGksIGVuZCkKICAgICAgICAgICAgc2VsZi5maW5hbF9wYWdlID0gc3RyKHBhZ2UpCiAgICAg
ICAgICAgIHBhZ2UgKz0gMQogICAgICAgIHNlbGYuY3VyUGFnZS5zZXRUZXh0KCcxJykKICAgICAgICBz
ZWxmLmRpc3BsYXlfdGFibGUoJzEnKQogICAgICAgIHNlbGYudG90YWxQYWdlLnNldFRleHQoJ1RvdGFs
IHBhZ2U6ICVzICcgJXNlbGYuZmluYWxfcGFnZSkKCiAgICBkZWYgX19ob21lX3BhZ2Uoc2VsZik6CiAg
ICAgICAgc2VsZi5jb250cm9sX3NpZ25hbC5lbWl0KFsiaG9tZSIsIHNlbGYuY3VyUGFnZS50ZXh0KCld
KQoKICAgIGRlZiBfX3ByZV9wYWdlKHNlbGYpOgogICAgICAgIHNlbGYuY29udHJvbF9zaWduYWwuZW1p
dChbInByZSIsIHNlbGYuY3VyUGFnZS50ZXh0KCldKQoKICAgIGRlZiBfX25leHRfcGFnZShzZWxmKToK
ICAgICAgICBzZWxmLmNvbnRyb2xfc2lnbmFsLmVtaXQoWyJuZXh0Iiwgc2VsZi5jdXJQYWdlLnRleHQo
KV0pCgogICAgZGVmIF9fZmluYWxfcGFnZShzZWxmKToKICAgICAgICBzZWxmLmNvbnRyb2xfc2lnbmFs
LmVtaXQoWyJmaW5hbCIsIHNlbGYuY3VyUGFnZS50ZXh0KCldKQoKICAgIGRlZiBfX2NvbmZpcm1fc2tp
cChzZWxmKToKICAgICAgICBzZWxmLmNvbnRyb2xfc2lnbmFsLmVtaXQoWyJjb25maXJtIiwgc2VsZi5z
a2lwUGFnZS50ZXh0KCldKQoKICAgIGRlZiBwYWdlX2NvbnRyb2xsZXIoc2VsZiwgc2lnbmFsKToKICAg
ICAgICB0cnk6CiAgICAgICAgICAgIGlmIG5vdCBzZWxmLmRhdGEgaXMgTm9uZToKICAgICAgICAgICAg
ICAgIGlmICdob21lJyA9PSBzaWduYWxbMF0gYW5kIHNpZ25hbFsxXSAhPSAnJzoKICAgICAgICAgICAg
ICAgICAgICBzZWxmLmRpc3BsYXlfdGFibGUoJzEnKQogICAgICAgICAgICAgICAgICAgIHNlbGYuY3Vy
UGFnZS5zZXRUZXh0KCcxJykKICAgICAgICAgICAgICAgIGVsaWYgJ2ZpbmFsJyA9PSBzaWduYWxbMF0g
YW5kIHNpZ25hbFsxXSAhPSAnJzoKICAgICAgICAgICAgICAgICAgICBzZWxmLmRpc3BsYXlfdGFibGUo
c2VsZi5maW5hbF9wYWdlKQogICAgICAgICAgICAgICAgICAgIHNlbGYuY3VyUGFnZS5zZXRUZXh0KHNl
bGYuZmluYWxfcGFnZSkKICAgICAgICAgICAgICAgIGVsaWYgJ3ByZScgPT0gc2lnbmFsWzBdIGFuZCBz
aWduYWxbMV0gIT0gJyc6CiAgICAgICAgICAgICAgICAgICAgcGFnZSA9IGludChzaWduYWxbMV0pIC0g
MSBpZiBpbnQoc2lnbmFsWzFdKSAtIDEgPiAwIGVsc2UgMQogICAgICAgICAgICAgICAgICAgIHNlbGYu
Y3VyUGFnZS5zZXRUZXh0KHN0cihwYWdlKSkKICAgICAgICAgICAgICAgICAgICBzZWxmLmRpc3BsYXlf
dGFibGUoc3RyKHBhZ2UpKQogICAgICAgICAgICAgICAgZWxpZiAnbmV4dCcgPT0gc2lnbmFsWzBdIGFu
ZCBzaWduYWxbMV0gIT0gJyc6CiAgICAgICAgICAgICAgICAgICAgcGFnZSA9IGludChzaWduYWxbMV0p
ICsgMSBpZiBpbnQoc2lnbmFsWzFdKSArIDEgPD0gaW50KHNlbGYuZmluYWxfcGFnZSkgZWxzZSBpbnQo
c2VsZi5maW5hbF9wYWdlKQogICAgICAgICAgICAgICAgICAgIHNlbGYuY3VyUGFnZS5zZXRUZXh0KHN0
cihwYWdlKSkKICAgICAgICAgICAgICAgICAgICBzZWxmLmRpc3BsYXlfdGFibGUoc3RyKHBhZ2UpKQog
ICAgICAgICAgICAgICAgZWxpZiAiY29uZmlybSIgPT0gc2lnbmFsWzBdIGFuZCBzaWduYWxbMV0gIT0g
Jyc6CiAgICAgICAgICAgICAgICAgICAgaWYgMSA8PSBpbnQoc2lnbmFsWzFdKSA8PSBpbnQoc2VsZi5m
aW5hbF9wYWdlKToKICAgICAgICAgICAgICAgICAgICAgICAgc2VsZi5jdXJQYWdlLnNldFRleHQoc2ln
bmFsWzFdKQogICAgICAgICAgICAgICAgICAgICAgICBzZWxmLmRpc3BsYXlfdGFibGUoc2lnbmFsWzFd
KQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgcGFzcwoKICAgIGRlZiBk
aXNwbGF5X3RhYmxlKHNlbGYsIHBhZ2UpOgogICAgICAgIGlmIHBhZ2UgaW4gc2VsZi5wYWdlX2RpY3Q6
CiAgICAgICAgICAgIHRtcF9kYXRhID0gc2VsZi5kYXRhW3NlbGYucGFnZV9kaWN0W3BhZ2VdWzBdOiBz
ZWxmLnBhZ2VfZGljdFtwYWdlXVsxXV0KICAgICAgICAgICAgc2VsZi50YWJsZVdpZGdldC5zZXRSb3dD
b3VudCh0bXBfZGF0YS5zaGFwZVswXSkKICAgICAgICAgICAgc2VsZi50YWJsZVdpZGdldC5zZXRDb2x1
bW5Db3VudCh0bXBfZGF0YS5zaGFwZVsxXSkKICAgICAgICAgICAgc2VsZi50YWJsZVdpZGdldC5zZXRI
b3Jpem9udGFsSGVhZGVyTGFiZWxzKHNlbGYuaGVhZGVyKQogICAgICAgICAgICBmb3IgaSBpbiByYW5n
ZSh0bXBfZGF0YS5zaGFwZVswXSk6CiAgICAgICAgICAgICAgICBmb3IgaiBpbiByYW5nZSh0bXBfZGF0
YS5zaGFwZVsxXSk6CiAgICAgICAgICAgICAgICAgICAgaWYgaiA9PSAwOgogICAgICAgICAgICAgICAg
ICAgICAgICBjZWxsID0gUVRhYmxlV2lkZ2V0SXRlbShzdHIodG1wX2RhdGFbaV1bal0pKQogICAgICAg
ICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgIGNlbGwgPSBRVGFibGVXaWRn
ZXRJdGVtKHN0cihyb3VuZChmbG9hdCh0bXBfZGF0YVtpXVtqXSksIDYpKSkKICAgICAgICAgICAgICAg
ICAgICBzZWxmLnRhYmxlV2lkZ2V0LnNldEl0ZW0oaSwgaiwgY2VsbCkKICAgICAgICBlbHNlOgogICAg
ICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAnUGFnZSBudW1iZXIgb3V0
IG9mIGluZGV4LicsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9r
KQogICAgICAgIGxhYmVscyA9IFsnQWxsJ10gKyBbc3RyKGkpIGZvciBpIGluIHNldChzZWxmLmRhdGFb
MTosIDFdKV0KICAgICAgICBzZWxmLmxhYmVsX2NvbWJvQm94LmNsZWFyKCkKICAgICAgICBzZWxmLmxh
YmVsX2NvbWJvQm94LmFkZEl0ZW1zKGxhYmVscykKICAgICAgICBzZWxmLmxhYmVsX2NvbWJvQm94LnNl
dEN1cnJlbnRJbmRleCgwKQoKICAgIGRlZiBob3Jpem9udGFsX2hlYWRlcl9jbGlja2VkKHNlbGYsIGlu
ZGV4KToKICAgICAgICBpZiBpbmRleCA9PSAwOgogICAgICAgICAgICBzZWxmLnNlbGVjdGVkX2NvbHVt
biA9IE5vbmUKICAgICAgICBlbHNlOgogICAgICAgICAgICBzZWxmLnNlbGVjdGVkX2NvbHVtbiA9IGlu
ZGV4CgogICAgZGVmIGRpc3BsYXlfaGlzdG9ncmFtKHNlbGYpOgogICAgICAgIGlmIHNlbGYuc2VsZWN0
ZWRfY29sdW1uIGlzIE5vbmU6CiAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICdF
cnJvcicsICdQbGVhc2Ugc2VsZWN0IGEgY29sdW1uIChleGNlcHQgY29sIDApKS4nLCBRTWVzc2FnZUJv
eC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykKICAgICAgICBlbGlmIHNlbGYuc2Vs
ZWN0ZWRfY29sdW1uID4gbGVuKHNlbGYuaGVhZGVyKSAtIDE6CiAgICAgICAgICAgIFFNZXNzYWdlQm94
LmNyaXRpY2FsKHNlbGYsICdFcnJvcicsICdJbmNvcnJlY3QgY29sdW1uIGluZGV4LicsIFFNZXNzYWdl
Qm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9rKQogICAgICAgIGVsc2U6CiAgICAg
ICAgICAgIGRhdGEgPSBucC5oc3RhY2soKHNlbGYuZGF0YVsxOiwgMV0ucmVzaGFwZSgoLTEsIDEpKSwg
c2VsZi5kYXRhWzE6LCBzZWxmLnNlbGVjdGVkX2NvbHVtbl0ucmVzaGFwZSgoLTEsIDEpKSkpLmFzdHlw
ZShmbG9hdCkKICAgICAgICAgICAgc2VsZi5oaXN0ID0gSGlzdG9ncmFtV2lkZ2V0KCkKICAgICAgICAg
ICAgc2VsZi5oaXN0LmluaXRfZGF0YShzZWxmLmhlYWRlcltzZWxmLnNlbGVjdGVkX2NvbHVtbl0sIGRh
dGEpCiAgICAgICAgICAgIHNlbGYuaGlzdC5zZXRTdHlsZVNoZWV0KHFkYXJrc3R5bGUubG9hZF9zdHls
ZXNoZWV0X3B5cXQ1KCkpCiAgICAgICAgICAgIHNlbGYuaGlzdC5zaG93KCkKCmNsYXNzIFRhYmxlV2lk
Z2V0Rm9yU2VsUGFuZWwoVGFibGVXaWRnZXQpOgogICAgZGVmIF9faW5pdF9fKHNlbGYpOgogICAgICAg
IHN1cGVyKFRhYmxlV2lkZ2V0Rm9yU2VsUGFuZWwsIHNlbGYpLl9faW5pdF9fKCkKCiAgICBkZWYgZGlz
cGxheV90YWJsZShzZWxmLCBwYWdlKToKICAgICAgICBpZiBwYWdlIGluIHNlbGYucGFnZV9kaWN0Ogog
ICAgICAgICAgICB0bXBfZGF0YSA9IHNlbGYuZGF0YVtzZWxmLnBhZ2VfZGljdFtwYWdlXVswXTogc2Vs
Zi5wYWdlX2RpY3RbcGFnZV1bMV1dCiAgICAgICAgICAgIHNlbGYudGFibGVXaWRnZXQuc2V0Um93Q291
bnQodG1wX2RhdGEuc2hhcGVbMF0pCiAgICAgICAgICAgIHNlbGYudGFibGVXaWRnZXQuc2V0Q29sdW1u
Q291bnQodG1wX2RhdGEuc2hhcGVbMV0pCiAgICAgICAgICAgIHNlbGYudGFibGVXaWRnZXQuc2V0SG9y
aXpvbnRhbEhlYWRlckxhYmVscyhzZWxmLmhlYWRlcikKICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2Uo
dG1wX2RhdGEuc2hhcGVbMF0pOgogICAgICAgICAgICAgICAgZm9yIGogaW4gcmFuZ2UodG1wX2RhdGEu
c2hhcGVbMV0pOgogICAgICAgICAgICAgICAgICAgIGlmIGogPT0gMDoKICAgICAgICAgICAgICAgICAg
ICAgICAgY2VsbCA9IFFUYWJsZVdpZGdldEl0ZW0oc3RyKHRtcF9kYXRhW2ldW2pdKSkKICAgICAgICAg
ICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAgICAgICBjZWxsID0gUVRhYmxlV2lkZ2V0
SXRlbShzdHIocm91bmQoZmxvYXQodG1wX2RhdGFbaV1bal0pLCA2KSkpCiAgICAgICAgICAgICAgICAg
ICAgc2VsZi50YWJsZVdpZGdldC5zZXRJdGVtKGksIGosIGNlbGwpCiAgICAgICAgZWxzZToKICAgICAg
ICAgICAgUU1lc3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgJ1BhZ2UgbnVtYmVyIG91dCBv
ZiBpbmRleC4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykK
ICAgICAgICBsYWJlbHMgPSBbJ0FsbCddICsgW3N0cihpKSBmb3IgaSBpbiBzZXQoc2VsZi5kYXRhWzE6
LCAwXSldCiAgICAgICAgc2VsZi5sYWJlbF9jb21ib0JveC5jbGVhcigpCiAgICAgICAgc2VsZi5sYWJl
bF9jb21ib0JveC5hZGRJdGVtcyhsYWJlbHMpCiAgICAgICAgc2VsZi5sYWJlbF9jb21ib0JveC5zZXRD
dXJyZW50SW5kZXgoMCkKCiAgICBkZWYgaG9yaXpvbnRhbF9oZWFkZXJfY2xpY2tlZChzZWxmLCBpbmRl
eCk6CiAgICAgICAgc2VsZi5zZWxlY3RlZF9jb2x1bW4gPSBpbmRleAoKICAgIGRlZiBkaXNwbGF5X2hp
c3RvZ3JhbShzZWxmKToKICAgICAgICBpZiBzZWxmLnNlbGVjdGVkX2NvbHVtbiBpcyBOb25lOgogICAg
ICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCAnUGxlYXNlIHNlbGVjdCBh
IGNvbHVtbi4nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykK
ICAgICAgICBlbGlmIHNlbGYuc2VsZWN0ZWRfY29sdW1uID4gbGVuKHNlbGYuaGVhZGVyKSAtIDE6CiAg
ICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYsICdFcnJvcicsICdJbmNvcnJlY3QgY29s
dW1uIGluZGV4LicsIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9r
KQogICAgICAgIGVsc2U6CiAgICAgICAgICAgIGRhdGEgPSBucC5oc3RhY2soKHNlbGYuZGF0YVs6LCAw
XS5yZXNoYXBlKCgtMSwgMSkpLCBzZWxmLmRhdGFbOiwgc2VsZi5zZWxlY3RlZF9jb2x1bW5dLnJlc2hh
cGUoKC0xLCAxKSkpKS5hc3R5cGUoZmxvYXQpCiAgICAgICAgICAgIHNlbGYuaGlzdCA9IEhpc3RvZ3Jh
bVdpZGdldCgpCiAgICAgICAgICAgIHNlbGYuaGlzdC5pbml0X2RhdGEoc2VsZi5oZWFkZXJbc2VsZi5z
ZWxlY3RlZF9jb2x1bW5dLCBkYXRhKQogICAgICAgICAgICBzZWxmLmhpc3Quc2V0U3R5bGVTaGVldChx
ZGFya3N0eWxlLmxvYWRfc3R5bGVzaGVldF9weXF0NSgpKQogICAgICAgICAgICBzZWxmLmhpc3Quc2hv
dygpCgo="""#U6CiAgICAgICAgICAgIGRhdGEgPSBucC5oc3RhY2soKHNlbGYuZGF0YVs6LCAwiSlQDl
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

class Tablewidget(QWidget):
    def __init__(self):
        super(TableWidget, self).__init__()
        self.data = None
        self.row = 50
        self.column = 0
        self.page_dict = {}
        self.final_page = ''
        self.header = []
        self.selected_column = None
        self.initUI()

    def initUI(self):        
        layout = QVBoxLayout(self)
        self.tableWidget = QTableWidget()
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)        
        self.tableWidget.setFont(QFont('Arial', 8))
        layout.addWidget(self.tableWidget)

        control_layout = QHBoxLayout()
        homePage = QPushButton("Home")
        prePage = QPushButton("< Previous")
        self.curPage = QLabel("1")
        nextPage = QPushButton("Next >")
        finalPage = QPushButton("Final")
        self.totalPage = QLabel("Total")
        skipLable_0 = QLabel("Jump to")
        self.skipPage = QLineEdit()
        self.skipPage.setMaximumWidth(50)
        confirmSkip = QPushButton("OK")
        spacer = QSpacerItem(20, 10, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.label_comboBox = QComboBox()
        self.label_comboBox.addItem('All')
        self.label_comboBox.setCurrentIndex(0)
        self.label_comboBox.setToolTip('Select the sample category used to draw the histogram.')
        self.display_hist = QPushButton(' Histogram ')        
        control_layout.addStretch(1)        
        control_layout.addWidget(homePage)
        control_layout.addWidget(prePage)
        control_layout.addWidget(self.curPage)
        control_layout.addWidget(nextPage)
        control_layout.addWidget(finalPage)
        control_layout.addWidget(self.totalPage)
        control_layout.addWidget(skipLable_0)
        control_layout.addWidget(self.skipPage)
        control_layout.addWidget(confirmSkip)
        control_layout.addItem(spacer)
        control_layout.addWidget(self.display_hist)
        control_layout.addStretch(1)
        layout.addLayout(control_layout)

    def setRowAndColumns(self, row, column):
        self.row = row
        self.column = column
        self.tableWidget.setRowCount(row)
        self.tableWidget.setColumnCount(column)

 
class TablewidgetForSelPanel(TableWidget):
    def __init__(self):
        super(TablewidgetForSelPanel, self).__init__()   


if __name__ == '__main__':
    df = pd.read_csv('mutitask.tsv', delimiter='\t', header=0)
    app = QApplication(sys.argv)
    win = TableWidget()
    win.setRowAndColumns(40, df.values.shape[1])
    win.setTabletHeader(df.columns)
    win.init_data(df.values)

    win.show()
    sys.exit(app.exec_())
