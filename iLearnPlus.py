#!/usr/bin/env python
# _*_ coding: utf-8 _*_
################################################################################
################################################################################
#  _  _                                _____   _               __     ___      #
# (_)| |                              |  __ \ | |             /_ |   / _ \     #
#  _ | |      ___   __ _  _ __  _ __  | |__) || | _   _  ___   | |  | | | |    #
# | || |     / _ \ / _` || '__|| '_ \ |  ___/ | || | | |/ __|  | |  | | | |    #
# | || |____|  __/| (_| || |   | | | || |     | || |_| |\__ \  | | _| |_| |    #
# |_||______|\___| \__,_||_|   |_| |_||_|     |_| \__,_||___/  |_|(_)\___/     #
################################################################################
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QDesktopWidget, 
                            QLabel, QHBoxLayout, QMessageBox, QAction, QFileDialog)
from PyQt5.QtGui import QIcon, QFont, QPixmap, QCloseEvent, QDesktopServices
from PyQt5.QtCore import Qt, QUrl
from util import Modules, InputDialog, PlotWidgets, MachineLearning
import iLearnPlusBasic, iLearnPlusEstimator, iLearnPlusAutoML, iLearnPlusLoadModel
import base64
import qdarkstyle
import threading
import pandas as pd
import numpy as np

## secret key ##################################################################
CBzeXMuZXhpdChhcHAuZXhlY18o="""#################################################
Y2xhc3MgTWFpbldpbmRvdyhRTWFpbldpbmRvdyk6CiAgICBkZWYgX19pbml0X18oc2VsZik6CiAgICAg
ICAgc3VwZXIoTWFpbldpbmRvdywgc2VsZikuX19pbml0X18oKQogICAgICAgIHNlbGYuaW5pdFVJKCkK
CiAgICBkZWYgaW5pdFVJKHNlbGYpOgogICAgICAgIHNlbGYuc2V0V2luZG93VGl0bGUoJ2lMZWFyblBs
dXMnKQogICAgICAgICMgc2VsZi5yZXNpemUoNzUwLCA1MDApCiAgICAgICAgc2VsZi5zZXRNYXhpbXVt
U2l6ZSg2MDAsIDQwMCkKICAgICAgICBzZWxmLnNldE1pbmltdW1TaXplKDYwMCwgNDAwKQogICAgICAg
IHNlbGYuc2V0V2luZG93SWNvbihRSWNvbignaW1hZ2VzL2xvZ28uaWNvJykpCiAgICAgICAgc2VsZi5z
ZXRGb250KFFGb250KCdBcmlhbCcpKQogICAgICAgIGJhciA9IHNlbGYubWVudUJhcigpCiAgICAgICAg
YXBwID0gYmFyLmFkZE1lbnUoJ0FwcGxpY2F0aW9ucycpCiAgICAgICAgYmFzaWMgPSBRQWN0aW9uKCdp
TGVhcm5QbHVzIEJhc2ljJywgc2VsZikKICAgICAgICBiYXNpYy50cmlnZ2VyZWQuY29ubmVjdChzZWxm
Lm9wZW5CYXNpY1dpbmRvdykKICAgICAgICBlc3RpbWF0b3IgPSBRQWN0aW9uKCdpTGVhcm5QbHVzIEVz
dGltYXRvcicsIHNlbGYpCiAgICAgICAgZXN0aW1hdG9yLnRyaWdnZXJlZC5jb25uZWN0KHNlbGYub3Bl
bkVzdGltYXRvcldpbmRvdykKICAgICAgICBhdXRvTUwgPSBRQWN0aW9uKCdpTGVhcm5QbHVzIEF1dG9N
TCcsIHNlbGYpCiAgICAgICAgYXV0b01MLnRyaWdnZXJlZC5jb25uZWN0KHNlbGYub3Blbk1MV2luZG93
KQogICAgICAgIGxvYWRNb2RlbCA9IFFBY3Rpb24oJ0xvYWQgbW9kZWwocyknLCBzZWxmKQogICAgICAg
IGxvYWRNb2RlbC50cmlnZ2VyZWQuY29ubmVjdChzZWxmLm9wZW5Mb2FkTW9kZWxXaW5kb3cpCiAgICAg
ICAgcXVpdCA9IFFBY3Rpb24oJ0V4aXQnLCBzZWxmKQogICAgICAgIHF1aXQudHJpZ2dlcmVkLmNvbm5l
Y3Qoc2VsZi5jbG9zZUV2ZW50KQogICAgICAgIGFwcC5hZGRBY3Rpb24oYmFzaWMpCiAgICAgICAgYXBw
LmFkZEFjdGlvbihlc3RpbWF0b3IpCiAgICAgICAgYXBwLmFkZEFjdGlvbihhdXRvTUwpCiAgICAgICAg
YXBwLmFkZFNlcGFyYXRvcigpCiAgICAgICAgYXBwLmFkZEFjdGlvbihsb2FkTW9kZWwpCiAgICAgICAg
YXBwLmFkZFNlcGFyYXRvcigpCiAgICAgICAgYXBwLmFkZEFjdGlvbihxdWl0KQoKICAgICAgICB2aXN1
YWwgPSBiYXIuYWRkTWVudSgnVmlzdWFsaXphdGlvbicpCiAgICAgICAgcm9jID0gUUFjdGlvbignUGxv
dCBST0MgY3VydmUnLCBzZWxmKQogICAgICAgIHJvYy50cmlnZ2VyZWQuY29ubmVjdChsYW1iZGE6IHNl
bGYucGxvdEN1cnZlKCdST0MnKSkKICAgICAgICBwcmMgPSBRQWN0aW9uKCdQbG90IFBSQyBjdXJ2ZScs
IHNlbGYpCiAgICAgICAgcHJjLnRyaWdnZXJlZC5jb25uZWN0KGxhbWJkYTogc2VsZi5wbG90Q3VydmUo
J1BSQycpKQogICAgICAgIGJveHBsb3QgPSBRQWN0aW9uKCdCb3hwbG90Jywgc2VsZikKICAgICAgICBi
b3hwbG90LnRyaWdnZXJlZC5jb25uZWN0KHNlbGYuZHJhd0JveHBsb3QpCiAgICAgICAgaGVhdG1hcCA9
IFFBY3Rpb24oJ0hlYXRtYXAnLCBzZWxmKQogICAgICAgIGhlYXRtYXAudHJpZ2dlcmVkLmNvbm5lY3Qo
c2VsZi5kcmF3SGVhdG1hcCkKICAgICAgICBzY2F0dGVyUGxvdCA9IFFBY3Rpb24oJ1NjYXR0ZXIgcGxv
dCcsIHNlbGYpCiAgICAgICAgc2NhdHRlclBsb3QudHJpZ2dlcmVkLmNvbm5lY3Qoc2VsZi5zY2F0dGVy
UGxvdCkKICAgICAgICBkYXRhID0gUUFjdGlvbignRGlzdHJpYnV0aW9uIHZpc3VhbGl6YXRpb24nLCBz
ZWxmKQogICAgICAgIGRhdGEudHJpZ2dlcmVkLmNvbm5lY3Qoc2VsZi5kaXNwbGF5SGlzdCkKICAgICAg
ICB2aXN1YWwuYWRkQWN0aW9ucyhbcm9jLCBwcmMsIGJveHBsb3QsIGhlYXRtYXAsIHNjYXR0ZXJQbG90
LCBkYXRhXSkKCiAgICAgICAgdG9vbHMgPSBiYXIuYWRkTWVudSgnVG9vbHMnKSAgICAgICAgCiAgICAg
ICAgZmlsZVRGID0gUUFjdGlvbignRmlsZSBmb3JtYXQgdHJhbnNmb3JtYXRpb24nLCBzZWxmKQogICAg
ICAgIGZpbGVURi50cmlnZ2VyZWQuY29ubmVjdChzZWxmLm9wZW5GaWxlVEYpCiAgICAgICAgbWVyZ2VG
aWxlID0gUUFjdGlvbignTWVyZ2UgY29kaW5nIGZpbGVzIGludG8gb25lJywgc2VsZikKICAgICAgICBt
ZXJnZUZpbGUudHJpZ2dlcmVkLmNvbm5lY3Qoc2VsZi5tZXJnZUNvZGluZ0ZpbGVzKQogICAgICAgIHRv
b2xzLmFkZEFjdGlvbnMoW2ZpbGVURiwgbWVyZ2VGaWxlXSkKCiAgICAgICAgaGVscCA9IGJhci5hZGRN
ZW51KCdIZWxwJykKICAgICAgICBkb2N1bWVudCA9IFFBY3Rpb24oJ0RvY3VtZW50Jywgc2VsZikKICAg
ICAgICBkb2N1bWVudC50cmlnZ2VyZWQuY29ubmVjdChzZWxmLm9wZW5Eb2N1bWVudFVybCkKICAgICAg
ICBhYm91dCA9IFFBY3Rpb24oJ0Fib3V0Jywgc2VsZikKICAgICAgICBhYm91dC50cmlnZ2VyZWQuY29u
bmVjdChzZWxmLm9wZW5BYm91dCkKICAgICAgICBoZWxwLmFkZEFjdGlvbnMoW2RvY3VtZW50LCBhYm91
dF0pCgogICAgICAgICMgbW92ZSB3aW5kb3cgdG8gY2VudGVyCiAgICAgICAgc2VsZi5tb3ZlQ2VudGVy
KCkKCiAgICAgICAgc2VsZi53aWRnZXQgPSBRV2lkZ2V0KCkKICAgICAgICBoTGF5b3V0ID0gUUhCb3hM
YXlvdXQoc2VsZi53aWRnZXQpCiAgICAgICAgaExheW91dC5zZXRBbGlnbm1lbnQoUXQuQWxpZ25DZW50
ZXIpCiAgICAgICAgbGFiZWwgPSBRTGFiZWwoKQogICAgICAgICMgbGFiZWwuc2V0TWF4aW11bVdpZHRo
KDYwMCkKICAgICAgICBsYWJlbC5zZXRQaXhtYXAoUVBpeG1hcCgnaW1hZ2VzL2xvZ28ucG5nJykpCiAg
ICAgICAgaExheW91dC5hZGRXaWRnZXQobGFiZWwpCiAgICAgICAgc2VsZi5zZXRDZW50cmFsV2lkZ2V0
KHNlbGYud2lkZ2V0KQoKICAgIGRlZiBtb3ZlQ2VudGVyKHNlbGYpOgogICAgICAgIHNjcmVlbiA9IFFE
ZXNrdG9wV2lkZ2V0KCkuc2NyZWVuR2VvbWV0cnkoKQogICAgICAgIHNpemUgPSBzZWxmLmdlb21ldHJ5
KCkKICAgICAgICBuZXdMZWZ0ID0gKHNjcmVlbi53aWR0aCgpIC0gc2l6ZS53aWR0aCgpKSAvIDIKICAg
ICAgICBuZXdUb3AgPSAoc2NyZWVuLmhlaWdodCgpIC0gc2l6ZS5oZWlnaHQoKSkgLyAyCiAgICAgICAg
c2VsZi5tb3ZlKGludChuZXdMZWZ0KSwgaW50KG5ld1RvcCkpCgogICAgZGVmIG9wZW5CYXNpY1dpbmRv
dyhzZWxmKToKICAgICAgICBzZWxmLmJhc2ljV2luID0gaUxlYXJuUGx1c0Jhc2ljLklMZWFyblBsdXNC
YXNpYygpCiAgICAgICAgc2VsZi5iYXNpY1dpbi5zZXRGb250KFFGb250KCdBcmlhbCcsIDEwKSkKICAg
ICAgICBzZWxmLmJhc2ljV2luLnNldFN0eWxlU2hlZXQocWRhcmtzdHlsZS5sb2FkX3N0eWxlc2hlZXRf
cHlxdDUoKSkKICAgICAgICBzZWxmLmJhc2ljV2luLmNsb3NlX3NpZ25hbC5jb25uZWN0KHNlbGYucmVj
b3ZlcikKICAgICAgICBzZWxmLmJhc2ljV2luLnNob3coKQogICAgICAgIHNlbGYuc2V0RGlzYWJsZWQo
VHJ1ZSkKICAgICAgICBzZWxmLnNldFZpc2libGUoRmFsc2UpCgogICAgZGVmIG9wZW5Fc3RpbWF0b3JX
aW5kb3coc2VsZik6CiAgICAgICAgc2VsZi5lc3RpbWF0b3JXaW4gPSBpTGVhcm5QbHVzRXN0aW1hdG9y
LklMZWFyblBsdXNFc3RpbWF0b3IoKQogICAgICAgIHNlbGYuZXN0aW1hdG9yV2luLnNldEZvbnQoUUZv
bnQoJ0FyaWFsJywgMTApKQogICAgICAgIHNlbGYuZXN0aW1hdG9yV2luLnNldFN0eWxlU2hlZXQocWRh
cmtzdHlsZS5sb2FkX3N0eWxlc2hlZXRfcHlxdDUoKSkKICAgICAgICBzZWxmLmVzdGltYXRvcldpbi5j
bG9zZV9zaWduYWwuY29ubmVjdChzZWxmLnJlY292ZXIpCiAgICAgICAgc2VsZi5lc3RpbWF0b3JXaW4u
c2hvdygpCiAgICAgICAgc2VsZi5zZXREaXNhYmxlZChUcnVlKQogICAgICAgIHNlbGYuc2V0VmlzaWJs
ZShGYWxzZSkKCiAgICBkZWYgb3Blbk1MV2luZG93KHNlbGYpOgogICAgICAgIHNlbGYubWxXaW4gPSBp
TGVhcm5QbHVzQXV0b01MLklMZWFyblBsdXNBdXRvTUwoKQogICAgICAgIHNlbGYubWxXaW4uc2V0Rm9u
dChRRm9udCgnQXJpYWwnLCAxMCkpCiAgICAgICAgc2VsZi5tbFdpbi5zZXRTdHlsZVNoZWV0KHFkYXJr
c3R5bGUubG9hZF9zdHlsZXNoZWV0X3B5cXQ1KCkpCiAgICAgICAgc2VsZi5tbFdpbi5jbG9zZV9zaWdu
YWwuY29ubmVjdChzZWxmLnJlY292ZXIpCiAgICAgICAgc2VsZi5tbFdpbi5zaG93KCkKICAgICAgICBz
ZWxmLnNldERpc2FibGVkKFRydWUpCiAgICAgICAgc2VsZi5zZXRWaXNpYmxlKEZhbHNlKQoKICAgIGRl
ZiBvcGVuTG9hZE1vZGVsV2luZG93KHNlbGYpOgogICAgICAgIHNlbGYubG9hZFdpbiA9IGlMZWFyblBs
dXNMb2FkTW9kZWwuaUxlYXJuUGx1c0xvYWRNb2RlbCgpCiAgICAgICAgIyBzZWxmLmxvYWRXaW4uc2V0
Rm9udChRRm9udCgnQXJpYWwnLCAxMCkpCiAgICAgICAgc2VsZi5sb2FkV2luLnNldFN0eWxlU2hlZXQo
cWRhcmtzdHlsZS5sb2FkX3N0eWxlc2hlZXRfcHlxdDUoKSkKICAgICAgICBzZWxmLmxvYWRXaW4uY2xv
c2Vfc2lnbmFsLmNvbm5lY3Qoc2VsZi5yZWNvdmVyKQogICAgICAgIHNlbGYubG9hZFdpbi5zaG93KCkK
ICAgICAgICBzZWxmLnNldERpc2FibGVkKFRydWUpCiAgICAgICAgc2VsZi5zZXRWaXNpYmxlKEZhbHNl
KQoKICAgIGRlZiBjbG9zZUV2ZW50KHNlbGYsIGV2ZW50KToKICAgICAgICByZXBseSA9IFFNZXNzYWdl
Qm94LnF1ZXN0aW9uKHNlbGYsICdDb25maXJtIEV4aXQnLCAnQXJlIHlvdSBzdXJlIHdhbnQgdG8gcXVp
dCBpTGVhcm5QbHVzPycsIFFNZXNzYWdlQm94LlllcyB8IFFNZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJv
eC5ObykKICAgICAgICBpZiByZXBseSA9PSBRTWVzc2FnZUJveC5ZZXM6CiAgICAgICAgICAgIHN5cy5l
eGl0KDApCiAgICAgICAgZWxzZToKICAgICAgICAgICAgaWYgZXZlbnQ6CiAgICAgICAgICAgICAgICBl
dmVudC5pZ25vcmUoKQoKICAgIGRlZiBwbG90Q3VydmUoc2VsZiwgY3VydmU9J1JPQycpOgogICAgICAg
IHNlbGYuY3VydmVXaW4gPSBNb2R1bGVzLlBsb3RDdXJ2ZShjdXJ2ZSkKICAgICAgICBzZWxmLmN1cnZl
V2luLnNldFN0eWxlU2hlZXQocWRhcmtzdHlsZS5sb2FkX3N0eWxlc2hlZXRfcHlxdDUoKSkKICAgICAg
ICBzZWxmLmN1cnZlV2luLnNob3coKQoKICAgIGRlZiBkaXNwbGF5SGlzdChzZWxmKToKICAgICAgICBz
ZWxmLmhpc3QgPSBNb2R1bGVzLkhpc3QoKQogICAgICAgIHNlbGYuaGlzdC5zZXRTdHlsZVNoZWV0KHFk
YXJrc3R5bGUubG9hZF9zdHlsZXNoZWV0X3B5cXQ1KCkpCiAgICAgICAgc2VsZi5oaXN0LnNob3coKQoK
ICAgIGRlZiBzY2F0dGVyUGxvdChzZWxmKToKICAgICAgICBzZWxmLnNjYXR0ZXIgPSBNb2R1bGVzLlNj
YXR0ZXJQbG90KCkKICAgICAgICBzZWxmLnNjYXR0ZXIuc2V0U3R5bGVTaGVldChxZGFya3N0eWxlLmxv
YWRfc3R5bGVzaGVldF9weXF0NSgpKQogICAgICAgIHNlbGYuc2NhdHRlci5zaG93KCkKCiAgICBkZWYg
b3BlbkZpbGVURihzZWxmKToKICAgICAgICB0cnk6CiAgICAgICAgICAgIGZpbGVOYW1lLCBvayA9IElu
cHV0RGlhbG9nLlFGaWxlVHJhbnNmb3JtYXRpb24uZ2V0VmFsdWVzKCkKICAgICAgICAgICAgaWYgb2s6
CiAgICAgICAgICAgICAgICBrdyA9IHt9CiAgICAgICAgICAgICAgICBURkRhdGEgPSBNYWNoaW5lTGVh
cm5pbmcuSUxlYXJuTWFjaGluZUxlYXJuaW5nKGt3KSAgICAgICAgICAgIAogICAgICAgICAgICAgICAg
VEZEYXRhLmxvYWRfZGF0YShmaWxlTmFtZSwgdGFyZ2V0PSdUcmFpbmluZycpCiAgICAgICAgICAgICAg
ICBpZiBub3QgVEZEYXRhLnRyYWluaW5nX2RhdGFmcmFtZSBpcyBOb25lOgogICAgICAgICAgICAgICAg
ICAgIHNhdmVkX2ZpbGUsIG9rID0gUUZpbGVEaWFsb2cuZ2V0U2F2ZUZpbGVOYW1lKHNlbGYsICdTYXZl
IHRvJywgJy4vZGF0YScsICdDU1YgRmlsZXMgKCouY3N2KTs7VFNWIEZpbGVzICgqLnRzdik7O1NWTSBG
aWxlcygqLnN2bSk7O1dla2EgRmlsZXMgKCouYXJmZiknKQogICAgICAgICAgICAgICAgICAgIGlmIG9r
OgogICAgICAgICAgICAgICAgICAgICAgICBvazEgPSBURkRhdGEuc2F2ZV9jb2RlcihzYXZlZF9maWxl
LCAndHJhaW5pbmcnKQogICAgICAgICAgICAgICAgICAgICAgICBpZiBub3Qgb2sxOgogICAgICAgICAg
ICAgICAgICAgICAgICAgICAgUU1lc3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgc3RyKHNl
bGYuVEZEYXRhLmVycm9yX21zZyksIFFNZXNzYWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNz
YWdlQm94Lk9rKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgUU1lc3Nh
Z2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgc3RyKGUpLCBRTWVzc2FnZUJveC5PayB8IFFNZXNz
YWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykKCiAgICBkZWYgbWVyZ2VDb2RpbmdGaWxlcyhzZWxmKToK
ICAgICAgICB0cnk6CiAgICAgICAgICAgIGNvZGluZ19maWxlcywgb2sgPSBRRmlsZURpYWxvZy5nZXRP
cGVuRmlsZU5hbWVzKHNlbGYsICdPcGVuIGNvZGluZyBmaWxlcyAobW9yZSBmaWxlIGNhbiBiZSBzZWxl
Y3RlZCknLCAnLi9kYXRhJywgJ0NTViBGaWxlcyAoKi5jc3YpOztUU1YgRmlsZXMgKCoudHN2KTs7U1ZN
IEZpbGVzKCouc3ZtKTs7V2VrYSBGaWxlcyAoKi5hcmZmKScpCiAgICAgICAgICAgIG1lcmdlZF9jb2Rp
bmdzID0gTm9uZQogICAgICAgICAgICBsYWJlbHMgPSBOb25lCiAgICAgICAgICAgIGlmIGxlbihjb2Rp
bmdfZmlsZXMpID4gMDoKICAgICAgICAgICAgICAgIGRhdGFmcmFtZSwgZGF0YWxhYmVsID0gTm9uZSwg
Tm9uZQogICAgICAgICAgICAgICAgZm9yIGZpbGUgaW4gY29kaW5nX2ZpbGVzOgogICAgICAgICAgICAg
ICAgICAgIGlmIGZpbGUuZW5kc3dpdGgoJy50c3YnKToKICAgICAgICAgICAgICAgICAgICAgICAgZGYg
PSBwZC5yZWFkX2NzdihmaWxlLCBzZXA9J1x0JywgaGVhZGVyPU5vbmUpCiAgICAgICAgICAgICAgICAg
ICAgICAgIGRhdGFmcmFtZSA9IGRmLmlsb2NbOiwgMTpdCiAgICAgICAgICAgICAgICAgICAgICAgIGRh
dGFmcmFtZS5pbmRleD1bJ1NhbXBsZV8lcyclaSBmb3IgaSBpbiByYW5nZShkYXRhZnJhbWUudmFsdWVz
LnNoYXBlWzBdKV0KICAgICAgICAgICAgICAgICAgICAgICAgZGF0YWZyYW1lLmNvbHVtbnMgPSBbJ0Zf
JXMnJWkgZm9yIGkgaW4gcmFuZ2UoZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVsxXSldCiAgICAgICAgICAg
ICAgICAgICAgICAgIGRhdGFsYWJlbCA9IG5wLmFycmF5KGRmLmlsb2NbOiwgMF0pLmFzdHlwZShpbnQp
CiAgICAgICAgICAgICAgICAgICAgZWxpZiBmaWxlLmVuZHN3aXRoKCcuY3N2Jyk6CiAgICAgICAgICAg
ICAgICAgICAgICAgIGRmID0gcGQucmVhZF9jc3YoZmlsZSwgc2VwPScsJywgaGVhZGVyPU5vbmUpCiAg
ICAgICAgICAgICAgICAgICAgICAgIGRhdGFmcmFtZSA9IGRmLmlsb2NbOiwgMTpdCiAgICAgICAgICAg
ICAgICAgICAgICAgIGRhdGFmcmFtZS5pbmRleD1bJ1NhbXBsZV8lcyclaSBmb3IgaSBpbiByYW5nZShk
YXRhZnJhbWUudmFsdWVzLnNoYXBlWzBdKV0KICAgICAgICAgICAgICAgICAgICAgICAgZGF0YWZyYW1l
LmNvbHVtbnMgPSBbJ0ZfJXMnJWkgZm9yIGkgaW4gcmFuZ2UoZGF0YWZyYW1lLnZhbHVlcy5zaGFwZVsx
XSldCiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFsYWJlbCA9IG5wLmFycmF5KGRmLmlsb2NbOiwg
MF0pLmFzdHlwZShpbnQpCiAgICAgICAgICAgICAgICAgICAgZWxpZiBmaWxlLmVuZHN3aXRoKCcuc3Zt
Jyk6CiAgICAgICAgICAgICAgICAgICAgICAgIHdpdGggb3BlbihmaWxlKSBhcyBmOgogICAgICAgICAg
ICAgICAgICAgICAgICAgICAgcmVjb3JkID0gZi5yZWFkKCkuc3RyaXAoKQogICAgICAgICAgICAgICAg
ICAgICAgICByZWNvcmQgPSByZS5zdWIoJ1xkKzonLCAnJywgcmVjb3JkKQogICAgICAgICAgICAgICAg
ICAgICAgICBhcnJheSA9IG5wLmFycmF5KFtbaSBmb3IgaSBpbiBpdGVtLnNwbGl0KCldIGZvciBpdGVt
IGluIHJlY29yZC5zcGxpdCgnXG4nKV0pCiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFmcmFtZSA9
IHBkLkRhdGFGcmFtZShhcnJheVs6LCAxOl0sIGR0eXBlPWZsb2F0KQogICAgICAgICAgICAgICAgICAg
ICAgICBkYXRhZnJhbWUuaW5kZXg9WydTYW1wbGVfJXMnJWkgZm9yIGkgaW4gcmFuZ2UoZGF0YWZyYW1l
LnZhbHVlcy5zaGFwZVswXSldCiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFmcmFtZS5jb2x1bW5z
ID0gWydGXyVzJyVpIGZvciBpIGluIHJhbmdlKGRhdGFmcmFtZS52YWx1ZXMuc2hhcGVbMV0pXQogICAg
ICAgICAgICAgICAgICAgICAgICBkYXRhbGFiZWwgPSBhcnJheVs6LCAwXS5hc3R5cGUoaW50KQogICAg
ICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgIHdpdGggb3BlbihmaWxl
KSBhcyBmOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgcmVjb3JkID0gZi5yZWFkKCkuc3RyaXAo
KS5zcGxpdCgnQCcpWy0xXS5zcGxpdCgnXG4nKVsxOl0KICAgICAgICAgICAgICAgICAgICAgICAgYXJy
YXkgPSBucC5hcnJheShbaXRlbS5zcGxpdCgnLCcpIGZvciBpdGVtIGluIHJlY29yZF0pCiAgICAgICAg
ICAgICAgICAgICAgICAgIGRhdGFmcmFtZSA9IHBkLkRhdGFGcmFtZShhcnJheVs6LCAwOi0xXSwgZHR5
cGU9ZmxvYXQpCiAgICAgICAgICAgICAgICAgICAgICAgIGRhdGFmcmFtZS5pbmRleD1bJ1NhbXBsZV8l
cyclaSBmb3IgaSBpbiByYW5nZShkYXRhZnJhbWUudmFsdWVzLnNoYXBlWzBdKV0KICAgICAgICAgICAg
ICAgICAgICAgICAgZGF0YWZyYW1lLmNvbHVtbnMgPSBbJ0ZfJXMnJWkgZm9yIGkgaW4gcmFuZ2UoZGF0
YWZyYW1lLnZhbHVlcy5zaGFwZVsxXSldCiAgICAgICAgICAgICAgICAgICAgICAgIGxhYmVsID0gW10K
ICAgICAgICAgICAgICAgICAgICAgICAgZm9yIGkgaW4gYXJyYXlbOiwgLTFdOgogICAgICAgICAgICAg
ICAgICAgICAgICAgICAgaWYgaSA9PSAneWVzJzoKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAg
ICBsYWJlbC5hcHBlbmQoMSkKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGVsc2U6CiAgICAgICAg
ICAgICAgICAgICAgICAgICAgICAgICAgbGFiZWwuYXBwZW5kKDApCiAgICAgICAgICAgICAgICAgICAg
ICAgIGRhdGFsYWJlbCA9IG5wLmFycmF5KGxhYmVsKQogICAgICAgICAgICAgICAgCiAgICAgICAgICAg
ICAgICAgICAgaWYgbWVyZ2VkX2NvZGluZ3MgaXMgTm9uZToKICAgICAgICAgICAgICAgICAgICAgICAg
bWVyZ2VkX2NvZGluZ3MgPSBucC5oc3RhY2soKGRhdGFsYWJlbC5yZXNoYXBlKCgtMSwgMSkpLCBkYXRh
ZnJhbWUudmFsdWVzKSkKICAgICAgICAgICAgICAgICAgICBlbHNlOgogICAgICAgICAgICAgICAgICAg
ICAgICBtZXJnZWRfY29kaW5ncyA9IG5wLmhzdGFjaygobWVyZ2VkX2NvZGluZ3MsIGRhdGFmcmFtZS52
YWx1ZXMpKQogICAgICAgICAgICBzYXZlZF9maWxlLCBvayA9IFFGaWxlRGlhbG9nLmdldFNhdmVGaWxl
TmFtZShzZWxmLCAnU2F2ZSB0bycsICcuL2RhdGEnLCAnQ1NWIEZpbGVzICgqLmNzdik7O1RTViBGaWxl
cyAoKi50c3YpOztTVk0gRmlsZXMoKi5zdm0pOztXZWthIEZpbGVzICgqLmFyZmYpJykKICAgICAgICAg
ICAgZGF0YSA9IG1lcmdlZF9jb2RpbmdzCiAgICAgICAgICAgIGlmIHNhdmVkX2ZpbGUuZW5kc3dpdGgo
Jy5jc3YnKToKICAgICAgICAgICAgICAgIG5wLnNhdmV0eHQoc2F2ZWRfZmlsZSwgZGF0YSwgZm10PSIl
cyIsIGRlbGltaXRlcj0nLCcpCiAgICAgICAgICAgIGlmIHNhdmVkX2ZpbGUuZW5kc3dpdGgoJy50c3Yn
KToKICAgICAgICAgICAgICAgIG5wLnNhdmV0eHQoc2F2ZWRfZmlsZSwgZGF0YSwgZm10PSIlcyIsIGRl
bGltaXRlcj0nLCcpCiAgICAgICAgICAgIGlmIHNhdmVkX2ZpbGUuZW5kc3dpdGgoJy5zdm0nKToKICAg
ICAgICAgICAgICAgIHdpdGggb3BlbihzYXZlZF9maWxlLCAndycpIGFzIGY6CiAgICAgICAgICAgICAg
ICAgICAgZm9yIGxpbmUgaW4gZGF0YToKICAgICAgICAgICAgICAgICAgICAgICAgZi53cml0ZSgnJXMn
ICUgbGluZVswXSkKICAgICAgICAgICAgICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UoMSwgbGVuKGxp
bmUpKToKICAgICAgICAgICAgICAgICAgICAgICAgICAgIGYud3JpdGUoJyAgJWQ6JXMnICUgKGksIGxp
bmVbaV0pKQogICAgICAgICAgICAgICAgICAgICAgICBmLndyaXRlKCdcbicpCiAgICAgICAgICAgIGlm
IHNhdmVkX2ZpbGUuZW5kc3dpdGgoJy5hcmZmJyk6CiAgICAgICAgICAgICAgICB3aXRoIG9wZW4oc2F2
ZWRfZmlsZSwgJ3cnKSBhcyBmOgogICAgICAgICAgICAgICAgICAgIGYud3JpdGUoJ0ByZWxhdGlvbiBk
ZXNjcmlwdG9yXG5cbicpCiAgICAgICAgICAgICAgICAgICAgZm9yIGkgaW4gcmFuZ2UoMSwgbGVuKGRh
dGFbMF0pKToKICAgICAgICAgICAgICAgICAgICAgICAgZi53cml0ZSgnQGF0dHJpYnV0ZSBmLiVkIG51
bWVyaWNcbicgJSBpKQogICAgICAgICAgICAgICAgICAgIGYud3JpdGUoJ0BhdHRyaWJ1dGUgcGxheSB7
eWVzLCBub31cblxuJykKICAgICAgICAgICAgICAgICAgICBmLndyaXRlKCdAZGF0YVxuJykKICAgICAg
ICAgICAgICAgICAgICBmb3IgbGluZSBpbiBkYXRhOgogICAgICAgICAgICAgICAgICAgICAgICBmb3Ig
ZmVhIGluIGxpbmVbMTpdOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgZi53cml0ZSgnJXMsJyAl
IGZlYSkKICAgICAgICAgICAgICAgICAgICAgICAgaWYgaW50KGxpbmVbMF0pID09IDE6CiAgICAgICAg
ICAgICAgICAgICAgICAgICAgICBmLndyaXRlKCd5ZXNcbicpCiAgICAgICAgICAgICAgICAgICAgICAg
IGVsc2U6CiAgICAgICAgICAgICAgICAgICAgICAgICAgICBmLndyaXRlKCdub1xuJykKICAgICAgICBl
eGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgICAgIFFNZXNzYWdlQm94LmNyaXRpY2FsKHNlbGYs
ICdFcnJvcicsIHN0cihlKSwgUU1lc3NhZ2VCb3guT2sgfCBRTWVzc2FnZUJveC5ObywgUU1lc3NhZ2VC
b3guT2spCgogICAgZGVmIG9wZW5Eb2N1bWVudFVybChzZWxmKToKICAgICAgICBRRGVza3RvcFNlcnZp
Y2VzLm9wZW5VcmwoUVVybCgnaHR0cHM6Ly9pbGVhcm5wbHVzLmVyYy5tb25hc2guZWR1LycpKQoKICAg
IGRlZiBvcGVuQWJvdXQoc2VsZik6CiAgICAgICAgUU1lc3NhZ2VCb3guaW5mb3JtYXRpb24oc2VsZiwg
J2lMZWFyblBsdXMnLCAnVmVyc2lvbjogMS4wXG5BdXRob3I6IFpoZW4gQ2hlblxuRS1tYWlsOiBjaGVu
emhlbi13aW4yMDA5QDE2My5jb20nLCBRTWVzc2FnZUJveC5PayB8IFFNZXNzYWdlQm94Lk5vLCBRTWVz
c2FnZUJveC5PaykKCiAgICBkZWYgZHJhd0JveHBsb3Qoc2VsZik6CiAgICAgICAgdHJ5OgogICAgICAg
ICAgICB4LCB5LCBkYXRhLCBvayA9IElucHV0RGlhbG9nLlFCb3hQbG90SW5wdXQuZ2V0VmFsdWVzKCkK
ICAgICAgICAgICAgaWYgb2s6CiAgICAgICAgICAgICAgICBzZWxmLmJveFdpbiA9IFBsb3RXaWRnZXRz
LkN1c3RvbVNpbmdsZUJveHBsb3RXaWRnZXQoZGF0YSwgeCwgeSkKICAgICAgICAgICAgICAgIHNlbGYu
Ym94V2luLnNob3coKQogICAgICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICAgICAgUU1l
c3NhZ2VCb3guY3JpdGljYWwoc2VsZiwgJ0Vycm9yJywgc3RyKGUpLCBRTWVzc2FnZUJveC5PayB8IFFN
ZXNzYWdlQm94Lk5vLCBRTWVzc2FnZUJveC5PaykKCiAgICBkZWYgZHJhd0hlYXRtYXAoc2VsZik6CiAg
ICAgICAgdHJ5OgogICAgICAgICAgICB4LCB5LCBkYXRhLCBvayA9IElucHV0RGlhbG9nLlFIZWF0bWFw
SW5wdXQuZ2V0VmFsdWVzKCkKICAgICAgICAgICAgaWYgb2s6CiAgICAgICAgICAgICAgICBzZWxmLmhl
YXRXaW4gPSBQbG90V2lkZ2V0cy5DdXN0b21IZWF0bWFwV2lkZ2V0KGRhdGEsIHgsIHkpCiAgICAgICAg
ICAgICAgICBzZWxmLmhlYXRXaW4uc2hvdygpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgog
ICAgICAgICAgICBRTWVzc2FnZUJveC5jcml0aWNhbChzZWxmLCAnRXJyb3InLCBzdHIoZSksIFFNZXNz
YWdlQm94Lk9rIHwgUU1lc3NhZ2VCb3guTm8sIFFNZXNzYWdlQm94Lk9rKQoKICAgIGRlZiByZWNvdmVy
KHNlbGYsIG1vZHVsZSk6CiAgICAgICAgdHJ5OgogICAgICAgICAgICBpZiBtb2R1bGUgPT0gJ0Jhc2lj
JzoKICAgICAgICAgICAgICAgIGRlbCBzZWxmLmJhc2ljV2luCiAgICAgICAgICAgIGVsaWYgbW9kdWxl
ID09ICdFc3RpbWF0b3InOgogICAgICAgICAgICAgICAgZGVsIHNlbGYuZXN0aW1hdG9yV2luCiAgICAg
ICAgICAgIGVsaWYgbW9kdWxlID09ICdBdXRvTUwnOgogICAgICAgICAgICAgICAgZGVsIHNlbGYubWxX
aW4KICAgICAgICAgICAgZWxpZiBtb2R1bGUgPT0gJ0xvYWRNb2RlbCc6CiAgICAgICAgICAgICAgICBk
ZWwgc2VsZi5sb2FkV2luCiAgICAgICAgICAgIGVsc2U6CiAgICAgICAgICAgICAgICBwYXNzCiAgICAg
ICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOgogICAgICAgICAgICBwYXNzCiAgICAgICAgc2VsZi5zZXRE
aXNhYmxlZChGYWxzZSkKICAgICAgICBzZWxmLnNldFZpc2libGUoVHJ1ZSkKCg=="""
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
class mainWindow(QMainWindow):
    def __init__(self):
        super(mainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('iLearnPlus')
        # self.resize(750, 500)
        self.setMaximumSize(600, 400)
        self.setMinimumSize(600, 400)
        self.setWindowIcon(QIcon('images/logo.ico'))
        self.setFont(QFont('Arial'))
        bar = self.menuBar()
        app = bar.addMenu('Applications')
        basic = QAction('iLearnPlus Basic', self)        
        estimator = QAction('iLearnPlus Estimator', self)        
        autoML = QAction('iLearnPlus AutoML', self)        
        loadModel = QAction('Load model(s)', self)        
        quit = QAction('Exit', self)        
        app.addAction(basic)
        app.addAction(estimator)
        app.addAction(autoML)
        app.addSeparator()
        app.addAction(loadModel)
        app.addSeparator()
        app.addAction(quit)

        visual = bar.addMenu('Visualization')
        roc = QAction('Plot ROC curve', self)        
        prc = QAction('Plot PRC curve', self)        
        boxplot = QAction('Boxplot', self)        
        heatmap = QAction('Heatmap', self)        
        scatterPlot = QAction('Scatter plot', self)        
        data = QAction('Distribution visualization', self)        
        visual.addActions([roc, prc, boxplot, heatmap, scatterPlot, data])

        tools = bar.addMenu('Tools')        
        fileTF = QAction('File format transformation', self)        
        mergeFile = QAction('Merge coding files into one', self)        
        tools.addActions([fileTF, mergeFile])

        help = bar.addMenu('Help')
        document = QAction('Document', self)        
        about = QAction('About', self)        
        help.addActions([document, about])        
        self.moveCenter()
        self.widget = QWidget()
        hLayout = QHBoxLayout(self.widget)
        hLayout.setAlignment(Qt.AlignCenter)
        label = QLabel()        
        label.setPixmap(QPixmap('images/logo.png'))
        hLayout.addWidget(label)
        self.setCentralWidget(self.widget)

    def moveCenter(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft), int(newTop))    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window.show()
    sys.exit(app.exec_())
    
    

