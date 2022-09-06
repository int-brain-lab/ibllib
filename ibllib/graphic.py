#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Sunday, February 3rd 2019, 11:59:56 am
import tkinter as tk
from tkinter import messagebox
import traceback
import warnings

for line in traceback.format_stack():
    print(line.strip())

warnings.warn('ibllib.graphic has been deprecated. '
              'See stack above', DeprecationWarning)


def popup(title, msg):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, msg)
    root.quit()
