import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cvxpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'PySimpleGUI'])
subprocess.call
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as pg
import math
from table import create_table
from table import download3
from table import download4
import ctypes
import platform
from frontiere_efficiente import matrice_covariance
from frontiere_efficiente import FrontiereEfficiente
from black_litterman import BlackLitterman
from PIL import Image

masi = pd.read_csv('data/masi').set_index('Date')
masi_w = pd.read_csv('data/masi_w').set_index('Unnamed: 0')['0']
cov = matrice_covariance(masi[4800:])
masi_keys = list(masi_w.index)
n = len(masi_keys)

pg.theme("DefaultNoMoreNagging")
def make_dpi_aware():
    if int(platform.release()) >= 8:
        ctypes.windll.shcore.SetProcessDpiAwareness(True)
make_dpi_aware()
def autopct(x):
    if x == 0:
        return None
    return int(x * 10) / 10
def to_pct_sqr(x):
    return (x/100)*(x/100) 
def aview_to_matrix(boite):
    a = np.zeros((1, n)); 
    id = 0
    while masi_keys[id] is not boite:
        id = id + 1
    a[0][id] = 1
    return a
def rview_to_matrix(boite1, boite2):
    a = np.zeros((1, n))
    id = 0
    while masi_keys[id] is not boite1:
        id = id + 1
    a[0][id] = 1
    id = 0
    while masi_keys[id] is not boite2:
        id = id + 1
    a[0][id] = -1
    return a

absolute_view = [
                 pg.Combo(masi_keys),
                 pg.Text('aura un rendement de'), pg.InputText(do_not_clear=False,size=(3,1)), pg.Text('%'),
                 pg.Text('         |             Incertitude:'), pg.InputText(do_not_clear=False,size=(3,1)), pg.Text('%')
                 ]

relative_view = [
                 pg.Combo(masi_keys),
                 pg.Text('va surperformer'),
                 pg.Combo(masi_keys),
                 pg.Text('de'), pg.InputText(do_not_clear=False,size=(3,1)), pg.Text('%'),
                 pg.Text('|             Incertitude:'), pg.InputText(do_not_clear=False,size=(3,1)), pg.Text('%')
                 ]
views_column = [
                [pg.Text('    Intégration des anticipations de rendement',font=("Helvetica",16,'bold'))],
                [pg.Text('')],
                [pg.HorizontalSeparator()],
                absolute_view,
                [pg.Button('Add Absolute view')],
                [pg.Text('')],
                relative_view,
                [pg.Button('Add Relative view')],
                [pg.HorizontalSeparator()],
                [pg.Text(' Prime de risque:\n(Par defaut: 5%)'), pg.InputText(size=(3,1)), pg.Text('%'),
                 pg.VSeperator(),
                 pg.Text(' Taux sans rique:\n(Par defaut: 2%)'), pg.InputText(size=(3,1)), pg.Text('%'),
                 pg.VSeperator(),
                 pg.Text('      Tau:\n(Par defaut:0.07)'), pg.InputText(size=(5,1))],
                [pg.HorizontalSeparator()],
                [pg.Text('')],
                [pg.Listbox(values=[],size=(65,4), enable_events=True, key='list')],
                [pg.Text('                                       '),
                 pg.Button('Submit', image_data=download3, button_color=('white', pg.theme_background_color()), border_width=0, ),
                 pg.Button('Restart', image_data=download3, button_color=('white', pg.theme_background_color()), border_width=0, )],
                ]
portfolio_column = [
                    [pg.Text('')],
                    [pg.Text('')],
                    [pg.Button('Rendements\nde marché', image_data=download4, button_color=('white', pg.theme_background_color()), border_width=0, )],
                    [pg.Text('')],
                    [pg.Button('Rendements\nde Black-Litterman', image_data=download4, button_color=('white', pg.theme_background_color()), border_width=0, )],
                    [pg.Text('')],
                    [pg.Button('Portefeuille optimal', image_data=download4, button_color=('white', pg.theme_background_color()), border_width=0, )],
                    [pg.Text('')],
                    [pg.Button('ALL')],
                    [pg.Text('')],
                    [pg.Text('')],
                    [pg.Text('             '),pg.Image("data/logo-noir.png",size=(150,50))],
                    ]
layout = [[
        pg.Column(views_column),
        pg.VSeperator(),
        pg.Column(portfolio_column, element_justification='center')
        ]]

window = pg.Window('Black-Litterman', layout, finalize=True,modal=True, grab_anywhere=True)

views_list = []
views_matrix = np.matrix([[]])
views_pct = np.matrix([[]])
confi_array = np.matrix([[]])
confi_matrix = np.matrix([[]])
P = None
Q = None
omega = None
k = 0
delta = 0.05
rf = 0.02
tau = 0.07027888706
while True:
    plt.show()
    event ,values = window.read()
    if event == pg.WIN_CLOSED:
        break
    if event == 'Add Absolute view' and values[1] and values[2]:
        if views_matrix.any():
            views_pct = np.concatenate((views_pct, [[int(values[2])/100]]))
            views_matrix = np.concatenate((views_matrix, aview_to_matrix(values[1])))
            if values[3] != '':
                confi_array = np.concatenate((confi_array, [[to_pct_sqr(int(values[3]))]]))
            else:
                k = 1
        else:
            views_pct = np.matrix([[int(values[2])/100]])
            views_matrix = aview_to_matrix(values[1])
            if values[3] != '':
                confi_array = np.matrix([[to_pct_sqr(int(values[3]))]])
            else:
                k = 1
        views_list.append(f'{values[1]} aura un rendement de {values[2]}%')
        confi_matrix = np.diagflat(confi_array)
    elif event == 'Add Relative view' and values[4] and values[5] and values[6]:
        if views_matrix.any():
            views_pct = np.concatenate((views_pct, [[int(values[6])/100]]))
            views_matrix = np.concatenate((views_matrix, rview_to_matrix(values[4], values[5])))
            if values[7] != '':
                confi_array = np.concatenate((confi_array, [[to_pct_sqr(int(values[7]))]]))
            else:
                k = 1
        else:
            views_pct = np.matrix([[int(values[6])/100]])
            views_matrix = rview_to_matrix(values[4],values[5])
            if values[7] != '':
                confi_array = np.matrix([[to_pct_sqr(int(values[7]))]])
            else:
                k = 1
        views_list.append(f'{values[4]} va surperformer {values[5]} de {values[6]}%')
        confi_matrix = np.diagflat(confi_array)

    elif event == 'Restart':
        views_list = []
        views_matrix = np.matrix([[]])
        views_pct = np.matrix([[]])
        confi_array = np.matrix([[]])
        confi_matrix = np.matrix([[]])
        P = None
        Q = None
        omega = None
        delta = 0.05
        rf = 0.02
        tau = 0.07027888706
        k = 0

    elif event == 'Submit':
        P = views_matrix
        Q = views_pct
        if k == 0:
            omega = confi_matrix
        if values[9] != '':
            delta = float(values[9]) / 100
        else:
            delta = 0.05
        if values[11] != '':
            rf = float(values[11]) / 100
        else:
            rf = 0.02
        if values[13] != '':
            tau = float(values[13])
        else:
            tau = 0.07027888706

    elif event == 'Rendements\nde marché':
        bl = BlackLitterman(sigma=cov, prime_de_risque=delta, taux_sans_risque=rf)
        bl_fe = bl.frontiere_efficiente()
        df = pd.DataFrame(np.diag(cov), index=cov.columns)
        df.columns = ['Volatilités(%)']
        df['Rendements(%)'] = round(bl.pi['Rendements de marché']*100, 1)
        df['Volatilités(%)'] = round(df['Volatilités(%)'].apply(math.sqrt)*100)
        df['Boites'] = df.index
        df = df[['Boites','Rendements(%)','Volatilités(%)']]
        df['Poids(%)'] = round(masi_w * 100, 1)
        df['Betas'] = bl_fe.betas['Betas']
        headers = df.columns.to_numpy()
        data_array = df.to_numpy()
        create_table(headers.tolist(), data_array.tolist())

    elif event == 'Rendements\nde Black-Litterman':
        bl = BlackLitterman(sigma=cov,P=P,Q=Q,omega=omega,tau=tau,prime_de_risque=delta,taux_sans_risque=rf)
        bl_rends = bl.rendements()
        (pd.concat([bl_rends,bl.pi],axis=1)*100).plot.bar(figsize=(10,4), fontsize=8,rot=0,grid=True,ylabel='%')
    
    elif event == 'Portefeuille optimal':
        bl = BlackLitterman(sigma=cov,P=P,Q=Q,omega=omega,tau=tau,prime_de_risque=delta,taux_sans_risque=rf)
        bl_fe = bl.frontiere_efficiente()
        bl_fe.portefeuille_optimal().plot.pie(subplots=True,title='Portefeuille optimal',figsize=(6,6),autopct=autopct,labeldistance=1.1)

    elif event=='ALL':
        bl = BlackLitterman(sigma=cov,P=P,Q=Q,omega=omega,tau=tau,prime_de_risque=delta,taux_sans_risque=rf)
        bl_fe = bl.frontiere_efficiente()
        df = pd.DataFrame(np.diag(cov), index=cov.columns)
        df.columns = ['Volatilités(%)']
        df['Rendements(%)'] = round(bl.pi['Rendements de marché']*100, 1)
        df['Volatilités(%)'] = round(df['Volatilités(%)'].apply(math.sqrt)*100,1)
        df['Boites'] = df.index
        df['BL-Rendements(%)'] = round(bl.rendements()['Rendements de Black-Litterman']*100, 1)
        bl_vol = pd.DataFrame(np.diag(bl.cov()), index=cov.columns) 
        bl_vol.columns = ['BL-Volatilités(%)']
        df['BL-Volatilités(%)'] = round(bl_vol['BL-Volatilités(%)'].apply(math.sqrt)*100,1)
        df['Poids (%)'] = round(masi_w * 100,1)
        df['Poids optimaux'] = round(bl_fe.portefeuille_optimal()*100,1)
        df['Poids de BL'] = round(bl.portefeuille()['Poids de Black-Litterman']*100,1)
        df['Betas'] = bl_fe.betas['Betas']
        df = df[['Boites','Rendements(%)','Volatilités(%)','Betas','BL-Rendements(%)','BL-Volatilités(%)','Poids (%)','Poids optimaux']]
        headers = df.columns.to_numpy()
        data_array = df.to_numpy()
        create_table(headers.tolist(), data_array.tolist())

    window['list'].update(views_list)
window.close()

