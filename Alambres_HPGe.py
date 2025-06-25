# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:44:0n_back 2024

@author: Alex
"""
import sys
sys.path.append('D:\\Codigos_py\\Repositorio')

import Activacion as act
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#%%Fondo para una adq de fondo
# Llamo a los espectros del fondo
path_fondo_OFF = "D:/Proyecto LINT at Prompt Gamma/2025-06-09_fluxatPGwithfoils/11.06.2025_PG_LINT/-txt/DF_fondo_rOFF_12.06.2025.txt"
path_fondo_ON = "D:/Proyecto LINT at Prompt Gamma/2025-06-09_fluxatPGwithfoils/11.06.2025_PG_LINT/-txt/DF_fondo_rON_10.06.2025.txt"

Fondo_OFF = act.fromspec(path_fondo_OFF)
Fondo_ON = act.fromspec(path_fondo_ON)

#%% FONDO - JOBs y paso temporal de cps 
nro_med = 64
medno = ["%.2d" % i for i in range(nro_med)]
ROI_Ar = [2204, 2217] 
path = lambda nro: f"D:/Irradiaciones de Flujo Térmico/Mapeo flujo Config DN9 - V 29jul24/Job_Fondo_29jul24/FONDO_4sep_reacONOFF_26a29jul24_med0{nro}.Txt"
t_partes = np.array(range(nro_med), dtype = "object")
t_lives = np.zeros(nro_med)
t_reals = np.zeros(nro_med)

Fondo_ROIs = np.zeros((nro_med, 2))
cps_ROIs = np.zeros((nro_med, 2))
all_counts = []

for ii, med in enumerate(medno):
    Fondo_spec = act.fromspec(path(med))
    t_partes[ii] = Fondo_spec.tinicio
    Fondo_ROIs[ii] = [Fondo_spec.ROI(ROI_Ar)[x] for x in ['net', 'net_err']]
    t_lives[ii] = Fondo_spec.tlive
    t_reals[ii] = Fondo_spec.treal
    cps_ROIs[ii] = Fondo_ROIs[ii]/t_lives[ii]
    all_counts.append(Fondo_spec.counts)

t_partes_h = [(t-t_partes[0]).total_seconds()/3600 for t in t_partes]
all_counts = np.row_stack(all_counts).T

plt.figure()
plt.errorbar(t_partes_h, cps_ROIs[:, 0], yerr=cps_ROIs[:, 1], fmt='.', label = 'Ar-41')
plt.xlabel('t [h]')
plt.ylabel('cps')
plt.grid(ls='--')
plt.title(f'Ar-41 en ROI = {ROI_Ar}')

#%% Suma de fondos
fin_OFF = 40
inicio_ON = 42

Fondo_OFF = act.fromspec(path('00'))
Fondo_ON = act.fromspec(path(str(inicio_ON)))

Fondo_ON.treal = t_reals[inicio_ON:].sum()
Fondo_ON.tlive = t_lives[inicio_ON:].sum()
Fondo_ON.counts = all_counts[:, inicio_ON:].sum(axis=1)

Fondo_OFF.treal = t_reals[:fin_OFF+1].sum()
Fondo_OFF.tlive = t_lives[:fin_OFF+1].sum()
Fondo_OFF.counts = all_counts[:, :fin_OFF+1].sum(axis=1)

#%% Eff - param

#Cargo datos
path_Eu152 = "D:/Proyecto LINT at Prompt Gamma/2025-06-09_fluxatPGwithfoils/11.06.2025_PG_LINT/-txt/DF_Eu152_76044A_sep3_rON_11.06.2025.txt"
path_Roical = "D:/Proyecto LINT at Prompt Gamma/2025-06-09_fluxatPGwithfoils/11.06.2025_PG_LINT/-txt/ROI_cal_DF_Eu152_rON.txt"

# coef_en = np.array([1.761200E+001, 5.759853E-001, 3.779022E-007])
#19feb24: np.array([2.113087E+001, 6.826579E-001, 7.437105E-007])
#29jul24: np.array([1.761200E+001, 5.759853E-001, 3.779022E-007])
#11-12jul25: 
    # DF_rON : np.array([1.658397E+001, 5.524793E-001,  1.509936E-007])
    # XC_rOFF : np.array([1.759531E+001, 5.770479E-001, 4.121741E-007])
    # XC_rON : np.array([1.759531E+001, 5.770479E-001, 4.121741E-007])


ROIs_cal = np.loadtxt(path_Roical) #llamo a las ROIs definidas
spec_cal = act.fromspec(path_Eu152)#, coef_en = coef_en) #recupero el espectro 

Fuente_pat = 'Eu152_76044A-440'
Fecha_cal = datetime(2025, 6, 11, 0,0,0) #día de la calibración (sólo días)
datos_cal = act.NAA_calib(Fuente_pat, Fecha_cal)

Epeak_Eu = ROIs_cal[:, 0]
ROIs_cal = ROIs_cal[:, 1:].astype(int)

#%% Eff - reporte
# =============================================================================
# CÁLCULO DE EFICIENCIA
# =============================================================================

grado_pol = 1

eff_fit = datos_cal.cal_eff(ROIs_cal, spec_cal, Fondo_ON, grado_pol, criterio = 0.003)

eff_data, err_eff, coef, coef_err, chi2, data_sel, residual, pvalor, ddof, rhos, var_mus = eff_fit

for jj in range(len(coef)):
    print(f'a{grado_pol-jj} = '+ ' '.join([*act.redondeo(coef[jj], coef_err[jj], 2, texto=True)])+ 
          f' keV^-{grado_pol-jj}')
print(f'χ2 = {chi2}, ddof = {ddof}, p-value = {pvalor}. \n Válido en E = ({Epeak_Eu.min().round(2)}-{Epeak_Eu.max().round(2)}) keV')

E_arr = np.linspace(Epeak_Eu.min(), Epeak_Eu.max(), num=1000)
logE_arr  = np.log(E_arr)
eff_eval = np.polyval(coef, logE_arr)
sigma_mu_est = np.sqrt(var_mus(logE_arr))

plt.figure(1, figsize = (5,5))
plt.errorbar(np.log(Epeak_Eu), np.log(eff_data), yerr= err_eff/eff_data, fmt='.')
plt.plot(logE_arr, eff_eval, label = 'fit, grado {}'.format(grado_pol))
plt.fill_between(logE_arr, eff_eval-sigma_mu_est, eff_eval+sigma_mu_est, color='tab:orange', alpha=0.2)
plt.grid(True, ls = '--')
plt.ylabel('$ln$ Eff')
plt.xlabel('$ln$ E')
plt.tight_layout()
plt.legend()

#%% Pico Incógnita o pico solo
# coef = np.array([-1.154,	0.64]) 
#29jul24: np.array([-1.159,  0.64]) 
#19feb24: array([-1.17555463,  0.74364904]) / INFORME = np.array([-1.168,  0.70])

# coef_err = np.array([0.02,	0.13])
#29jul24: np.array([0.022,	0.14])
#19feb24: array([0.0195103 , 0.12846296]) / INFORME = np.array([0.016,	0.11]) 

coef_tabla = np.vstack((coef, coef_err)).T

fecha_incog = datetime(2025, 6, 11)

n_back = 3

ROIs = {'137Cs': [1160, 1174]}
#29jul24: [1108, 1126], 19feb24: [933, 943]

Comp_incog = {'137Cs': 1}
Incog = act.Alambre(Comp_incog, 0)

Fondo_ROIs = {mat:Fondo_ON.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs}
cps_ROIs = {mat:np.array([Fondo_ROIs[mat][xx] for xx in ['net', 'net_err']])/Fondo_ON.tlive for mat in ROIs}

path_Cs137 = "D:/Proyecto LINT at Prompt Gamma/2025-06-09_fluxatPGwithfoils/11.06.2025_PG_LINT/-txt/DF_Cs137_76072_sep3_rON_11.06.2025.txt"
spec_incog = act.fromspec(path_Cs137, coef_en = np.array([1.658397E+001, 5.524793E-001,  1.509936E-007]))
pico_incog = spec_incog.ROI(list(ROIs.values())[0], n_bkg=n_back)

E_incog = pico_incog['en_max']
cps_pico = np.array([pico_incog['net'], pico_incog['net_err']])/spec_incog.tlive - cps_ROIs['137Cs']

Act_incog, Act_RA3, diff = act.Actividad(np.array([E_incog]), np.array([cps_pico[0]]), np.array([cps_pico[1]]), 
                          spec_incog.treal, coef_tabla, isfromRA3=True, Fuente='Cs137_76072-440', dt = fecha_incog, var_mu = var_mus)


#%% ALAMBRES - Defino ROIs y calculo sus cps de fondo
# =============================================================================
# ALAMBRES
# =============================================================================

Composition = {'63Cu': 0.9845, '197Au': 0.0155} #, '40Ar': 0.0
tirr = 1800

# Llamo a los picos según tabla de IAEA por API, en base a la composición del alambre
Alambre_W = act.Alambre(Composition, tirr) 
n_back = 3

ROIs = {'198Au': [680-n_back, 686+n_back], '64Cu': [844-n_back, 868+n_back]}
# 29jul24: {'198Au': [680-n_back, 686+n_back], '64Cu': [844-n_back, 868+n_back]}
# 19feb24: {'198Au': [567, 578], '64Cu': [705, 729]}

#Puedo usar los calculados recientemente o arrays ya definidos
# coef_eff = coef#3 sep: np.array([-1.159,  0.64])
# coef_eff_err = coef_err #3 sep: np.array([0.022,	0.14])
# coef_tabla = np.vstack((coef_eff, coef_eff_err)).T

Fondo_ROIs_OFF = {mat:Fondo_OFF.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs}
cps_ROIs_OFF = {mat:np.array([Fondo_ROIs_OFF[mat][xx] for xx in ['net', 'net_err']])/Fondo_OFF.tlive for mat in ROIs}

Fondo_ROIs_ON = {mat:Fondo_ON.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs}
cps_ROIs_ON = {mat:np.array([Fondo_ROIs_ON[mat][xx] for xx in ['net', 'net_err']])/Fondo_ON.tlive for mat in ROIs}

#%% Llamo a las partes del archivo y calculo la actividad en cada uno
# coef_en = np.array([1.769383E+001, 5.759117E-001, 3.971684E-007])#31jul24 deforme: np.array([17.5278, 0.575533, 3.6319e-7]) 

nromed = {'W21': 14}#, 'W06': 5, 'W05': 11}
# 29jul24: {'W21': 14, 'W06': 5, 'W05': 11}, 19feb24: {'W17': 1, 'W34': 1, 'W39': 1}

cps_Alambres = {}
Act_W = {}
dts = {}
t_inicio = datetime(2024, 7, 29, 10, 35, 12)
#fin de irradiación 29jul24: datetime(2024, 7, 29, 10, 35, 12)
#19feb24(HPGe): datetime(2024, 2, 19, 14, 32, 27)
tdead = {}
path_job = lambda alambre, med: f"D:/Irradiaciones de Flujo Térmico/Mapeo flujo Config DN9 - V 29jul24/Mediciones Alambres/Job_Alambres/-txt/{alambre}_4sep_reacON_med0{med}.Txt"
#19feb24: "D:/Calibracion SPND/Calibracion DW9 A0 19feb24/DATOS/-txt/{alambre}_Reac ON_4sep_20feb24_med0{med}.Txt"
#29jul24: "D:/Irradiaciones de Flujo Térmico/Mapeo flujo Config DN9 - V 29jul24/Mediciones Alambres/Job_Alambres/-txt/{alambre}_4sep_reacON_med0{med}.Txt"


for alambre in nromed:
    medno = ["%.2d" % i for i in range(nromed[alambre])]
    Act_W[alambre] = {mat:np.zeros((len(medno), 2)) for mat in ROIs}
    cps_Alambres[alambre] = {mat:np.zeros((len(medno), 2)) for mat in ROIs}
    deadtime = np.zeros(len(medno))
    dt = np.zeros(len(medno))
    for ii, med in enumerate(medno):
        data = act.fromspec(path_job(alambre, med), coef_en)
        dt[ii] = (data.tinicio - t_inicio).total_seconds()
        deadtime[ii] = 100*(1 - data.tlive/data.treal)
        for mat in ROIs:    
            data_ROI = data.ROI(ROIs[mat], n_bkg=n_back)
            Net_cps, Err_cps = (data_ROI['net']/data.tlive - cps_ROIs_ON[mat][0], np.sqrt((data_ROI['net_err']/data.tlive)**2 + cps_ROIs_ON[mat][1]**2))
            cps_Alambres[alambre][mat][ii] = [Net_cps, Err_cps]
            Act_W[alambre][mat][ii] = Alambre_W.Act_alambre(mat, np.array([data_ROI['en_max']]), np.array([Net_cps]), Err_cps, dt[ii], data.treal, coef_tabla, rhos[0][0])
    dts[alambre] = dt 
    tdead[alambre] = deadtime
    
alambre = 'W05'
for med in range(15, 19):
    data = act.fromspec(path_job(alambre, med))
    dts[alambre] = np.append(dts[alambre], (data.tinicio - t_inicio).total_seconds())
    tdead[alambre] = np.append(tdead[alambre], 100*(1 - data.tlive/data.treal))
    for mat in ROIs:    
        data_ROI = data.ROI(ROIs[mat], n_bkg=n_back)
        Net_cps, Err_cps = (data_ROI['net']/data.tlive - cps_ROIs_OFF[mat][0], np.sqrt((data_ROI['net_err']/data.tlive)**2 + cps_ROIs_OFF[mat][1]**2))
        cps_Alambres[alambre][mat] = np.vstack((cps_Alambres[alambre][mat], [Net_cps, Err_cps]))        
        Act_W[alambre][mat] = np.vstack((Act_W[alambre][mat], Alambre_W.Act_alambre(mat, np.array([data_ROI['en_max']]), np.array([Net_cps]), Err_cps, dts[alambre][-1], data.treal, coef_tabla, rhos[0][0])))

#%% Figuras de evolucion de act. y cps en alambres
# f = {'W4': 1, 'W37': 1.05}
# life = {'64Cu': 47287.11, '198Au': 306830.72, '41Ar': 1}

for alambre in nromed:
    for ii, mat in enumerate(ROIs):
        plt.figure(90+ii)
        plt.errorbar(dts[alambre]/3600, Act_W[alambre][mat][:, 0], 
                     yerr=Act_W[alambre][mat][:, 1], fmt='.', 
                     label = alambre) 
        plt.xlabel('t [horas]')
        plt.ylabel('Act [Bq]')
        # plt.yscale('log')
        plt.grid(True)
        plt.title('Actividad de '+ mat)
        plt.legend()
        plt.tight_layout()
    
for alambre in nromed:
    for ii, mat in enumerate(ROIs):
        plt.figure(5+ii)
        plt.errorbar(dts[alambre]/3600, cps_Alambres[alambre][mat][:, 0], 
                     yerr=cps_Alambres[alambre][mat][:, 1], fmt='.', 
                     label = alambre) 
        plt.xlabel('t [horas]')
        plt.ylabel('cps')
        # plt.yscale('log')
        plt.grid(True)
        plt.title('CPS '+ mat)
        plt.legend()
        plt.tight_layout()

plt.figure(8)
for alambre in nromed:
    plt.plot(dts[alambre]/3600, tdead[alambre], '.', label = alambre) #*np.exp(np.log(2)*(dts[alambre])/hl)
    # plt.plot(dts[alambre]/3600, np.full(len(dts[alambre]), Act_W[alambre][mat][0, 0]*np.exp(np.log(2)*(dts[alambre][0])/hl)), '-', lw = 2.0) #*np.exp(-np.log(2)*(dts[alambre]-dts[alambre][0])/hl)
    plt.xlabel('t [horas]')
    plt.ylabel('dead time %')
    # plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


#%% Cálculo de flujo total
Gth = 0.969
SSg = {'197Au': 1.035, '63Cu': 1.032, '55Mn': 1}

masas = {'W21': 0.08041, 'W06': 0.07053, 'W05': 0.07116}
err_rel_m = 0.005

tirr_err = 1 #s
 
# 19feb24: {'W17': 0.06793, 'W39': 0.06782, 'W34': 0.06920}
# 29jul24: {'W21': 0.08041, 'W06': 0.07053, 'W05': 0.07116}

Flujos_W = {}
Npadres = {}

for alambre in Act_W:
    Flujos_W[alambre] = {}
    Npadres[alambre] = Alambre_W.N_padres(masas[alambre], err_rel_m)
    for mat in Composition:
        iso = Alambre_W.act_els[mat]
        l = np.log(2)/Alambre_W.hl[iso]
        f_t = (1 - np.exp(-l*tirr))
        f_t_err = l*tirr_err*np.exp(-l*tirr)
        sigma = act.seccioneff_Maxw(act.cross_sec[mat], 38)
        A, Aerr = Act_W[alambre][iso].mean(axis = 0)
        Aerr = (1/np.sqrt(len(Act_W[alambre][iso])))*np.sqrt((Act_W[alambre][iso][:, 1]**2).mean() + np.var(Act_W[alambre][iso][:, 1], ddof=1))
        f_err = np.sqrt((Aerr/A)**2 + err_rel_m**2 + (f_t_err/f_t)**2) #parte del cálculo con errores
        Flujos_W[alambre][mat] = SSg[mat]*A/(Gth*sigma*Npadres[alambre][mat][0]*f_t)*np.array([1, f_err])

#%% Figuras de evolución ficticia del flujo

for ii, alambre in enumerate(nromed):
    for mat in Composition:
        plt.figure(9+ii)
        plt.errorbar(dts[alambre]/3600, Flujos_W[alambre][mat][:, 0]*1e-9, yerr=Flujos_W[alambre][mat][:, 1]*1e-9, 
                     fmt = '.', label = mat) #*np.exp(np.log(2)*(dts[alambre])/hl)
    plt.xlabel('td, tiempo después de irradiar [horas]')
    plt.ylabel('Flujo calculado a td [$10^9$ nv]')
    # plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.title(alambre)
    plt.tight_layout()

