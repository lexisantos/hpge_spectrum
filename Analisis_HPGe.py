import sys
sys.path.append('') #ingrese ubicación de Repo_HPGe.py

import Repo_HPGe as act
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

#%%Fondo para UNA adq de fondo
# Llamo a los espectros del fondo
path_fondo_OFF = ".Txt" #ruta absoluta
path_fondo_ON = ".Txt" #ruta absoluta

Fondo_OFF = act.fromspec(path_fondo_OFF)
Fondo_ON = act.fromspec(path_fondo_ON)

#%% FONDO - JOBs y paso temporal de cps 
nro_med = 64 #Cambiar por la cantidad de jobs tomados para el fondo
medno = ["%.2d" % i for i in range(nro_med)]
ROI_Ar = [2204, 2217] #Chequear posición del pico de 41Ar
path = lambda nro: f"med0{nro}.Txt" #Cambiar ruta de archivo
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

#Gráfico para definir el job de corte ON/FF
plt.figure()
plt.errorbar(t_partes_h, cps_ROIs[:, 0], yerr=cps_ROIs[:, 1], fmt='.', label = 'Ar-41')
plt.xlabel('t [h]')
plt.ylabel('cps')
plt.grid(ls='--')
plt.title(f'Ar-41 en ROI = {ROI_Ar}')

#%% Suma de fondos 
## Definido el job de corte, sumo todos los fondos OFF y ON para obtener dos espectros distintos.

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
path_Eu152 = ".Txt" #Agregar dirección del espectro en .Txt (Export from Gammavision)
path_Roical = ".txt" #Pseudo tabla con | 'Energía_pico', 'Canal_inicio', 'Canal_final' | por cada ROI

ROIs_cal = np.loadtxt(path_Roical)
spec_cal = act.fromspec(path_Eu152) #Espectro de la fuente 

Fuente_pat = 'Eu152_76044A-440' #Nombre de serie que indica la tabla del RA-3
Fecha_cal = datetime(2024, 7, 31, 0, 0, 0) #día de la calibración (sólo días)
datos_cal = act.NAA_calib(Fuente_pat, Fecha_cal)

Epeak_Eu = ROIs_cal[:, 0]
ROIs_cal = ROIs_cal[:, 1:].astype(int)

#%% Eff - reporte
# =============================================================================
# CÁLCULO DE EFICIENCIA
# =============================================================================

grado_pol = 1

eff_fit = datos_cal.cal_eff(ROIs_cal, spec_cal, grado_pol, criterio = 0.025, tolerancia = 0.05) #Calcula la eficiencia en función de las energías de los picos.

eff_data, err_eff, coef, coef_err, chi2, data_sel, residual, pvalor, ddof, rhos, var_mus = eff_fit

for jj in range(len(coef)):
    print(f'a{grado_pol-jj} = '+ ' '.join([*act.redondeo(coef[jj], coef_err[jj], 2, texto=True)])+ 
          f' keV^-{grado_pol-jj}')
print(f'χ2 = {chi2}, ddof = {ddof}, p-value = {pvalor}. \n Válido en E = ({Epeak_Eu.min().round(2)}-{Epeak_Eu.max().round(2)}) keV')

E_arr = np.linspace(Epeak_Eu.min(), Epeak_Eu.max(), num=1000) #Array de energías en el rango de interés
logE_arr  = np.log(E_arr) 
eff_eval = np.polyval(coef, logE_arr) #Modelo de eff usando estimadores obtenidos
sigma_mu_est = np.sqrt(var_mus(logE_arr)) #Intervalo de confianza del ajuste

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
coef = np.array([-1.168,  0.70]) #Puede cambiarse por un array fijo, si es que ya se ejecutó el bloque anterior (o para reusar ctes de otra calibración)
#29jul24: np.array([-1.159,  0.64]), 19feb24: array([-1.17555463,  0.74364904])

coef_err = np.array([0.016,	0.11]) 
#29jul24: np.array([0.022,	0.14]), 19feb24: array([0.0195103 , 0.12846296])

coef_tabla = np.vstack((coef, coef_err)).T #Coeficientes de eficiencia en una tabla.

fecha_incog = datetime(2024, 7, 31) #Fecha de medición del patrón

n_back = 3 #Nro de puntos para el background en cada ROI

ROIs = {'137Cs': [1108, 1126]} #Defino ROI para la incógnita
Comp_incog = {'137Cs': 1} 
Incog = act.Alambre(Comp_incog, 0) #Como si fuese un alambre de actividad desconocida, de composición 1 y tiempo de irradiación 0.

## Recordar cambiar por el tipo de FONDO (OFF, ON)
Fondo_net = {mat:Fondo_OFF.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs} #Extraigo las cuentas netas por ROI del FONDO (net, net err)
cps_Fondo = {mat:np.array([Fondo_net[mat][xx] for xx in ['net', 'net_err']])/Fondo_OFF.tlive for mat in ROIs} #Tasa de conteo del FONDO por ROI

path_Cs137 = ".Txt" #Dirección del espectro de la incógnita
spec_incog = act.fromspec(path_Cs137, coef_en) #Levanto espectro
pico_incog = {mat: spec_incog.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs} #Datos de las ROIs

E_incog = {mat: pico_incog[mat]['en_max'] for mat in ROIs} #Energía del pico por ROI

cps_pico_137Cs = np.array([pico_incog['137Cs'][xx] for xx in ['net', 'net_err']])/Fondo_OFF.tlive - cps_ROIs['137Cs'] #CPS neto del pico de Cesio (bah, todavía no lo sé)

Act_incog, Act_RA3, diff = act.Actividad(*np.array([E_incog, *cps_pico_137Cs]).reshape((3, 1)), spec_incog.treal, coef_eff, isfromRA3=True, Fuente='Cs137_76071-440', dt = fecha_incog) #Actividad del pico incógnita, Act reportado corrido en el tiempo, diferencia media relativa


#%% ALAMBRES - Defino ROIs y calculo sus cps de fondo
# =============================================================================
# ALAMBRES
# =============================================================================

Composition = {'63Cu': 0.9845, '197Au': 0.0155} #, '40Ar': 0.0
tirr = 1802

# Llamo a los picos según tabla de IAEA por API, en base a la composición del alambre
Alambre_W = act.Alambre(Composition, tirr) 
n_back = 3

ROIs = {'198Au': [680-n_back, 686+n_back], '64Cu': [844-n_back, 868+n_back]}
# 29jul24: {'198Au': [680-n_back, 686+n_back], '64Cu': [844-n_back, 868+n_back]}
# 19feb24: {'198Au': [567, 578], '64Cu': [705, 729]}

#Puedo usar la calibración en eficiencia recién hecha o arrays ya definidos
coef_eff = coef#np.array([-1.159,  0.64])
coef_eff_err = coef_err #np.array([0.022,	0.14])

Fondo_ROIs_OFF = {mat:Fondo_OFF.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs}
cps_ROIs_OFF = {mat:np.array([Fondo_ROIs_OFF[mat][xx] for xx in ['net', 'net_err']])/Fondo_OFF.tlive for mat in ROIs}

Fondo_ROIs_ON = {mat:Fondo_ON.ROI(ROIs[mat], n_bkg=n_back) for mat in ROIs}
cps_ROIs_ON = {mat:np.array([Fondo_ROIs_ON[mat][xx] for xx in ['net', 'net_err']])/Fondo_ON.tlive for mat in ROIs}

#%% Llamo a las partes del archivo y calculo la actividad en cada uno
coef_en = np.array([1.769383E+001, 5.759117E-001, 3.971684E-007]) #constantes de cal en Energía
#31jul24 deforme: np.array([17.5278, 0.575533, 3.6319e-7])  

nromed = {'W21': 14, 'W06': 5, 'W05': 11} #jobs o mediciones por cada alambre
cps_Alambres = {}
Act_W = {}
dts = {}
t_inicio = datetime(2024, 7, 29, 10, 35, 12) #Hora en que se dejó de irradiar.

tdead = {}
path_job = lambda alambre, med: f"med0{med}.Txt"

for alambre in nromed: #Tener cuidado si es que hay un pase FONDO ON->OFF en el medio de alguna medición
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
            Act_W[alambre][mat][ii] = Alambre_W.Act_alambre(mat, np.array([data_ROI['en_max']]), np.array([Net_cps]), Err_cps, dt[ii], data.treal, coef_eff)
    dts[alambre] = dt 
    tdead[alambre] = deadtime
    
#%% Gráficos por Alambre

for alambre in nromed: #Seguimiento de la actividad inicial, calculada después de cierto dt (debería dar constante)
    for ii, mat in enumerate(ROIs):
        plt.figure(ii)
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
    
for alambre in nromed: #Lo mismo que antes, sólo que CPS en vez de Actividad (ver si son consistentes)
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

plt.figure(99) #Ver la evolución del tiempo muerto para cada alambre.
for alambre in nromed:
    plt.plot(dts[alambre]/3600, tdead[alambre], '.', label = alambre) 
    plt.xlabel('t [horas]')
    plt.ylabel('dead time %')
    # plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


#%% Cálculo de flujo por job

Gth = 0.969
SSg = {'197Au': 1.035, '63Cu': 1.032}

masas = {'W17': 0.06793, 'W39': 0.06782, 'W34': 0.06920}
err_rel_m = 0.005
 
# 19feb24: {'W17': 0.06793, 'W39': 0.06782, 'W34': 0.06920}
# 29jul24: {'W21': 0.08041, 'W06': 0.07053, 'W05': 0.07116}

Flujos_W = {}
Npadres = {}
G_s = {}

for alambre in nromed:
    Flujos_W[alambre] = {}
    G_s[alambre] = {}
    Npadres[alambre] = Alambre_W.N_padres(masas[alambre], err_rel_m)
    for mat in Composition:
        iso = Alambre_W.act_els[mat]
        f_t = (1 - np.exp(-np.log(2)*tirr/Alambre_W.hl[iso]))
        sigma = act.seccioneff_Maxw(act.cross_sec[mat], 38)
        A, Aerr = Act_W[alambre][iso].reshape(-1)
        f_err = np.sqrt((Aerr/A)**2 + err_rel_m**2) #parte del cálculo con errores
        Flujos_W[alambre][mat] = SSg[mat]*A/(Gth*sigma*Npadres[alambre][mat][0]*f_t)*np.array([1, f_err])

#%% Figuras de evolución ficticia del flujo (debería ser ~constante, como la Act inicial)

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
