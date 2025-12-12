#Recordar correr 'pip install xlrd', cambiar la direccion de
# pip install pandas
# pip install plotly
# pip install ipywidgets
# # jupyter labextension install jupyterlab-plotly

import numpy as np
import pandas as pd
from datetime import datetime
import re
import urllib.request
import io, requests
from itertools import combinations
from scipy.stats import chi2
import matplotlib.pyplot as plt

day_year = 365.24219878
s_day = 86400
N_av = 6.02214076E23

tabla_RA3 = pd.read_excel('/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente') #str(input('Ingrese dirección de tabla RA3:\n'))

Livechart = "https://nds.iaea.org/relnsd/v1/data?"

cross_sec = {'63Cu': 4.5e-24, '197Au': 98.65e-24, '55Mn': 13.36e-24}

path_API = ''

def seccioneff_Maxw(s_0, T):
    """
    Corrige la sección eficaz por temperatura, considerando una distribución Maxwelliana.

    Parameters
    ----------
    s_0 : float
        Sección eficaz térmica.
    T : float
        Temperatura en K.

    Returns
    -------
    float
        sección eficaz corregida..

    """
    T += 273.15
    T0 = round(0.0253/8.6173324e-5, 2)
    a = np.sqrt(np.pi*T0/(4*T))
    return s_0*a

def redondeo(mean, err, cs, texto = False):
    """
    Devuelve al valor medio con la misma cant. de decimales que el error (con 2 c.s.).

    Parameters
    ----------
    mean : float
        Valor medio.
    err : float
        Error.
    cs : int
        Cifras significativas.
    texto : bool, optional (D = False)
        Especificar si sale en formato de texto, para agregar el ±.

    Returns
    -------
    tuple, float or str
        (mean, err) si text = False; (mean, ±, err) en caso contrario.

    """
    digits = -np.floor(np.log10(err)).astype(int)+cs-1
    if err<1:
        err_R = format(np.round(err, decimals = digits), f'.{digits}f')
        mean_R = str(np.round(mean, decimals = len(err_R)-2))
    else:
        err_R = format(np.round(err, decimals = digits), '.0f')
        mean_R = format(np.round(mean, decimals = cs-1-len(err_R)), '.0f')
    if texto == True:
        return (mean_R, '±',err_R)
    else:
        return (float(mean_R), float(err_R))

def ajuste_pol(grado, xdata, ydata, y_err):
    """
    Resuelve en forma exacta para un modelo polinómico, lineal en parámetros.

    Parameters
    ----------
    grado : int
        Grado del polinomio a ajustar.
    xdata : array, float
        Tira de datos de variable independiente.
    ydata : array, float
        Tira de datos de variable dependiente.
    y_err : array, float
        Corresponde al error absoluto de ydata.

    Returns
    -------
    array, float
        Estimadores de los coeficientes del polinomio.
    array, float
        Desviaciones estándar de los estimadores. Sale de la matriz de covarianza.
    J_min_observado : float
        Chi cuadrado. Indica el valor en el que la función de costo llega a su mínimo.
    residuos : array, float
        Residuos del ajuste para cada adquisición/dato.
    pvalor : float
        p-value correspondiente al ajuste. Se obtiene a partir de la integral del complemento de una función chi2 de grado dof, calculada hasta el valor de J_min_observado.
    ddof : int
        Grados de libertad del problema.
    rhos : array, float
        Coeficientes de correlación.
    var_mu : function
        Función que toma como input algún array y devuelve el intervalo de confianza.

    """
    f = lambda x: np.column_stack([a*b for a, b in combinations(x, 2)])
    matrix_fromx = lambda data: np.column_stack([data**exp for exp in np.arange(grado+1)])
    cova_y = np.diag(y_err**2)
    design_matrix = matrix_fromx(xdata)
    cova_mle = np.linalg.inv(design_matrix.T @ np.linalg.inv(cova_y) @ design_matrix )
    matrix_B = cova_mle @ design_matrix.T @ np.linalg.inv(cova_y)
    par_est = matrix_B @ ydata
    par_err = np.sqrt(np.diag(cova_mle))
    comb = f(par_err) #combinatoria de stds de par_err
    upper_tri = cova_mle[np.triu_indices_from(cova_mle, k=1)] #upper triangle from covariance matrix
    rhos = upper_tri/comb #factores de correlación (01, 02, 03, ... 0max, 12, 13). Usa combinatoria de las stds y los elementos de la cov
    var_mu = lambda xfit: matrix_fromx(xfit)**2 @ np.diag(cova_mle) + 2*np.vstack([f(x) for x in matrix_fromx(xfit)]) @ upper_tri #varianza del parámetro hallado

    residuos = ydata - design_matrix @ par_est
    J_min_observado = residuos.T @ np.linalg.inv(cova_y) @ residuos #chi2
    ddof = len(xdata) - len(par_est)
    pvalor = chi2.sf(J_min_observado, ddof)

    return par_est[::-1], par_err[::-1], J_min_observado, residuos, pvalor, ddof, rhos, var_mu

def lc_pd_dataframe(url):
    """
    Lectura de API a partir de una url.

    Parameters
    ----------
    url : str
        URL de donde se van a extraer los datos.

    Returns
    -------
    DataFrame
        Devuelve los datos leídos en formato DataFrame.

    """
    try:
        urlData  = requests.get(url).content
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
        return rawData
    except:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        return pd.read_csv(urllib.request.urlopen(req))

def select_data(measured, data_table, tolerance, idx=0):
    """
    Selecciona los valores de data_table según los valores en measured, bajo un grado de tolerancia.

    Parameters
    ----------
    measured : array
        Tira de datos con valores a comparar.
    data_table : matrix
        Datos a filtrar. Debe tener más de una columna, puesto que una sirve de referencia (en pos idx).
    tolerance : float
        Menor que uno. Indica qué tan alejado del valor de referencia me permito estar.
    idx : optional, int
        Posición en columna de referencia para data_table.
    Returns
    -------
    array
        data_table, sin la columna de referencia, con las filas seleccionadas.

    """
    rows, cols = len(measured), len(data_table.T)
    data_sel = np.zeros((rows, cols))
    for jj, en in enumerate(measured):
        for ee in data_table:
            x = en/ee[idx]
            if 1-tolerance<x<1+tolerance:
                data_sel[jj] = ee
    return data_sel

def dif_rel(x1, x2):
    """
    Devuelve la diferencia media relativa entre dos valores.

    """
    return (2*(x1 - x2)/(x1 + x2))*100

def G_abs(sigma, r, h, N, dim: bool=False):
    """
    Corrección por self-shielding térmico (absorción) para geometría cilíndrica.

    Parameters
    ----------
    sigma : float,
        Sección eficaz microscópica (cm^2).
    r : float,
        Radio del cilindro.
    h : float,
        Altura del cilindro.
    N : float,
        Número de nucleídos.
    dim : bool, optional
        Si quiere que también devuelva el (D = True).

    Returns
    -------
    G : float
        Coeficiente de autoapantallamiento (val. típico < 1).

    S, V (if dim = True): float
       Valor de superficie y volumen de la geometría descrita.

    """
    S = 2*np.pi*r*h + 2*np.pi*r**2
    V = h*np.pi*r**2
    x = (N/V)*sigma*2*V/S
    Euler = 0.5772156649
    G = 1 - (4/3)*x - (x**2)*(np.log(x/2) + Euler - 5/4)/2 - (x**4)*(np.log(x/2) + Euler - 7/4)/24
    if dim:
        return G, S, V
    else:
        return G


## Revisar que no esté cargando datos de la IAEA cada vez que quiero calcular una actividad

def DDA(hl, real_time):
    """
    Calcula el factor de corrección Decay During Adquisition con la fórmula del GammaVision.

    Parameters
    ----------
    hl : float
        Halflife, tiempo de vida medio del nucleído.
    real_time : float
        Tiempo real final de la adquisición.

    Returns
    -------
    corr : float
        Factor de corrección.

    """
    f = np.log(2)*(real_time/hl)
    corr = f/(1-np.exp(-f))
    return corr

def Actividad(Energias, cps_peaks, err_cps, treal, poly_params, var_mu=None, dt: float=0.,
              tolerancia: float=0.0025, isfromRA3: bool = False, Fuente=None, datadecay=None):
    """
    Cálculo de actividades.

    Parameters
    ----------
    Energias : array, float
        Energías de los picos de las regiones de interés.
    cps_peaks : array, float
        Tasa de conteo de los picos, para las energías antes definidas.
    err_cps : array, float
        Error de la tasa de conteo.
    treal : float
        Tiempo real de adquisición.
    poly_params : array (shape= Mx2), float
        Valores de los M coeficientes del polinomio de eficiencia, ordenados en filas junto a sus errores.
    var_mu : function or float (D = None)
        Función para calcular la varianza relativa de la eficiencia (devuelto en ajuste_pol), o en su defecto, el valor del coeficiente de Pearson para un ajuste lineal.
        En cualquier otro caso, el error relativo se considera 0.
    dt : float or datetime, optional (D = 0)
        Fecha (datetime) de medición de la fuente (isfromRA = True)
        Tiempo (s) hasta el punto de referencia temporal (isfromRA = False).
    tolerancia : float, optional
        Tolerancia usada para seleccionar datos de la tabla de la IAEA. (D = 0.0025)
    isfromRA3 : bool, optional
        Indicar si es una fuente de la instalación del RA-3. (D = False)
    Fuente : str, optional
        Funcion si isfromRA3 es True. Indicar el nombre de la fuente. (D = None)
    datadecay : dict, optional
        Agregar el dict con la info devuelta de la IAEA (cruda). La key es el isótopo decae. (D = None)

    Returns
    -------
    Para isfromRA3 = True : tuple(float, float, float)
        Actividad calculada, actividad deducida por tabla del RA3, diferencia media relativa entre ambas.

    Para isfromRA3 = False : float
        Actividad final calculada.

    """
    if Fuente == None and datadecay!=None:
        eiu = datadecay.get(['energy', 'intensity', 'unc_i']).to_numpy()
        hl = datadecay.get(['half_life_sec']).to_numpy().mean()
    elif Fuente != None and datadecay == None:
        data_iaea = loadfromIAEA(Fuente, 'decay')
        eiu = data_iaea.get(['energy', 'intensity', 'unc_i']).to_numpy()
        hl = data_iaea.get(['half_life_sec']).to_numpy().mean()
    else:
        print('Ingrese el parámetro \'Fuente\', o en su defecto, los datos asociados con \'datadecay\'.')
    peaks = select_data(Energias, eiu, tolerancia)
    peaks[np.isnan(peaks)] = 0
    peaks[:, 1:] = peaks[:, 1:]/100
    effs = np.exp(np.polyval(poly_params[:, 0], np.log(peaks[:, 0])))
    if callable(var_mu):
        effs_err = np.sqrt(var_mu(np.log(peaks[:, 0])))
    elif isinstance(var_mu, float) and len(poly_params[:, 1])==2:
        effs_err = np.sqrt(np.polyval(poly_params[:, 1]**2, np.log(peaks[:, 0])**2) + 2*var_mu*np.multiply(*poly_params[:, 1])*np.log(peaks[:, 0]))
    else:
        effs_err = 0
    Act_calc = cps_peaks/(peaks[:, 1]*effs)
    try:
        Err_calc = Act_calc*np.sqrt((np.log(2)/hl)**2 + (err_cps/cps_peaks)**2 + (peaks[:, 2]/peaks[:, 1])**2 + effs_err**2)
    except:
        return (cps_peaks, peaks[:, 1])
    Act_final = DDA(hl, treal)*np.array([Act_calc, Err_calc]).T
    if isfromRA3 == True:
        data_doc = tabla_RA3.loc[Fuente][['Act       [Bq]', 'σ Act']].values.astype(float)
        data_doc[1] = data_doc[1]*data_doc[0]
        fecha_doc = tabla_RA3.loc[Fuente]['Fecha']
        dt = float((fecha_doc - dt).days)*s_day
        data_cal = data_doc*np.exp(np.log(2)*dt/hl)
        diff = dif_rel(Act_calc, data_cal[0])
        return Act_final, data_cal, diff, peaks
    else:
        return Act_final*np.exp(np.log(2)*dt/hl)

# def df_discriminator(df, text):
#     mask = np.vstack([df[col].str.contains(r"SPECTRUM", na=False) for col in df])
#     icut = np.where(mask==[True])[1][0]
#     fcut = len(df) - icut
#     df_up = pd.DataFrame([x[0].split(':') for x in df[df.index<icut].values])
#     df_up
#     for x in df[df.index<icut].values:
#         x = x[0].split(':')
#         if len(x)>=2:
#             print(x)


def check_ROI(ROI):
    if not isinstance(ROI, np.ndarray):
      if isinstance(ROI, list):
        ROIs_cal = np.array(ROI)
      elif isinstance(ROI, str):
          ROIs_cal = np.loadtxt(ROI)
      else:
          ROIs_cal = input('Formato no válido de ROI. Ingrese una lista o una dirección de archivo:\n')
    else:
      ROIs_cal = ROI
    return ROIs_cal

def add_effparams_to_obj(obj, eff_data, list_result, fuente): 
	coef, perr, chi2, res, pvalor, ddof, rhos, var_mu = list_result
	obj.eff_params[fuente] = {'an': coef, 'an_err': perr, 'chi-square': chi2, 'p-value': pvalor, 'ddof': ddof, 
                           'residuals': res, 'var_mu': var_mu, 'rhos': rhos}
	obj.eff_coef_tabla = np.vstack((coef, perr)).T
	ss_res = np.sum(res**2)
	ss_tot = np.sum((np.log(eff_data)-np.mean(np.log(eff_data)))**2)
	R2 = 1 - (ss_res / ss_tot)
	obj.eff_goodness = pd.DataFrame(np.array([chi2, pvalor, ddof, R2]), index = ['chi^2', 'p-value', 'ddof', 'R^2'], columns = ['goodness of fit'])

class fromspec:
    def __init__(self, path, coef_en=[]):
        """
        Lectura de espectro. La clase va a tener asociada las cuentas, canales, tiempo real, tiempo vivo, y coeficientes de calibración E/canales.

        Parameters
        ----------
        path : str
            Define la ruta de acceso al archivo .txt donde se encuentra el espectro.
        coef_en : list or array, optional
            Lista de coeficientes de la calibración en energía. (D = 0).

        """
        df = pd.read_csv(path, encoding='latin-1')
        mask = np.vstack([df[col].str.contains(r"SPECTRUM", na=False) for col in df])
        icut = np.where(mask==[True])[1][0]
        fcut = len(df) - icut

        df_data = pd.read_csv(path, names=['col', 'value'], index_col='col', skipfooter=fcut, encoding='latin-1', engine = 'python', delimiter=':  |\n')
        df_spec = pd.DataFrame([x[0].split() for x in df[df.index>icut].values]).set_index(0)
        self.tlive = float(df_data.loc['Live Time'].iloc[0])
        self.treal = float(df_data.loc['Real Time'].iloc[0])
        try:
            self.tinicio = datetime.strptime(df_data.loc['Acquisition start date'].iloc[0]+'_'+df_data.loc['Acquisition start time'].iloc[0], '%d-%b-%Y_%H:%M:%S')
        except:
            self.tinicio = datetime.strptime(df_data.loc['Acquisition start date'].iloc[0]+'_'+df_data.loc['Acquisition start time'].iloc[0], '%d%b%Y_%H:%M:%S')
        self.nchannels = int(df_data.loc['Number of channels'].iloc[0])
        self.channels = np.arange(self.nchannels)
        self.counts = df_spec.to_numpy(dtype=float).reshape(-1)[:self.nchannels]
        if len(coef_en)==0:
            self.coef_en =  np.array(df_data.loc['Energy Fit'].iloc[0].split('  '), dtype=float)[::-1]
        else:
            self.coef_en = coef_en[::-1]
		self.energies = np.polyval(self.coef_en, self.channels)
    def ROI(self, interval, n_bkg: int=3):
        """
        Cálculo de valores importantes dentro de una ROI.

        Parameters
        ----------
        interval : list
            Canales que definen la ROI (incluye background).
        n_bkg : int, optional
            Cantidad de puntos para el background. (D = 3)

        Returns
        -------
        peak_info : dict
            Diccionario con las cuentas básicas al seleccionar una ROI en Gammavision.

        """
        l, h = interval
        peak_info = {}
        peak_info["gross"] = np.sum(self.counts[l:h+1])
        peak_info["adj_gross"] = np.sum(self.counts[l + n_bkg:h - n_bkg+1])
        peak_info["background"] = (np.sum(self.counts[l:l + n_bkg]) + np.sum(self.counts[h - n_bkg+1:h+1]))*(h - l + 1)/(2*n_bkg)
        peak_info["net"] = peak_info["adj_gross"] - peak_info["background"]*(h + 1 - l - 2*n_bkg)/(h - l + 1)
        peak_info["net_err"] = np.sqrt(peak_info["adj_gross"] + peak_info["background"]*((h - l + 1 - 2*n_bkg)**2/(2*n_bkg*(h - l + 1))))#/((2*n_bkg)*(h - l + 1)))
        peak_info["chn_max"] = self.channels[l + np.argmax(self.counts[l:h+1])]
        peak_info["en_max"] = np.polyval(self.coef_en, peak_info["chn_max"])
        peak_info["tasa_neta"], peak_info["tasa_neta_err"] = np.array([peak_info["net"], peak_info["net_err"]])/self.tlive
        return peak_info

    def graph(self, scale: str = 'log'):
        plt.figure()
        plt.plot(self.channels, self.counts)
        plt.grid(True, ls='--')
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.yscale(scale)

class loadfromIAEA:
    def __init__(self, Fuente, state, radiation_type: str='g', only_stable: bool=True, savedata = False):
        """
        A partir de un isótopo, ve sus datos en la tabla de la IAEA usando su API.

        Reference: https://iaea-nds.github.io/lc_api_notebook/

        Parameters
        ----------
        Fuente : str
            Ingrese nombre del isótopo (e.g. 64Cu, Au-198, etc.).
        state : str
            Condición en el que se encuentra el isótopo: 'decay', 'estable'.
        radiation_type : str, optional
            Filtro para ver las tablas según el tipo de radiación. (D = 'g')
        only_stable : bool, optional
            Filtra para no ver estados excitados. (D = True)


        """
        archive = r'{}\\{}_{}_rad-{}.csv'.format(path_API, Fuente, state, radiation_type)
        nuclei = re.split('-|_| ', Fuente)[0]
        try:
            df = pd.read_csv(archive)
        except:
            path = {'decay': f"fields=decay_rads&nuclides={nuclei}&rad_types={radiation_type}",
                    'estable': f"fields=ground_states&nuclides={nuclei}"}
            df = lc_pd_dataframe(Livechart + path[state.lower()])
        self.estado = state
        self.nucleo = nuclei
        if state.lower() == 'decay' and only_stable == True:
            self.data = df.query("p_energy==0")
        else:
            self.data = df

        if savedata:
            df.to_csv(archive)

    def get(self, cols):
        """
        Devuelve columnas de interés a partir de una tabla.

        Parameters
        ----------
        cols : list, str
            Nombres de los encabezados de las columnas de interés.

        Returns
        -------
        df : DataFrame
            Datos seleccionados de lo obtenido por la IAEA.

        """
        df = self.data[cols]
        df = df[df[cols].notna()]
        return df

class HPGE_calib:
    def __init__(self, det, spec_cal_path, spec_fondo_path, Fuente: str = 'Eu152_76044A-440'):
        """
        Calibración en eficiencia del detector HPGe.

        Parameters
        ----------
        Fuente : str
            Nombre de serie de la fuente usada para la calibración. Ejemplo: 'Eu152_76044A-440'.
        FechaCalib : datetime
            Fecha de calibración de la Fuente R.

        """
        self.spec_cal = fromspec(spec_cal_path)
        self.spec_fondo = fromspec(spec_fondo_path)
        self.det = det
        self.fuente = Fuente
        self.datafromIAEA = loadfromIAEA(Fuente, 'decay', savedata=True).data
        self.fecha_cal = self.spec_cal.tinicio.date()
        self.fecha_doc = tabla_RA3.loc[Fuente]['Fecha'].date()
        self.dt_caldoc = float((self.fecha_doc - self.fecha_cal).days)*s_day
        intensity, unc_int, energy, hl = self.datafromIAEA.get(['intensity', 'unc_i', 'energy', 'half_life_sec']).to_numpy().T
        self.halflife_s = hl.mean()
        self.act_doc = tabla_RA3.loc[Fuente][['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_doc[1] = self.act_doc[1]*self.act_doc[0]
        self.act_cal = self.act_doc*np.exp(np.log(2)*self.dt_caldoc/self.halflife_s)
        self.int_energy = np.transpose([energy, intensity, unc_int])
        self.eff_params = dict()


    def __efficiencycalc__(self, ROIs, grado_pol: int = 1, n_bkg: int = 3, criterio: float=0, tolerancia: float=0.0025, fit: bool= True):
      """
      Cálculo de la eficiencia a partir de ROIs definidas previamente.

      Parameters
      ----------
      ROIs : matrix
          De dimensión N x 2, con N el nro. de ROIs o picos.
      spec_cal : class
          Clase con el espectro ya leído (fromspec).
      spec_fondo : class
          Clase con el espectro del fondo ya leído (fromspec).
      grado_pol : int, optional
          Grado del polinomio considerado para la eficiencia (D = 1).
      n_bkg : int, optional
          Número de puntos de fondo por ROI (D = 3).
      criterio : float, optional
          Valor de corte (cota inferior) para considerar intensidades (D = 0).
      tolerancia : float, optional
          Valor de tolerancia para comparar entre energías (tabla IAEA vs pico máx en ROIs). (D = 0.0025).
      fit : bool, optional
          Si se quere hacer el ajuste, o si sólo se quiere recuperar la eficiencia en función de la energía.


      Returns
      -------
      eff : array, float
          Valores calculados de eficiencia en función de la energía.
      eff_err : array, float
          Errores de 'eff'. Calculados por propagación de errores.

      """

      cps_net = np.zeros(len(ROIs))
      Epeak_spec = np.zeros(len(ROIs))
      cps_net_err = np.zeros(len(ROIs))
      for ii, roi in enumerate(ROIs):
          # roi = (roi + n_bkg*np.array([-1, 1])).astype(int)
          data_ROI = {'Fuente': self.spec_cal.ROI(roi, n_bkg), 'Fondo': self.spec_fondo.ROI(roi, n_bkg)}
          cps_net[ii] = data_ROI['Fuente']['tasa_neta'] - data_ROI['Fondo']['tasa_neta']
          cps_net_err[ii] = np.sqrt(data_ROI['Fuente']['tasa_neta_err']**2 + data_ROI['Fondo']['tasa_neta_err']**2 )
          Epeak_spec[ii] = data_ROI['Fuente']['en_max']
      i_sel = self.int_energy[:, 1]>criterio*100
      I_E_IAEA = np.array(self.int_energy[i_sel])
      data_sel = select_data(Epeak_spec, I_E_IAEA, tolerancia)/100
      eff = cps_net/(self.act_cal[0]*data_sel[:, 1])
      try:
          eff_err = eff*np.sqrt((cps_net_err/cps_net)**2 + (data_sel[:, 2]/data_sel[:, 1])**2
                                + (self.act_cal[1]/self.act_cal[0])**2)
      except:
          eff_err = None
      if fit:
        list_eff_res = ajuste_pol(grado_pol, np.log(data_sel[:, 0]*100), np.log(eff), y_err=eff_err/eff)
        add_effparams_to_obj(self, eff_data = eff, list_result = list_eff_res, fuente = self.fuente)
      return eff, eff_err

    def __graph_eff__(self, Energies, fuente_sel, grado: int = 1):
      eff_params = self.eff_params[fuente_sel]
      E_arr = np.linspace(Energies.min(), Energies.max(), num=1000)
      logE_arr  = np.log(E_arr)
      eff_eval = np.polyval(eff_params['an'], logE_arr)
      sigma_mu_est = np.sqrt(eff_params['var_mu'](logE_arr))
      i0 = 0
      plt.figure(figsize = (5,5))
      for dd in self.npicos:
          i1 = self.npicos[dd] + i0
          plt.errorbar(np.log(Energies[i0:i1]), np.log(self.eff_data)[i0:i1], yerr= self.err_eff[i0:i1]/self.eff_data[i0:i1], fmt='.', label = f'{dd}')
          i0 = i1
      plt.plot(logE_arr, eff_eval, 'tab:gray', linewidth = 2.0, label = 'Model Fit')
      plt.fill_between(logE_arr, eff_eval-sigma_mu_est, eff_eval+sigma_mu_est, color='tab:gray', alpha=0.2)
      plt.grid(True, ls = '--')
      plt.ylabel('$ln$ Eff')
      plt.xlabel('$ln$ E')
      plt.legend()
      plt.title('Eficiencia para ' + f'{self.det}'+ ' 3sep ' + 'Fit grado= {}'.format(grado))# + f'{Fecha_cal.date()}')
      plt.tight_layout()

    def eff(self, ROI, ajustar: bool = True, grado: int = 1, graph: bool = True):
      self.grado_ajuste_eff = grado
      ROIs_cal = check_ROI(ROI)
      while isinstance(ROIs_cal, (str, list)):
          ROIs_cal = check_ROI(ROI)
      self.eff_Epeak_Eu = ROIs_cal[:, 0]
      if ajustar:
        self.eff_data, self.err_eff = self.__efficiencycalc__(ROIs_cal[:, 1:].astype(int), fit=ajustar)
        self.npicos = {self.fuente.split('_')[0]: len(self.err_eff)} 
      if graph:
        self.__graph_eff__(ROIs_cal[:, 0], grado = self.grado_ajuste_eff, fuente_sel = self.fuente)

    def eff_adddata(self, path_new_spec, ROIs, npicos, fuente, ajuste: bool = True, graph: bool = True):
      ROIs_cal = check_ROI(ROIs).astype(int)
      while isinstance(ROIs_cal, (str, list)):
          ROIs_cal = check_ROI(ROIs).astype(int)
      # Access the ROI method on self.fondo
      spec_incog = fromspec(path_new_spec, coef_en = self.spec_cal.coef_en[::-1])
      pico_incog = [spec_incog.ROI(ROIs_cal)] if ROIs_cal.shape == (2,) else [spec_incog.ROI(roi) for roi in ROIs_cal]
      E_incog = np.array([pico_incog[ii]['en_max'] for ii in range(npicos)]) #energía del pico
      t_inicio = spec_incog.tinicio.replace(hour=0, minute=0, second=0, microsecond=0) #Fecha de espectro '00 hs'
      data_fondo = [self.spec_fondo.ROI(ROIs_cal)] if ROIs_cal.shape == (2,) else [self.spec_fondo.ROI(roi) for roi in ROIs_cal]
      cps_fondo = [np.array([back["tasa_neta"], back["tasa_neta_err"]]) for back in data_fondo]
      cps_pico = np.array([np.array([pico_incog[ii]["tasa_neta"], pico_incog[ii]["tasa_neta_err"]]) - cps_fondo[ii] for ii in range(npicos)])
      data_act = Actividad(E_incog, cps_pico[:, 0], cps_pico[:, 1], spec_incog.treal, self.eff_coef_tabla,
                            isfromRA3=True, Fuente=fuente, dt = t_inicio, var_mu = self.eff_params[self.fuente]['var_mu'])
      df_act = pd.DataFrame(np.hstack((data_act[3], data_act[0].reshape((npicos, 2)), np.array(data_act[2]).reshape((npicos, 1)))),
                            columns = ['Energy [keV]', 'BR %', 'BR err %', 'Act [Bq]', 'Error Act [Bq]', 'Diff Tab %'])
      df_act[['BR %', 'BR err %']] = df_act[['BR %', 'BR err %']]*100

      setattr(self, f'data_act_{fuente[2:]}', df_act)
      setattr(self, f'data_RA3_{fuente[2:]}', data_act[1])

      eff_incog = cps_pico[:, 0]/(data_act[1][0]*data_act[3][0][1])
      eff_incog_err = eff_incog*np.sqrt((data_act[3][:, 2]/data_act[3][:, 1])**2 +
                        (cps_pico[:, 1]/cps_pico[:, 0])**2 +
                        np.divide(*data_act[1][::-1])**2)

      self.eff_data = np.append(self.eff_data, eff_incog)
      self.err_eff = np.append(self.err_eff, eff_incog_err)
      self.eff_Epeak_Eu = np.append(self.eff_Epeak_Eu, E_incog)
      self.npicos[fuente.split('_')[0]] = len(eff_incog_err) 

      if ajuste:
        list_eff_res = ajuste_pol(self.grado_ajuste_eff, np.log(self.eff_Epeak_Eu), np.log(self.eff_data), y_err=self.err_eff/self.eff_data)
        add_effparams_to_obj(self, eff_data = self.eff_data, list_result = list_eff_res, fuente = fuente)
      if graph:
        self.__graph_eff__(self.eff_Epeak_Eu, grado = self.grado_ajuste_eff, fuente_sel = fuente)


class Alambre:
    def __init__(self, data_det, composition, irradiation_time, spec_path, t_inicio, dn: int=1):
      '''
      Para crear un objeto de la clase Alambre necesito:

        Parameters
        ----------
        data_det : obj
            Objeto de la clase HPGE_calib. Se espera que ya se haya hecho la calibración del detector de Germanio utilizado.
        composition : dict, float
            Tiene como keys los componentes estables del alambre (por ej., '56Mn' o '63Cu'), y como valor el %m/m (del 0 al 1)
        irradiation_time : int
            En segundos. El tiempo total de irradiación.
        spec_path : str
            Ruta de archivo del espectro del alambre.
        t_inicio : datatime
            Timestamp o vector de tiempos donde señala el inicio de la irradiación.
        dn : int, optional
            Cantidad de neutrones absorbidos. Cambio de A en la reacción. (D = 1).
        '''
      #Datos de composición, nucleidos, tiempo de vida medio y abundancia:
      self.comp = composition
      self.A_stables = {x: int(re.findall(r'\d+', x)[0]) for x in self.comp}
      self.A_activated = {x: x.replace(str(self.A_stables[x]), str(self.A_stables[x]+dn)) for x in self.comp}

      #tablas de datos: isótopo y núcleo estable
      self.stable_data = {x: loadfromIAEA(x, 'estable') for x in self.comp}
      self.data_decay = {iso: loadfromIAEA(iso, 'decay') for iso in list(self.A_activated.values())}

      #Extraigo datos específicos de las tablas: abundancia y tiempo de vida medio
      self.abundance = {x: self.stable_data[x].data['abundance'][0]/100 for x in self.comp}
      self.hl = {iso: self.data_decay[iso].get(['half_life_sec']).to_numpy().mean() for iso in list(self.A_activated.values())}

      #Datos del detector de germanio, ruta del espectro del alambre, tiempo de inicio de irradiación, y tiempo de irradiación total
      self.data_det = data_det
      self.spec_path = spec_path
      self.t_inicio = t_inicio
      self.ti = irradiation_time  

    def pico(self, ROI, graph_ROI: bool = False):
      '''
      Extrae datos (tasa, energía) de un pico/roi dado, distinguiendo por isótopo formado.

      Parameters
      ----------
      ROI : dict, list
          Distingue según isótopo. A cada uno se le asigna las listas con valores de la ROI. Por ejemplo:
          {'64Cu': [290, 300], '198Au': [250, 262]}
      graph_ROI : bool, optional
          Si quiere que haya un gráfico que destaque el pico. Por defecto es False.

      Returns
      -------
      DataFrame
          Tabla con los .

      '''
      net_cps = {}
      err_cps = {}
      energy = {}
      data = fromspec(self.spec_path, coef_en = self.data_det.spec_cal.coef_en[::-1])
      self.spec_alambre = data
      #tiempo de irradiación + tiempo de detección/medición
      self.dt = (data.tinicio - self.t_inicio).total_seconds()
      self.rois = ROI
      for mat, roi in ROI.items():
          cps_fondo, cps_fondo_err = np.array(list(map(self.data_det.spec_fondo.ROI(roi).get, ['tasa_neta', 'tasa_neta_err'])))
          data_roi = data.ROI(roi)
          net, err, energy[mat] = np.array(list(map(data_roi.get, ["tasa_neta", "tasa_neta_err", "en_max"])))
          net_cps[mat] = net - cps_fondo
          err_cps[mat] = np.sqrt(err**2 + cps_fondo_err**2)
      self.tasa_pico = pd.DataFrame.from_dict(net_cps, orient='index', columns = ['tasa'])
      self.tasa_pico_err = pd.DataFrame.from_dict(err_cps, orient='index', columns = ['tasa_err'])
      self.Emax_picos = pd.DataFrame.from_dict(energy, orient='index', columns = ['Emax'])

      idx = np.array(list(ROI.values()))
      if graph_ROI:
        self.idx_picos = {}
        plt.figure()
        plt.plot(data.channels[idx.min()-100:idx.max()+100], data.counts[idx.min()-100:idx.max()+100])
        for ii, iso in zip(idx, ROI.keys()):
          idx_cond = np.logical_and(data.channels> ii[0], data.channels<ii[1])
          self.idx_picos[iso] = idx_cond
          plt.plot(data.channels[idx_cond], data.counts[idx_cond], label = f'ROI_{iso}')
        plt.grid(True, ls='--')
        plt.xlabel('Channel')
        plt.ylabel('Counts')
        plt.legend()
      return pd.concat([self.tasa_pico, self.tasa_pico_err], axis=1)

    def calcular_act(self, sel_eff = None):
      Act_iso = pd.DataFrame(columns = ['Fuente', 'Energía [keV]', 'Act [Bq]', 'dAct [Bq]'])
      if sel_eff:
        eff_params = self.data_det.eff_params[sel_eff]
      else: 
        eff_params = list(self.data_det.eff_params.values())[-1]
      #Act_iso.set_index('Energy', inplace= True)
      for iso in self.rois.keys():
        if isinstance(self.tasa_pico.loc[iso, 'tasa'], float):
          net, err = [np.array([self.tasa_pico.loc[iso, 'tasa']]), np.array([self.tasa_pico_err.loc[iso, 'tasa_err']])]
          e = np.array([self.Emax_picos.loc[iso, 'Emax']])
        else:
          net, err = [self.tasa_pico.loc[iso, 'tasa'], self.tasa_pico_err.loc[iso, 'tasa_err']]
          e = self.Emax_picos.loc[iso, 'Emax']
        A = Actividad(e, net, err, dt = self.dt, treal = self.spec_alambre.treal,
                      poly_params = self.data_det.eff_coef_tabla, var_mu = eff_params['rhos'][0][0],
                      Fuente = iso) #en var_mu ingresamos el valor del coef de Pearson
        for ii, ee, aa in zip(list(range(len(e))), e, A):
          Act_iso.loc[ii] = np.hstack((iso, ee, aa))
      self.tabla_Act = Act_iso
      return Act_iso

    def __Npadres__(self, masa_total, masa_errrel):
      """
      Calcula la cantidad de núcleos padres para cierta masa, teniendo en cuenta la abundancia y la composición previamente definidas.

      Parameters
      ----------
      masa_total : float
          Masa (g) del alambre.
      masa_errrel : float
          Error relativo (%) de masa total

      Returns
      -------
      N : dict, {str: array}
          Cantidad de nucleos por nucleído estable que compone al alambre, junto a su error.
          {nucleído padre: nro. de núcleos}

      """
      m_parcial = np.array([self.comp[el]*self.abundance[el] for el in self.comp])*masa_total
      Mr = np.array([self.stable_data[el].data.get('atomic_mass')[0]*1e-6 for el in self.comp])
      Nmean = N_av*m_parcial/Mr
      N = {x:y for x,y in zip(self.comp, Nmean.reshape((len(Nmean), 1))*np.array([1, masa_errrel]))}
      return N

    #Gth = 
    def calc_flujo(self, masa, masa_err_rel, tirr_err: float = 1.0, 
                   Gth: float = 0.969, SSg = {'197Au': 1.035, '63Cu': 1.032, '55Mn': 1}):
      Flujos = pd.DataFrame(columns = ['Fuente', 'Flujo [nv]', 'dFlujo [nv]'])
      N0 = self.__Npadres__(masa, masa_err_rel)
      for ii, mat in enumerate(self.comp.keys()):
        iso = self.A_activated[mat]
        l = np.log(2)/self.hl[iso]
        f_t = (1 - np.exp(-l*self.ti))
        f_t_err = l*tirr_err*abs(1 - f_t)
        sigma = seccioneff_Maxw(cross_sec[mat], 38)
        if iso in list(self.rois.keys()):
          A, Aerr = self.tabla_Act[self.tabla_Act['Fuente'] == iso][['Act [Bq]', 'dAct [Bq]']].to_numpy(dtype = float).mean(axis = 0)
          f_err = np.sqrt((Aerr/A)**2 + masa_err_rel**2 + (f_t_err/f_t)**2) #parte del cálculo con errores
          Flujos.loc[ii] = np.hstack((iso, (SSg[mat]*A/(Gth*sigma*N0[mat][0]*f_t))*np.array([1, f_err])))
        else:
          continue 
          #print(' --- Nuclei:', mat, '---', '\n SSg:', SSg[mat], '\n A:', [A, Aerr], '\n Gth:', Gth, '\n sigma:', sigma,
          #      '\n Npadres:', N0[mat], '\n f(ti, td):', f_t, '\n error:', np.array([1, f_err]), '\n')
      return Flujos

def tlive_estimation(tirr, sigma, Egamma, BR, Npadres, hl, flujo, Sg: float = 1.03, Gth: float = 0.969, td: float = 900,
                     Net_min: float = 1E4, coef_table = np.array([[-1.187, 0.02], [ 1.29, 0.13]])):
  '''




  Parameters
  ----------
  tirr : float
      Tiempo [s] de irradiación.
  sigma : float
      Sección eficaz (n, gamma) por absorción.
  Egamma : float
      Energía [keV] a la que salen los gammas.
  BR : float
      Branching ratio (entre 0 y 1).
  Npadres : float
      Núcleos padres del compuesto.
  hl : float
      Vida media del isótopo que se forma (N+1 del núcleo estáble).
  flujo : float
      Flujo [nv] promedio de la fuente de neutrones.
  Sg : float, opcional
      Pérdida de energía dentro del alambre. (D = 1.03).
  Gth : float, opcional
      Autoapantallamiento de neutrones. (D = 0.969).
  td : float, opcional
      Tiempo [s] de espera desde que dejó de irradiar. (D = 900).
  Net_min : float, opcional
      Mínimo de cuentas netas a medir. (D = 10000).
  coef_table : ndarray, matrix, opcional
      Tabla con coeficientes de eficiencia. [[a1,  a1 error], [a0, a0 error]]
      (D = np.array([[-1.191,  0.02 ], [ 0.87 ,  0.13 ]]).

  Returns
  -------
  t_live : float
      Estimación de tiempo vivo [s].

  '''
  l = np.log(2)/hl
  f_t = np.exp(-l*td)*(1 - np.exp(-l*tirr))
  eff = np.exp(np.polyval(coef_table[:, 0], np.log(Egamma)))
  # sigma = seccioneff_Maxw(sigma, 38)
  t_live = Sg*Net_min/(Gth*flujo*BR*eff*f_t*sigma*Npadres)
  return t_live

