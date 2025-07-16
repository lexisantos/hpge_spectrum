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

day_year = 365.24219878
s_day = 86400 
N_av = 6.02214076E23

tabla_RA3 = pd.read_excel('D:/Documentos RA-3/Copia de Listado de fuentes v13.xls',
                          sheet_name='Fuentes', index_col='Fuente') #str(input('Ingrese dirección de tabla RA3:\n'))

Livechart = "https://nds.iaea.org/relnsd/v1/data?"

cross_sec = {'63Cu': 4.5e-24, '197Au': 98.65e-24, '55Mn': 16.36e-24}

path_API = 'D:\\Codigos_py\\Repositorio\\data_API'

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
    var_mu = lambda xfit: matrix_fromx(xfit)**2 @ np.diag(cova_mle) + 2*np.row_stack([f(x) for x in matrix_fromx(xfit)]) @ upper_tri #varianza del parámetro hallado 
    
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
    Err_calc = Act_calc*np.sqrt((np.log(2)/hl)**2 + (err_cps/cps_peaks)**2 + (peaks[:, 2]/peaks[:, 1])**2 + effs_err**2)
    Act_final = DDA(hl, treal)*np.array([Act_calc, Err_calc]).T
    if isfromRA3 == True:
        data_doc = tabla_RA3.loc[Fuente][['Act       [Bq]', 'σ Act']].values.astype(float)
        data_doc[1] = data_doc[1]*data_doc[0]
        fecha_doc = tabla_RA3.loc[Fuente]['Fecha']
        dt = float((fecha_doc - dt).days)*s_day
        data_cal = data_doc*np.exp(np.log(2)*dt/hl)
        diff = dif_rel(Act_calc, data_cal[0])
        return Act_final, data_cal, diff
    else:
        return Act_final*np.exp(np.log(2)*dt/hl)

# def df_discriminator(df, text):
#     mask = np.row_stack([df[col].str.contains(r"SPECTRUM", na=False) for col in df])
#     icut = np.where(mask==[True])[1][0]
#     fcut = len(df) - icut
#     df_up = pd.DataFrame([x[0].split(':') for x in df[df.index<icut].values]) 
#     df_up 
#     for x in df[df.index<icut].values:
#         x = x[0].split(':')
#         if len(x)>=2:
#             print(x)
    
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
        mask = np.row_stack([df[col].str.contains(r"SPECTRUM", na=False) for col in df])
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
        return peak_info
 
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

class NAA_calib:
    def __init__(self, Fuente, FechaCalib):
        """
        Calibración en eficiencia del detector HPGe.

        Parameters
        ----------
        Fuente : str
            Nombre de serie de la fuente usada para la calibración. Ejemplo: 'Eu152_76044A-440'.
        FechaCalib : datetime
            Fecha de calibración de la Fuente R.

        """
        self.fuente = Fuente
        self.datafromIAEA = loadfromIAEA(Fuente, 'decay', savedata=True).data
        self.datafromRA3 = tabla_RA3.loc[Fuente]
        self.fecha_cal = FechaCalib
        self.fecha_doc = self.datafromRA3['Fecha']
        self.dt_caldoc = float((self.fecha_doc - self.fecha_cal).days)*s_day
        intensity, unc_int, energy, hl = self.datafromIAEA.get(['intensity', 'unc_i', 'energy', 'half_life_sec']).to_numpy().T
        self.halflife_s = hl.mean()
        self.act_doc = self.datafromRA3[['Act       [Bq]', 'σ Act']].values.astype(float)
        self.act_doc[1] = self.act_doc[1]*self.act_doc[0]
        self.act_cal = self.act_doc*np.exp(np.log(2)*self.dt_caldoc/self.halflife_s)
        self.int_energy = np.transpose([energy, intensity, unc_int])
        #np.array([np.mean(hl), np.mean(self.datafromRA3.get('unc_hls'))])
        # self.dt_caldoc = dt_caldoc
            
    def cal_eff(self, ROIs, spec_cal, spec_fondo, grado_pol: int = 1, n_bkg: int = 3, criterio: float=0, tolerancia: float=0.0025, only_data: bool = False):
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
        only_data : bool, optional
            Si no se quere hacer el ajuste y sólo se quiere recuperar la eficiencia en función de energía.

        Returns
        -------
        eff : array, float
            Valores calculados de eficiencia en función de la energía.
        eff_err : array, float
            Errores de 'eff'. Calculados por propagación de errores.
        coef : array, float
            Coeficientes calculados del ajuste polinómico.
        perr : array, float
            Errores de 'coef'.
        chi2 : float
            Mínimo de función de costo.
        data_sel : array, float
            Intensidades seleccionadas usando 'criterio', con sus errores.
        res : array, float
            Residuos del ajuste.
        pvalor : float
            p-value del ajuste.
        ddof : int
            Grados de libertad del sistema.
        rhos : array, float
            Coeficientes de correlación.
        var_mu : function
            Función que toma como input algún array y devuelve el intervalo de confianza.

        """
        cps_net = np.zeros(len(ROIs))
        Epeak_spec = np.zeros(len(ROIs))
        cps_net_err = np.zeros(len(ROIs))
        for ii, roi in enumerate(ROIs):
            # roi = (roi + n_bkg*np.array([-1, 1])).astype(int)
            data_ROI = {'Fuente': spec_cal.ROI(roi, n_bkg), 'Fondo': spec_fondo.ROI(roi, n_bkg)}
            cps_net[ii] = data_ROI['Fuente']['net']/spec_cal.tlive - data_ROI['Fondo']['net']/spec_fondo.tlive
            cps_net_err[ii] = np.sqrt((data_ROI['Fuente']['net_err']/spec_cal.tlive)**2 + (data_ROI['Fondo']['net_err']/spec_fondo.tlive)**2 )
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
        if only_data:
            return eff, eff_err
        else:   
            coef, perr, chi2, res, pvalor, ddof, rhos, var_mu = ajuste_pol(grado_pol, np.log(data_sel[:, 0]*100), np.log(eff), y_err=eff_err/eff)
            self.eff_params = {'an': coef, 'an_err': perr, 'chi-square': chi2, 'p-value': pvalor, 'ddof': ddof,
                               'residuals': res, 'grado_pol': grado_pol, 'var_mu': var_mu}
            return eff, eff_err, coef, perr, chi2, data_sel, res, pvalor, ddof, rhos, var_mu


class Alambre:
    def __init__(self, composition, irradiation_time, dn: int=1):
        """
        Crea el class asociado a un alambre.

        Parameters
        ----------
        composition : dict, {str: float}
            Composición del alambre según sus nucleídos estables. Por ej., {'63Cu': 0.9845, '197Au': 0.0155} 
        irradiation_time : float
            Tiempo (s) de irradiación del alambre.
        dn : int, optional
            Cantidad de neutrones absorbidos. Cambio de A en la reacción. (D = 1).
            
        """
        self.ti = irradiation_time
        self.comp = composition
        self.As = {x: int(re.findall(r'\d+', x)[0]) for x in self.comp}
        self.stable_data = {x: loadfromIAEA(x, 'estable') for x in self.comp}
        self.abundance = {x: self.stable_data[x].data['abundance'][0]/100 for x in self.comp}
        # self.sigmas = {x: self.stable_data[x].get('abundance') for x in self.comp}
        self.act_els = {x: x.replace(str(self.As[x]), str(self.As[x]+dn)) for x in self.comp}
        self.data_decay = {iso: loadfromIAEA(iso, 'decay') for iso in list(self.act_els.values())}
        self.hl = {iso: self.data_decay[iso].get(['half_life_sec']).to_numpy().mean() for iso in list(self.act_els.values())}
    def N_padres(self, masa_total, masa_errrel):
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
    def Act_alambre(self, iso, Epeak, Net_cps, Net_cps_err,
                    gap_time, treal, eff_params, var_mu):
        """
        Calcula la actividad para a partir de los datos definidos previamente, y de las cuentas por ROI.

        Parameters
        ----------
        iso : str
            Núcleo hija luego de absorber el neutrón. Usar notación que se use en la api de IAEA.
        Epeak : float
            Energía del pico en la ROI.
        Net_cps : float
            Cuentas por segundo netas de la ROI.
        Net_cps_err : float
            Error de cuentas netas.
        gap_time : float
            Tiempo (s) desde que se dejó de irradiar hasta que empezó a medirse el alambre.
        treal : float
            Tiempo real final de adqusición.
        eff_params : array
            Coeficientes del polinomio de ajuste durante la calibración en eficiencia.
        var_mu : function
            Función de la varianza de la eficiencia en función de la energía.

        Returns
        -------
        Actp : array (1, 2), float  
            Actividad parcial del nucleído 'iso'.

        """
        Actp = Actividad(Epeak, Net_cps, Net_cps_err, treal, eff_params, var_mu, gap_time, datadecay=self.data_decay[iso])
        return Actp

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
