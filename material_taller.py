
###IMPORTAMOS LAS LIBRERÍAS NECESARIAS

from neuroiatools.EEGManager.RawArray import makeRawData
from neuroiatools.DisplayData.plotEEG import plotEEG
from neuroiatools.SignalProcessor.ICA import getICA
import h5py
import numpy as np
import pandas as pd
import mne

## ************************ 1. PRIMEROS PASOS ************************
"""
Generamos variables que nos permitiran cargar los datos de EEG y los eventos generados por la app.
Los datos de EEG están en un archivo .hdf5 y los eventos generados por la app están en un archivo .txt.
"""
sfreq = 512 # Frecuencia de muestreo
##cargamos los nombres de los electrodos del g.HIAMP
montage_df = pd.read_csv("ghiamp_montage.sfp",sep="\t",header=None)
ch_names = list(montage_df[0])
channels_to_remove = ["A1","A2"]
ch_names = [ch for ch in ch_names if ch not in channels_to_remove]

##Datos del sujeto y la sesión
n_sujeto = 9
run = 2 ##NÚMERO DE RUN 1 o 2
sesion = 1 #1 ejecutado, 2 imaginado
rootpath = "datasets\\"
sujeto = f"sujeto_{n_sujeto}\\"
tarea = "ejec" if sesion == 1 else "imag" ##tarea ejecutada o imaginada
eeg_file = f"sujeto{n_sujeto}_{tarea}_{run}.hdf5"
event_file = f"eventos_{tarea}_{run}.txt"

##cargamos archivo y lo dejamos de la forma canales x muestras
data = h5py.File(rootpath+sujeto+eeg_file, "r")
raweeg = data["RawData"]["Samples"][:,:62].swapaxes(1,0) #descartamos canales A1 y A2
print(raweeg.shape)

##cargo los eventos marcados por el g.HIAMP
events_time_ghiamp = np.astype(data["AsynchronData"]["Time"][:][1:].reshape(-1), int)/sfreq
##cargo los eventos generados por la app nuestra
eventos_app = pd.read_csv(rootpath+sujeto+event_file)
clases = eventos_app["className"].values

###Creación de un Montage para el posicionamiento de los electrodos
montage = mne.channels.read_custom_montage("ghiamp_montage.sfp")
##ploteamos el montage
# montage.plot(show_names=True)

noisy_eeg_data = makeRawData(raweeg, sfreq, channel_names=ch_names, montage=montage,
                       event_times=events_time_ghiamp, event_labels=clases)

##corto la señal en events_time_ghiamp[0] -3 segundos
noisy_eeg_data.crop(events_time_ghiamp[0]-3)

## ************************ 2. PRIMERA INSPECCIÓN PARA EVALUAR QUITAR CANALES ************************
"""
En esta sección se inspecciona la señal para ver si hay canales ruidosos o con mala impedancia. Simplemente
se plotea la señal completa para ver qué canales se pueden eliminar.
"""
##ploteamos la señal completa para ver si hay canales ruidosos debido a una mala impedancia
plotEEG(noisy_eeg_data.drop_channels(["FT8","T8","F10"]), scalings = 200,show=True, block=True, n_channels=1,
        duration = 5, start = 150, remove_dc = False, bad_color = "red", color = {"eeg":"#000000"},bgcolor = "#FAFAFA",
        highpass=5, lowpass=None, title="Señal de EEG de ejemplo",butterfly=False,picks=["C3"])