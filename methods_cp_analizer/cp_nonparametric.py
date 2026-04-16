import numpy as np
from models.result import Results

def non_parametric_anaysis(self):
    title="NO PARAMETRICO - METODO POR PERCENTILES"
    data_flat=self.data_subgroups.flatten()
    xpu=np.percentile(data_flat,99.865)
    xpl=np.percentile(data_flat,0.135)
    cnp=(self.usl-self.lsl)/(xpu-xpl)
    cnpk=min(((np.median(data_flat)-self.lsl)/(np.median(data_flat)-xpl)),((self.usl-np.median(data_flat))/(xpu-np.median(data_flat))))
    return Results(data=self.data_subgroups,
            usl=self.usl,
            lsl=self.lsl,
            title=title,
            ad=None,
            p_value=None,
            params=(),
            cp=cnp,
            cpk=cnpk,
            pp=None,
            ppk=None,
            pdf_values=[])