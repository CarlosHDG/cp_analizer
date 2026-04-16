from scipy.stats import boxcox_normmax
import numpy as np
from data_analizer import normal_analysis
from models.result import Results

def boxcox_transformation_analysis(self):
    title="BOX-COX TRANSFORMATION"
    data_flat=self.data_subgroups.flatten()
    params_box_cox=boxcox_normmax(data_flat,method="mle")
    params_boxcox_adjusted=max(-5,min(5,round(float(params_box_cox))))
    data_boxcox = (data_flat**params_boxcox_adjusted) if params_boxcox_adjusted != 0 else np.log(data_flat)
    usl_boxcox = (self.usl**params_boxcox_adjusted) if params_boxcox_adjusted != 0 else np.log(self.usl)
    lsl_boxcox = (self.lsl**params_boxcox_adjusted)if params_boxcox_adjusted != 0 else np.log(self.lsl)
    sub_group_size=self.data_subgroups.shape[1]
    data_boxcox_transformed=data_boxcox.reshape(-1,sub_group_size)
    normal=normal_analysis(self,data_boxcox_transformed,usl_boxcox,lsl_boxcox)
    return Results(data=data_boxcox_transformed,
                    usl=usl_boxcox,
                    lsl=lsl_boxcox,
                    title=title,
                    ad=normal.ad,
                    p_value=normal.p_value,
                    params=(params_boxcox_adjusted,),
                    cp=None,
                    cpk=None,
                    pp=normal.pp,
                    ppk=normal.ppk,
                    pdf_values=normal.pdf_values)