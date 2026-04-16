from pydantic import BaseModel,field_validator
import numpy as np

class Results(BaseModel):
    data : list
    usl:float
    lsl:float
    title:str
    ad:float|None
    p_value:float|None
    params:tuple
    cp:float|None
    cpk:float|None
    pp:float|None
    ppk:float|None
    pdf_values:list

    @field_validator("pdf_values",mode="after")
    def convert_to_array(cls,v):
        return np.array(v,dtype=float)