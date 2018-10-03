import numpy as np

## -----------------------------------------------------------------------------------------
def calc_p4extra(df,prefix):
    px = df[prefix+"_px"]
    py = df[prefix+"_py"]
    pz = df[prefix+"_pz"]
    en = df[prefix+"_e"]

    df[prefix+"_pt"] = np.sqrt( px**2 + py**2 )
    df[prefix+"_eta"] = np.arcsinh( pz / df[prefix+"_pt"] )
    df[prefix+"_phi"] = np.arcsin( py / df[prefix+"_pt"] )
    df[prefix+"_m"] = np.sqrt( en**2 - px**2 -py**2 -pz**2 )

## -----------------------------------------------------------------------------------------
def calc_sump4(df,dest,part1,part2):
    for comp in "_px","_py","_pz","_e":
        df[dest+comp] = df[part1+comp] + df[part2+comp]
    calc_p4extra(df,dest)

## -----------------------------------------------------------------------------------------
def cal_cos_theta_hlx(df):
    pass

## -----------------------------------------------------------------------------------------
def calc_cos_theta_cs(diPho,diJet):
    pass

## -----------------------------------------------------------------------------------------
def cos_theta_hlx(leadPho,subleadPho,leadJ,subleadJ):
    pass


## -----------------------------------------------------------------------------------------
def cos_theta_cs(diPho,diJet):
    pass
