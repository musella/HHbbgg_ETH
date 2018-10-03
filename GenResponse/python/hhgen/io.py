import root_pandas as rpd
import pandas as pd
import numpy as np

## --------------------------------------------------------------------
def read_trees(fname,untagged,tagged,ncats,seed=12345,**kwargs):

    dfs = [ rpd.read_root(fname,untagged,**kwargs) ]
    for icat in range(ncats):
        tname = tagged % icat
        try: 
            dfs.append( rpd.read_root(fname,tname,**kwargs) )
        except:
            dfs.append( pd.DataFrame() )
    
    for icat,idf in enumerate(dfs):
        idf[ "cat" ] = icat
    df = pd.concat( dfs )

    random_index = np.arange( df.shape[0] )
    np.random.shuffle(random_index)
    
    df["random_index"] = random_index
    df.set_index("random_index",inplace=True)
    df.sort_index(inplace=True)
        
    return df

