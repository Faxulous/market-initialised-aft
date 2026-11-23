import pandas as pd, numpy as np, glob

W, H, F = 16, 12, glob.glob("data/**/whoscored_events.parquet", recursive=True)
acc = {k: np.zeros(W*H) for k in 'psg'} | {'t': np.zeros((W*H, W*H)), 'st': np.zeros(W*H)}

print("Training Global xT Model...")
for f in F:
    try:
        df = pd.read_parquet(f, columns=['event_type','event_outcome','start_x','start_y','end_x','end_y'])
        xy = lambda c,m: (df[c]*m/100).fillna(-1).astype(int).clip(-1, m-1)
        x, y, ex, ey = xy('start_x',W), xy('start_y',H), xy('end_x',W), xy('end_y',H)
        i, ie, v = y*W+x, ey*W+ex, (x>=0)&(y>=0)
        
        m_p, m_s, m_g = (df.event_type=='Pass')&v, df.event_type.isin(['SavedShot','MissedShots','Goal','ShotOnPost'])&v, (df.event_type=='Goal')&v
        succ = m_p & (df.event_outcome=='Successful') & (ex>=0) & (ey>=0)
        
        for k, m in zip('psg', [m_p, m_s, m_g]): acc[k] += np.bincount(i[m], minlength=W*H)
        if succ.any():
            acc['t'] += np.bincount(i[succ]*W*H + ie[succ], minlength=(W*H)**2).reshape(W*H,W*H)
            acc['st'] += np.bincount(i[succ], minlength=W*H)
    except: pass

with np.errstate(invalid='ignore', divide='ignore'):
    mp = np.nan_to_num(acc['p']/(acc['p']+acc['s'])) * np.nan_to_num(acc['st']/np.maximum(acc['p'],1))
    xT = np.linalg.solve(np.eye(W*H) - np.diag(mp) @ np.nan_to_num(acc['t']/np.maximum(acc['t'].sum(1)[:,None],1)), 
                         np.nan_to_num(acc['s']/(acc['p']+acc['s'])) * np.nan_to_num(acc['g']/acc['s'])).reshape(H,W)

print("Applying xT...")
for f in F:
    try:
        df = pd.read_parquet(f)
        for p, (xc, yc) in {'start':('start_x','start_y'), 'end':('end_x','end_y')}.items():
            x, y = (df[xc]*W/100).fillna(-1).astype(int).clip(-1,W-1), (df[yc]*H/100).fillna(-1).astype(int).clip(-1,H-1)
            df[f'xT_{p}'] = np.where((x>=0)&(y>=0), xT[y, x], 0.0)
        df['xT_added'] = df['xT_end'] - df['xT_start']
        df.to_parquet(f)
    except: pass
