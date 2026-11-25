import pandas as pd, glob, os
D=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audits', 'free_sources_samples', '')  # sample files are not distributed
prim=['lse-group','euronext','xetra','six-swiss-exchange','bolsa-de-madrid','nasdaq-omx','wiener-borse']
def stats(df):
    tot=df.notional.sum(); s=df.groupby('subcategory').notional.sum()
    cl=df[df.subcategory=='auctions-closing']
    return dict(tot=tot/1e9, close=s.get('auctions-closing',0)/1e9,
        lit=(s.get('displayed-',0)+s.get('displayed-retail',0))/1e9,
        cboe3c=cl[cl.venue=='cboe-europe'].notional.sum()/1e9, aquis=cl[cl.venue=='aquis'].notional.sum()/1e9,
        primshare=df[df.venue.isin(prim)].notional.sum()/tot)
res={}
for f in sorted(glob.glob(D+'day_*.csv'))+[D+'d529.csv',D+'dmay.csv']:
    df=pd.read_csv(f); d=df.daterange.iloc[0]; d='May-2026 avg' if d=='all' else d
    res[d]=stats(df)
for d,x in res.items():
    print(f"{d:14s} tot {x['tot']:7.2f} close {x['close']:6.2f} {100*x['close']/x['tot']:5.1f}% lit {100*x['lit']/x['tot']:5.1f}% 3C {100*x['cboe3c']/x['close']:4.1f}% AMC {100*x['aquis']/x['close']:4.1f}% prim {100*x['primshare']:5.1f}%")
m=res['May-2026 avg']; e=res['2026-05-29']
n=21
exc=(m['close']*n-e['close'])/(m['tot']*n-e['tot'])
print('May ex 29 May close share', round(100*exc,1), 'avg total ex', round((m['tot']*n-e['tot'])/(n-1),2), 'avg close ex', round((m['close']*n-e['close'])/(n-1),2))
a,b,c=res['2026-09-17'],res['2026-09-18'],res['2026-09-21']
for k in ['tot','close','lit']:
    print(k, 'x vs 17', round(b[k]/a[k],2), 'x vs 21', round(b[k]/c[k],2), 'x vs avg', round(b[k]/((a[k]+c[k])/2),2))
p,q=res['2025-11-24'],res['2025-11-25']
for k in ['tot','close','lit']: print('nov25', k, round(p[k]/q[k],2))
print('29May vs May ex', round(e['tot']/((m['tot']*n-e['tot'])/(n-1)),2), round(e['close']/((m['close']*n-e['close'])/(n-1)),2))
