"""Reproduce manuscript-style Figure 5 from the Seed-0 checkpoint.

Run from project root with the paper's Python environment:
    python figures/make_fig5_device_schedule.py

Optional offline audit of the plotting code using already verified values:
    python figures/make_fig5_device_schedule.py --csv path/to/fig5_plotted_values.csv

Outputs go to fig5_reproduction/ and never alter the checkpoint/results.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]

def plot(df,out):
    panels=[(661,'AC'),(661,'EV'),(3039,'AC'),(3039,'DW'),(3039,'Dryer'),(8565,'AC'),(8565,'Dryer')]
    assert set(zip(df.house_id,df.appliance))==set(panels)
    fig=plt.figure(figsize=(10.5,10),dpi=200)
    gs=GridSpec(9,1,height_ratios=[1.25,1.25,.42,1,1,1,.42,1,1],hspace=.27,top=.89,bottom=.09,left=.12,right=.985)
    # Spacer rows separate EU groups and prevent labels from touching prior y-axis ticks.
    axes=[fig.add_subplot(gs[i,0]) for i in (0,1,3,4,5,7,8)]
    heights={(661,'AC'):(3.4,[0,1.5,3]),(661,'EV'):(4.8,[0,2,4]),
             (3039,'AC'):(4.9,[0,2,4]),(3039,'DW'):(1.9,[0,.8,1.6]),
             (3039,'Dryer'):(2.2,[0,.8,1.6]),(8565,'AC'):(3.2,[0,1,2]),
             (8565,'Dryer'):(2.5,[0,1,2])}
    h=np.arange(1,25); w=.38; offset=.18
    for ax,(house,app) in zip(axes,panels):
        d=df[(df.house_id==house)&(df.appliance==app)].sort_values('hour')
        assert list(d.hour)==list(range(1,25))
        b=d.baseline_kWh.to_numpy(); a=d.after_DR_kWh.to_numpy()
        c=d.curtailed_kWh.to_numpy(); o=d.shifted_out_kWh.to_numpy(); s=d.shifted_in_kWh.to_numpy()
        assert np.allclose(b-c-o+s,a,atol=1e-7),(house,app)
        ax.bar(h-offset,b,width=w,color='#ededed',edgecolor='#777777',linewidth=1.15,zorder=1)
        ax.bar(h+offset,a,width=w,color='#1e88e5',edgecolor='black',linewidth=.9,zorder=2)
        m=c>1e-10
        ax.bar(h[m]-offset,c[m],bottom=(b-c)[m],width=w,facecolor='none',edgecolor='red',hatch='///',linewidth=.9,zorder=3)
        m=o>1e-10
        ax.bar(h[m]-offset,o[m],bottom=(b-c-o)[m],width=w,facecolor='none',edgecolor='#ff7f0e',hatch='xxx',linewidth=.9,zorder=3)
        m=s>1e-10
        ax.bar(h[m]+offset,s[m],bottom=(a-s)[m],width=w,facecolor='none',edgecolor='#2ca02c',hatch='\\\\',linewidth=.9,zorder=3)
        upper,ticks=heights[(house,app)]
        ax.set_ylim(0,upper);ax.set_yticks(ticks);ax.set_xlim(.5,24.5)
        ax.set_xticks(np.arange(1,24,2))
        if ax is not axes[-1]:ax.set_xticklabels([])
        else:ax.set_xticklabels([str(i) for i in range(1,24,2)],fontweight='bold',fontsize=12)
        for label in ax.get_yticklabels():label.set_fontweight('bold')
        for spine in ax.spines.values():spine.set_linewidth(1.4)
        ax.tick_params(width=1.1,length=5,labelsize=10)
        right=(house,app) in [(3039,'AC'),(8565,'AC'),(3039,'Dryer')]
        ax.text(.82 if right else .015,.83,app,transform=ax.transAxes,
                fontsize=16,fontweight='bold',ha='right' if right else 'left',va='center')
    for idx,k in [(0,1),(2,2),(5,3)]:
        bounds=axes[idx].get_position()
        fig.text(.045,bounds.y1+.013,f'({"abc"[k-1]}) EU{k}',fontsize=18,fontweight='bold',va='bottom')
    fig.text(.02,.50,'Energy (kWh)',rotation=90,fontsize=21,fontweight='bold',ha='center',va='center')
    fig.text(.54,.024,'Hour',fontsize=17,fontweight='bold',ha='center')
    legend=[Patch(facecolor='#ededed',edgecolor='#777777',label='Baseline'),
            Patch(facecolor='#1e88e5',edgecolor='black',label='After DR'),
            Patch(facecolor='none',edgecolor='red',hatch='///',label='Curtailed'),
            Patch(facecolor='none',edgecolor='#ff7f0e',hatch='xxx',label='Shifted out'),
            Patch(facecolor='none',edgecolor='#2ca02c',hatch='\\\\',label='Shifted in')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.53,.982),ncol=5,
               frameon=False,fontsize=15,columnspacing=1)
    fig.savefig(out/'device_schedule_day_208_revised.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'device_schedule_day_208_revised.pdf',bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',type=Path,help='Use the independently audited CSV without a checkpoint replay')
    ap.add_argument('--output-dir',type=Path,default=ROOT/'fig5_reproduction')
    arg=ap.parse_args();arg.output_dir.mkdir(parents=True,exist_ok=True)
    if arg.csv:
        data=pd.read_csv(arg.csv)
    else:
        # Audited replay script produces a 24-hour CSV from the frozen checkpoint.
        subprocess.run([sys.executable,str(Path(__file__).with_name('fig5_checkpoint_replay.py')),
                        str(ROOT),str(arg.output_dir)],check=True,cwd=ROOT)
        data=pd.read_csv(arg.output_dir/'fig5_plotted_values.csv')
    data.to_csv(arg.output_dir/'fig5_plotted_values.csv',index=False)
    plot(data,arg.output_dir)
    print('Saved reproducible Fig. 5 to',arg.output_dir)

if __name__=='__main__':main()
