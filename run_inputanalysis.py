import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import os
import equadratures as eq
from funcs.getFRF_funcs.getFRF_lidar import *

# Load temporally aligned data - need to add lidarelev_fullspan
with open('IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,xc_fullspan,dXcdt_fullspan,lidarelev_fullspan = pickle.load(file)
# Load data availability info
with open('IO_datavail.pickle','rb') as file:
    datavail_wave8m, datavail_wave17m, datavail_tidegauge, datavail_lidar_elev2p, datavail_lidarwg080, datavail_lidarwg090, datavail_lidarwg100, datavail_lidarwg110, datavail_lidarwg140, datavail_Xc, datavail_dXcdt = pickle.load(file)

# Determine when all data of interest align with Xc time series of interest
tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg110
datavail_all = np.ones(shape=time_fullspan.shape)
datavail_all[tmp < 7] = 0   # data considered = 2 real WGs, 1 runup gauge, 1 tidal gauge, and 2 virtual WGs
contourii = 2                      # contour Zc[ii=3] = 1.5m
tmp = datavail_Xc[contourii, :] + datavail_all + datavail_dXcdt[contourii, :]
ij_alldatavail = (tmp == 3) # here, all data refers to (1) hydro, (2) Xc, and (3) dXc/dt

# Remove mean and std from all available data
datall_hydro = np.hstack((data_wave8m[ij_alldatavail],data_wave17m[ij_alldatavail],data_tidegauge[ij_alldatavail],data_lidar_elev2p[ij_alldatavail],data_lidarwg110[ij_alldatavail]))

# Remove mean and standard deviation
dataMean = np.mean(datall_hydro,axis=0)
dataStd = np.std(datall_hydro,axis=0)
dataNorm = (datall_hydro - dataMean) / dataStd

# Make dataframe
hydrovariablenames = ['Hs(8m)','Tp(8m)','dir(8m)','Hs(17m)','Tp(17m)','dir(17m)',
                    'WL(NOAA)','r2p','Hs(x=110m)','HsIN(x=110m)','HsIG(x=110m)','Tp(x=110m)','TmIG(x=110m)']
scaled_df = pd.DataFrame(dataNorm)
scaled_df.columns = hydrovariablenames
fig, ax = plt.subplots()
sns.heatmap(scaled_df.corr(),annot=True)
plt.tight_layout()
imagedir = './figs/data/overlapping_hydro_initial/pca/'
imagename = imagedir + 'variable_crosscorrelation.png'
if os.path.exists(imagedir) == False:
    os.makedirs(imagedir)
# fig.savefig(imagename, format='png', dpi=1200)

# make histograms of data used for dataframe
for ii in np.arange(scaled_df.columns.size):
    fig, ax = plt.subplots()
    var = dataNorm[:,ii]
    binlocs = np.arange(np.nanmin(var),np.nanmax(var),0.1)
    plt.hist(var, bins=binlocs, density=True, stacked=True, rwidth=1)
    plt.title('Normalized '+scaled_df.columns[ii]+
              ', mean = '+str("%0.2f" % dataMean[ii])+
              ', stdv = '+str("%0.2f" % dataStd[ii]))
    plt.xlabel('value')
    plt.ylabel('density')
    imagedir = './figs/data/overlapping_hydro_initial/histograms/'
    imagename = imagedir + scaled_df.columns[ii] + '_pdf.svg'
    if os.path.exists(imagedir) == False:
        os.makedirs(imagedir)
    # fig.savefig(imagename, format='svg', dpi=1200)

# Perform PCA
pca = PCA(n_components=10)
pca.fit(scaled_df)
data_pca = pca.transform(scaled_df)
# data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3','PC4','PC5','PC6',
#                                           'PC7','PC8','PC9','PC10','PC11','PC12',
#                                           'PC13','PC14','PC15','PC16','PC17'])
data_pca = pd.DataFrame(data_pca,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])
# plot correlation of PCs
fig, ax = plt.subplots()
sns.heatmap(data_pca.corr(),annot=True)
plt.tight_layout()
imagedir = './figs/data/overlapping_hydro_initial/pca/'
imagename = imagedir + 'pca_crosscorrelation.png'
if os.path.exists(imagedir) == False:
    os.makedirs(imagedir)
# fig.savefig(imagename, format='png', dpi=1200)


# plot bar graph of variance_ratio explained by PCs
fig, ax = plt.subplots()
plt.bar(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_)
plt.ylabel('Explained variance')
plt.xlabel('PCA components')
plt.plot(range(1,len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red',
         label="Cumulative Explained Variance")
plt.legend(loc='upper left')
ax.grid(which='both', axis='both')
imagedir = './figs/data/overlapping_hydro_initial/pca/'
imagename = imagedir + 'pca_varianceexplained.png'
if os.path.exists(imagedir) == False:
    os.makedirs(imagedir)
# fig.savefig(imagename, format='png', dpi=1200)

fig, ax = plt.subplots()
sns.heatmap(pca.components_,vmin=-0.6,vmax=0.6,
             yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
             cmap='coolwarm',
             xticklabels=list(scaled_df.columns),
             cbar_kws={"orientation": "vertical"})
ax.set_aspect("equal")
plt.tight_layout()
imagedir = './figs/data/overlapping_hydro_initial/pca/'
imagename = imagedir + 'pca_variablecontribution.png'
if os.path.exists(imagedir) == False:
    os.makedirs(imagedir)
# fig.savefig(imagename, format='png', dpi=1200)

# Plot Xc and dXc/dt as a function of PCs
num_PCs = 4
cont_elev = np.arange(0,2.5,0.5)    # <<< MUST BE POSITIVELY INCREASING
fig100, ax100 = plt.subplots(1,  cont_elev.size)
fig333, ax333 = plt.subplots(1, cont_elev.size)
for jj in np.arange(cont_elev.size):
    # Find the actual times of overlap of hydro with contour_elevation[jj]
    tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg110
    datavail_all = np.ones(shape=time_fullspan.shape)
    datavail_all[tmp < 5] = 0  # data considered = 2 real WGs, 1 runup gauge, 1 tidal gauge, and 2 virtual WGs
    tmp = datavail_Xc[jj, :] + datavail_all + datavail_dXcdt[jj, :]
    ij_alldatavail = (tmp == 3)
    datall_hydro = np.hstack((data_wave8m[ij_alldatavail], data_wave17m[ij_alldatavail],
                              data_tidegauge[ij_alldatavail], data_lidar_elev2p[ij_alldatavail],
                              data_lidarwg110[ij_alldatavail]))
    # Remove mean and standard deviation
    dataMean = np.mean(datall_hydro, axis=0)
    dataStd = np.std(datall_hydro, axis=0)
    dataNorm = (datall_hydro - dataMean) / dataStd
    # Perform PCA, again..................
    pca = PCA(n_components=10)
    scaled_df = pd.DataFrame(dataNorm)
    pca.fit(scaled_df)
    tmp_pca = pca.transform(scaled_df)
    df_pca = pd.DataFrame(tmp_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10'])
    exec('fig' + str(jj) + ', ax' + str(jj) + ' = plt.subplots(1,4)')
    # tmp_flag = np.empty(shape=time_fullspan.shape)
    # tmp_flag[:] = np.nan
    tmp_flag = (np.abs(dXcdt_fullspan[jj,:]) > 2) & (ij_alldatavail)
    var = dXcdt_fullspan[jj,:]
    binlocs = np.arange(np.nanmin(var), np.nanmax(var), 0.1)
    ax333[jj].hist(var, bins=binlocs, density=True, stacked=True, rwidth=1)
    fig333.suptitle('dXc/dt')
    ax333[jj].set_title('z = ' + str(cont_elev[jj]) + 'm, N = ' + str(sum(~np.isnan(var))))
    ax333[jj].set_xlabel('value')
    ax333[0].set_ylabel('density')
    ax333[jj].set_xlim(-2.,2.)

    exec('abs_delXc'+str(jj)+'_gt_2 = tmp_flag')
    for ii in np.arange(num_PCs):
        # first, plot Xc vs PC, scatter colored by DelXc
        contourjj = cont_elev[jj]
        xplot = xc_fullspan[jj,ij_alldatavail]
        cplot = dXcdt_fullspan[jj,ij_alldatavail]
        varnameii = 'PC'+str(ii+1)
        yplot = eval('df_pca[varnameii]')
        ph = eval('ax'+str(jj)+'[ii]'+'.scatter(xplot,yplot,5,cplot)')
        ph.set_cmap('coolwarm')
        ph.set_clim(-1, 1)
        # st = fig.suptitle('Xc(z = '+str(contourjj)+'m), N = '+str(631), fontsize="x-large")
        titlstr = 'Xc(z = '+str(contourjj)+'m), N = '+str(sum(ij_alldatavail))
        st = eval('fig' + str(jj)+'.suptitle(titlstr, fontsize="x-large")')
        if ii == num_PCs-1:
            cbar = eval('fig.colorbar(ph, ax=ax' + str(jj) + '[ii]' +')')
            cbar.set_label('DELTA = Xc(t+1) - Xc(t)')
        elif ii == 0:
            ylblstr = 'PC' + str(eval('ii'))
            ax.set_ylabel(ylblstr)
        xlblstr = 'Xc(t)'
        titlestr = 'PC'+str(ii+1)+', ' + '%0.1f' % (100*pca.explained_variance_ratio_[ii],)+'%'
        eval('ax' + str(jj) + '[ii]' + '.set_xlabel(xlblstr)')
        eval('ax'+str(jj)+'[ii]'+'.set_ylim(-6, 6)')
        eval('ax' + str(jj) + '[ii]' + '.set_title(titlestr)')
    # Now, plot contributions to each PC
    if jj == 0:
        sns.heatmap(np.transpose(pca.components_[0:4,:]), vmin=-0.6, vmax=0.6,
                    xticklabels=['%0.1f' % (100 * pca.explained_variance_ratio_[0],) + '%',
                                 '%0.1f' % (100 * pca.explained_variance_ratio_[1],) + '%',
                                 '%0.1f' % (100 * pca.explained_variance_ratio_[2],) + '%',
                                 '%0.1f' % (100 * pca.explained_variance_ratio_[3],) + '%'],
                    cmap='coolwarm',
                    yticklabels=list(hydrovariablenames),
                    cbar=False,annot=True,fmt='.2f',annot_kws={'size': 12},
                    ax=ax100[jj])
    elif jj == 4:
        sns.heatmap(np.transpose(pca.components_[0:4, :]), vmin=-0.6, vmax=0.6,
                xticklabels=['%0.1f' % (100 * pca.explained_variance_ratio_[0],) + '%',
                             '%0.1f' % (100 * pca.explained_variance_ratio_[1],) + '%',
                             '%0.1f' % (100 * pca.explained_variance_ratio_[2],) + '%',
                             '%0.1f' % (100 * pca.explained_variance_ratio_[3],) + '%'],
                cmap='coolwarm',
                yticklabels=[],
                cbar=True, annot=True, fmt='.2f', annot_kws={'size': 12},
                ax=ax100[jj])
    else:
        sns.heatmap(np.transpose(pca.components_[0:4, :]), vmin=-0.6, vmax=0.6,
                    xticklabels=['%0.1f' % (100*pca.explained_variance_ratio_[0],) +'%',
                                 '%0.1f' % (100*pca.explained_variance_ratio_[1],) +'%',
                                 '%0.1f' % (100*pca.explained_variance_ratio_[2],) +'%',
                                 '%0.1f' % (100*pca.explained_variance_ratio_[3],) +'%'],
                    cmap='coolwarm',
                    yticklabels=[],
                    cbar=False,annot=True,fmt='.2f',annot_kws={'size': 12},
                    ax=ax100[jj])
    ax100[jj].set_title('Xc(z = '+str(cont_elev[jj])+'m)')


for jj in np.arange(cont_elev.size):
    # Find the actual times of overlap of hydro with contour_elevation[jj]
    tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg110
    datavail_all = np.ones(shape=time_fullspan.shape)
    datavail_all[tmp < 5] = 0  # data considered = 2 real WGs, 1 runup gauge, 1 tidal gauge, and 2 virtual WGs
    tmp = datavail_Xc[jj, :] + datavail_all + datavail_dXcdt[jj, :]
    ij_alldatavail = (tmp == 3)
    datall_hydro = np.hstack((data_wave8m[ij_alldatavail], data_wave17m[ij_alldatavail],
                            data_tidegauge[ij_alldatavail], data_lidar_elev2p[ij_alldatavail],
                            data_lidarwg110[ij_alldatavail]))
    fig, ax = plt.subplots()
    var = dXcdt_fullspan[jj,:]
    binlocs = np.arange(np.nanmin(var), np.nanmax(var), 0.1)
    plt.hist(var, bins=binlocs, density=True, stacked=True, rwidth=1)
    plt.title('dXc/dt(z = '+str(cont_elev[jj])+'m), N = ' + str(sum(~np.isnan(var))))
    plt.xlabel('value')
    plt.ylabel('density')
    ax.set_xlim(-2.5,2.5)
    print('min = '+str(np.nanmin(var))+', max='+str(np.nanmax(var)))


# plot time series of Xc, highlight where dXc/dt >= 2
full_path = 'C:/Users/rdchlerh/Desktop/FRF_data/dune_lidar/lidar_transect/FRF-geomorphology_elevationTransects_duneLidarTransect_201510.nc'
qaqc_fac, lidar_pmissing, lidar_elev, lidar_elevstd, lidar_time, lidar_xFRF, lidar_yFRF = (
            getlocal_lidar(full_path))
jj = 0
fig, ax = plt.subplots()
tplot = pd.to_datetime(time_fullspan, unit='s', origin='unix')
yplot = xc_fullspan[jj,:]
plt.plot(tplot,yplot,'k')
plt.scatter(tplot,yplot,s=3,c='k')
yplot = eval('xc_fullspan[jj,abs_delXc'+str(jj)+'_gt_2]')
xplot = eval('tplot[abs_delXc'+str(jj)+'_gt_2]')
plt.scatter(xplot,yplot,s=20,facecolors='none',edgecolors='r')
ax.grid(which='both',axis='both')
for tt in np.nditer(np.where(eval('abs_delXc'+str(jj)+'_gt_2'))):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(13, 8))
    iplot = np.arange(tt-3,tt+4)
    tmp = tplot[iplot]-tplot[iplot[0]]
    tscale = (tmp.seconds/3600) / (tmp[-1].seconds/3600)
    cmap = plt.cm.rainbow_r(tscale)
    ax1.plot([min(lidar_xFRF), max(lidar_xFRF)],[cont_elev[jj], cont_elev[jj]],'k')
    ax1.set_prop_cycle('color', cmap)
    xplot = lidar_xFRF
    yplot = lidarelev_fullspan[:,iplot]
    ph = []
    for ii in np.arange(tscale.size):
        tmp = ax1.plot(xplot,yplot[:,ii],c=cmap[ii,:])
        ph.append(tmp)
    # ph = ax.plot(xplot,yplot)
    fig.suptitle(str(tplot[int(tt)].round(freq='min')))
    ax1.grid(which='both',axis='both')
    ax1.set_xlim(xc_fullspan[jj,tt]-10,xc_fullspan[jj,tt]+10)
    ax1.set_ylim(cont_elev[jj]-0.5,cont_elev[jj]+0.5)
    ax1.set_xlabel('xFRF [m]')
    ax1.set_ylabel('z [m]')
    # ax2.plot(tplot[iplot],xc_fullspan[jj,iplot],'o')
    for ii in np.arange(tscale.size):
        # ax2.plot(tplot[iplot[ii]],xc_fullspan[jj,iplot[ii]],'o',color=cmap[ii,:])
        ax2.plot(tplot[iplot[ii]], dz_at_xc[jj, iplot[ii]], 'o', color=cmap[ii, :])
    ax2.grid(which='both',axis='both')
    # ax2.set_ylabel('xc [m]')
    ax2.set_xlabel('time')
    ax2p2 = ax2.twinx()
    ax2p2.plot(tplot[iplot],dXcdt_fullspan[jj,iplot],'x',color=(0.5,0.5,0.5))
    ax2p2.set_ylabel('dx/dt [m/hr]',color=(0.5,0.5,0.5))
    ax2.set_xlim(tplot[iplot[0]-1],tplot[iplot[-1]+1])
    ax2p2.set_xlim(tplot[iplot[0]-1],tplot[iplot[-1]+1])
    imagedir = './figs/data/dxdt_greaterthan2m/xc_eq_' + str(cont_elev[jj]) + 'm/'
    tmpstr = str(tplot[int(tt)].round(freq='min'))
    imagename = imagedir + 'fig_wDZ_' + tmpstr.replace(':','') + '.png'
    if os.path.exists(imagedir) == False:
        os.makedirs(imagedir)
    fig.savefig(imagename, format='png', dpi=600)

dz_at_xc = np.empty(shape=xc_fullspan.shape)
dz_at_xc[:] = np.nan
for jj in np.arange(cont_elev.size):
    for tt in np.arange(time_fullspan.size-1):
        z_ti = lidarelev_fullspan[:,tt]
        z_tip1 = lidarelev_fullspan[:,tt+1]
        xctmp = xc_fullspan[jj,tt]
        if ~np.isnan(xctmp):
            iiclose = int(np.argwhere(abs(lidar_xFRF - xctmp) == min(abs(lidar_xFRF - xctmp))))
            z_xc_ti = np.interp(0, lidar_xFRF[iiclose - 1:iiclose + 2], z_ti[iiclose - 1:iiclose + 2])
            z_xc_tip1 = np.interp(0, lidar_xFRF[iiclose - 1:iiclose + 2], z_tip1[iiclose - 1:iiclose + 2])
            dz_at_xc[jj,tt] = z_xc_tip1 - z_xc_ti
fig, ax = plt.subplots()
xplot = np.reshape(dXcdt_fullspan,-1,)
yplot = np.reshape(dz_at_xc,-1,)
plt.scatter(xplot,yplot,10,alpha=0.1)
ax.grid(which='both', axis='both')
ax.set_xlabel('dXc/Dt')
ax.set_ylabel('Z(Xc_t+1) - Z(Xc_t)')
fig, ax = plt.subplots()
ikeep = ~np.isnan(xplot) & ~np.isnan(yplot) & (abs(xplot) <= 2)
ax.hist2d(xplot[ikeep], yplot[ikeep], bins=(500,1000), cmap=plt.cm.jet)
ax.set_xlim(-1,1)

























# Develop kth-order PCE of Xc = f(input_i=1:d), where d = 13 and k = 2

# Normalize IO data
jj=3
tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg110
datavail_all = np.ones(shape=time_fullspan.shape)
datavail_all[tmp < 5] = 0  # data considered = 2 real WGs, 1 runup gauge, 1 tidal gauge,1 virtual WGs
tmp = datavail_Xc[jj, :] + datavail_all + datavail_dXcdt[jj, :]
ij_alldatavail = (tmp == 3)
datall_hydro = np.hstack((data_wave8m[ij_alldatavail,:], data_wave17m[ij_alldatavail,:],
                          data_tidegauge[ij_alldatavail,:], data_lidar_elev2p[ij_alldatavail,:],
                          data_lidarwg110[ij_alldatavail,:]))
# datall_topo = np.hstack((dXcdt_fullspan[ij_alldatavail],
# Remove mean and standard deviation
dataMean = np.mean(datall_hydro, axis=0)
dataStd = np.std(datall_hydro, axis=0)
dataNorm = (datall_hydro - dataMean) / dataStd
hydroNorm = dataNorm
delta_xc_overlap = dXcdt_fullspan[jj,ij_alldatavail]
delta_xcMean = np.mean(delta_xc_overlap)
delta_xcStd = np.std(delta_xc_overlap)
delta_xcNorm = (delta_xc_overlap - delta_xcMean) / delta_xcStd
xc_overlap = xc_fullspan[jj,ij_alldatavail]
xcMean = np.mean(xc_overlap)
xcStd = np.std(xc_overlap)
xcNorm = (xc_overlap - xcMean) / xcStd
inputNorm = np.empty(shape=(xcNorm.size,hydroNorm.shape[1]+1))
inputNorm[:,0:-1] = hydroNorm[:,:]
inputNorm[:,-1] = xcNorm[:]

# fig, ax = plt.subplots(14)
# variablenames = ['Hs(8m)','Tp(8m)','dir(8m)','Hs(17m)','Tp(17m)','dir(17m)',
#                  'WL(NOAA)','r2p','Hs(x=110m)','HsIN(x=110m)','HsIG(x=110m)',
#                  'Tp(x=110m)','TmIG(x=110m)','Xc']
# for pp in np.arange(14):
#     ax[pp].plot(inputNorm[:,pp],label=variablenames[pp])
# plt.legend()

# make histograms describing param distributions (equad input)
order_params = 2            # order (k)
for ii in np.arange(inputNorm.shape[1]):
    dataii = inputNorm[:,ii]
    input_dist = eq.Weight(dataii, support=[np.min(dataii), np.max(dataii)], pdf=False)
    input_param = eq.Parameter(distribution='data', weight_function=input_dist, lower=np.min(dataii), upper=np.max(dataii), order=order_params)
    exec('input_'+str(ii)+' = input_param')
params = [input_0,input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,input_9,
          input_10,input_11,input_12,input_13]
mybasis = eq.Basis('total-order')



Ntrain = 2000
Xtrain = inputNorm[0:Ntrain,:]
Ytrain = delta_xcNorm[0:Ntrain]
Xtest = inputNorm[Ntrain+1:,:]
Ytest = delta_xcNorm[Ntrain+1:]
mypoly  = eq.Poly(params, mybasis, method='least-squares',sampling_args={'sample-points':Xtrain,'sample-outputs':Ytrain})
mypoly.basis.cardinality
mypoly.set_model()
mypoly.plot_sobol()
# mypoly.plot_sobol(order=2)
ytmp = mypoly.get_polyfit(Xtest)
ypred = np.array(ytmp).reshape(ytmp.size,)
ytrue = Ytest
err = ypred[:]-ytrue[:]
ii_notoutlier = (abs(err) <= 3*np.std(err))

ypred_rescale = ypred[ii_notoutlier] * delta_xcStd + delta_xcMean
ytrue_rescale = ytrue[ii_notoutlier] * delta_xcStd + delta_xcMean
rmse = eq.datasets.score(ypred_rescale, ytrue_rescale, metric='rmse')
nrmse = rmse/np.mean(ytrue_rescale)
fig, ax = plt.subplots()
plt.plot(ytrue,err,'o')
ax.set_ylim(-1,1)
ax.set_xlim(-1,1)




