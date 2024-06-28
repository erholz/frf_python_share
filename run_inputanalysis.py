import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # to load the dataframe
from sklearn.decomposition import PCA  # to apply PCA
import seaborn as sns
import os
import equadratures as eq

# Load temporally aligned data
with open('IO_alignedintime.pickle', 'rb') as file:
    time_fullspan,data_wave8m,data_wave17m,data_tidegauge,data_lidar_elev2p,data_lidarwg080,data_lidarwg090,data_lidarwg100,data_lidarwg110,data_lidarwg140,xc_fullspan,dXcdt_fullspan = pickle.load(file)
# Load data availability info
with open('IO_datavail.pickle','rb') as file:
    datavail_wave8m, datavail_wave17m, datavail_tidegauge, datavail_lidar_elev2p, datavail_lidarwg080, datavail_lidarwg090, datavail_lidarwg100, datavail_lidarwg110, datavail_lidarwg140, datavail_Xc, datavail_dXcdt = pickle.load(file)

# Determine when all data of interest align with Xc time series of interest
tmp = datavail_wave8m + datavail_wave17m + datavail_tidegauge + datavail_lidar_elev2p + datavail_lidarwg110
datavail_all = np.ones(shape=time_fullspan.shape)
datavail_all[tmp < 5] = 0   # data considered = 2 real WGs, 1 runup gauge, 1 tidal gauge, and 2 virtual WGs
contourii = 3                      # contour Zc[ii=3] = 1.5m
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
fig100, ax100 = plt.subplots(1, 5)
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
    exec('abs_delXc'+str(jj)+'_gt_2 = np.where(np.abs(dXcdt_fullspan[jj,ij_alldatavail]) > 2)')
    for ii in np.arange(num_PCs):
        # first, plot Xc vs PC, scatter colored by DelXc
        contourjj = cont_elev[jj]
        xplot = xc_fullspan[jj,ij_alldatavail]
        cplot = dXcdt_fullspan[jj,ij_alldatavail]
        varnameii = 'PC'+str(ii+1)
        yplot = eval('df_pca[varnameii]')
        ph = eval('ax'+str(jj)+'[ii]'+'.scatter(xplot,yplot,5,cplot,cmap=cm)')
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
    # plt.tight_layout()
    # cbar_kws = {"orientation": "vertical"},
    # xticklabels = ["PC" + str(x) for x in range(1, 4 + 1)],























# # Develop kth-order PCE of Xc = f(input_i=1:d), where d = 13 and k = 2
# order_params = 2            # order (k)
# for ii in np.arange(scaled_df.columns.size):
#     dataii = dataNorm[:,ii]
#     input_dist = eq.Weight(dataii, support=[np.min(dataii), np.max(dataii)], pdf=False)
#     input_param = eq.Parameter(distribution='data', weight_function=input_dist, lower=np.min(dataii), upper=np.max(dataii), order=order_params)
#     exec('input_'+str(ii)+' = input_param')
# params = [input_0,input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,input_9,
#           input_10,input_11,input_12]
# mybasis = eq.Basis('total-order')
# mypoly  = eq.Poly(params, mybasis, method='least-squares')
# mypoly.basis.cardinality
# mypoly.plot_sobol()