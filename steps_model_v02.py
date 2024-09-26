# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:41:45 2024

@author: grover.laporte
"""


import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
import streamlit as st
from scipy.fftpack import fft, ifft, fftfreq
#import warnings



class AccelerationData(object):
    def __init__(self,data,freq=100):
        """
        Acceleration Data - input acceleration data as 4 columns of data -  
        """
        #self.raw_data = data.copy()
        data=data.reset_index()

        self.columns = list(data.columns) 
        self.freq = freq #(in hertz - events per second)
        self.delta_t = 1/freq #(in seconds)
        self.time = np.arange(0,len(data)*self.delta_t,self.delta_t) #in seconds
        self.df = pd.DataFrame(columns = ['time','x','y','z'])
        try:
            self.df['time'] = data[['time']].copy()
            self.df[['x','y','z']] = data[["x","y","z"]].copy()
        except:
            self.df['time']=data.iloc[:,0].copy()
            self.df[['x','y','z']] = data.iloc[:,1:3].copy()
            self.time = np.arange(0,len(self.df)*self.delta_t,self.delta_t) 
        self.df['t']=self.time
        #example_duration - the time (in seconds) used to calculate statistics
        #smaller is better, but increases the amount of storage and time needed to calculate
        self.example_duration = 10
        #ed_groups - example duration groups...needed to determine the number of groups of data
        self.ed_groups = int(self.time[-1]/self.example_duration)
        #resultant vector of the x,y,z components
        self.df['resultant'] = np.sqrt(self.df['x']**2+self.df['y']**2+self.df['z']**2)
        self.N = int(3*freq/4)
        self.ma(self.N)
        self.alpha = 0.1
        self.exp_smooth(self.alpha)
        self.create_seconds_index()
        self.calculate_group_stats()
        plt.style.use('ggplot')
        
    def create_seconds_index(self):
        """
        create_seconds_index - ran during the constructor method. This method creates 
            a dictionary of seconds 
            
            seconds - dictionary of seconds containing the index of the data elements in each second.
            
            Example: 100 Hz sample => seconds[5] = [500,501,502,...599]; while seconds[12]=[1200,1201,...1299]
                                        for the fifth second, these are the indices of the data for that second.
        """
        self.seconds={}
        freq = self.freq
        for i in range(int(len(self.df)/freq)+2):
            self.seconds[i]=list(range(i*freq,(i+1)*(freq)))
                
    def seconds_data(self,start,stop=-1):
        if stop == -1:
            try:
                return self.df.iloc[self.seconds[start]]
            except:
                return "End of data"
        else:
            try:
                idx = []
                for i in range(start,stop+1):
                    idx += self.seconds[i]
                return self.df.iloc[idx]
            except:
                idx = np.arange(self.seconds[start][0],len(self.df))
                return self.df.iloc[idx]
    
    def fft_denoise(self,df,graph=0,axis="resultant"):
        sampling_rate = self.freq #sample rate from data
        #warnings.filterwarnings('ignore')
        f = df[axis].values
        t = df['t'].values
        n = len(f)
        N = int(n/2)
        # fhat - magnitude and phase that you would have to add to get the data set
        fhat = fft(f,n)
        # Power Spectral Density vector - the vector of power for each frequency
        PSD = np.abs(fhat*np.conj(fhat)/n)
        #d is the inverse of the sampling rate 
        # sample spacing is 0.001 here, which means the sample rate is 1/.001 = 1000 Hz
        freq = fftfreq(n,d=1/sampling_rate) 

        ###### Graph the data and the transformation ######
        if graph:
            plt.rcParams['figure.figsize'] = [16,12]
            plt.rcParams.update({'font.size':18})
            plt.style.use('ggplot')
            fig,ax = plt.subplots(3,1)
            fig.subplots_adjust(hspace=0.4)
            ax[0].plot(t,f,color='c',lw=1.5,label="data",alpha=0.4)
            ax[0].set_title("Original Data")
            ax[0].set_xlim(t[0],t[-1])
            ax[0].set_xlabel("time")
            ax[0].legend()


            # Only graph the left side of the PSD vector - it is symmetric.
            ax[1].stem(freq[1:N],abs(PSD[1:N]),"b",markerfmt=' ',basefmt='-b',label='noisy')
            ax[1].set_xlim(freq[0],freq[N-1])
            ax[1].set_xlabel("Frequency (Hertz)")
            ax[1].set_ylabel("Power")
            ax[1].set_title("Frequency Domain")
            ax[1].legend()
            
        ###### Remove noise ######
        # Find the largest power and remove all noise that is not within a certain range of the largest.
        PSD_ = PSD[:N].copy()
        PSD_[0]=0
        # Find the index of the largest power
        indices = np.argsort(PSD_)[::-1]
        
        
        #### Filtered Data ####################
        if graph:
            max_power = PSD_[indices[0]]
            cut_off = 0.9*max_power
            num = len(PSD_[PSD_>cut_off])
            idx = indices[:num]
            cut_off = PSD[PSD>cut_off].min()-0.001
            fhat_new = fhat*(PSD>cut_off)
            fnew = np.real(ifft(fhat_new))
            ax[2].plot(t,fnew,color="salmon",label = "Filtered Data")
            ax[2].set_title("Filtered Data: Frequencies "+str(freq[idx]))
            ax[2].legend()
            plt.show()

        return freq[indices[0]],freq[indices[1]],np.real(PSD[indices[0]])
            
    def calculate_group_stats(self):
        stats = {}
        ed = self.example_duration
        for i in range(self.ed_groups+1):
            start = i*ed
            stop = i*ed+(ed-1)
            
            d = self.seconds_data(start,stop).dropna()
            
            stats[i]={}
            stats[i]['time'] = [start,stop]
            stats[i]['indices']=[d.index[0],d.index[-1]]
            stats[i]['total_time']=(self.time[d.index[-1]]-
                                    self.time[d.index[0]]+
                                    self.delta_t)
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='resultant_ma')
            stats[i]['fft_freq1_res_ma']=freq1
            stats[i]['fft_freq2_res_ma']=freq2
            stats[i]['fft_mag_res_ma']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='x_ma')
            stats[i]['fft_freq1_x_ma']=freq1
            stats[i]['fft_freq2_x_ma']=freq2
            stats[i]['fft_mag_x_ma']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='y_ma')
            stats[i]['fft_freq1_y_ma']=freq1
            stats[i]['fft_freq2_y_ma']=freq2
            stats[i]['fft_mag_y_ma']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='z_ma')
            stats[i]['fft_freq1_z_ma']=freq1
            stats[i]['fft_freq2_z_ma']=freq2
            stats[i]['fft_mag_z_ma']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='resultant_exp')
            stats[i]['fft_freq1_res_exp']=freq1
            stats[i]['fft_freq2_res_exp']=freq2
            stats[i]['fft_mag_res_exp']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='x_exp')
            stats[i]['fft_freq1_x_exp']=freq1
            stats[i]['fft_freq2_x_exp']=freq2
            stats[i]['fft_mag_x_exp']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='y_exp')
            stats[i]['fft_freq1_y_exp']=freq1
            stats[i]['fft_freq2_y_exp']=freq2
            stats[i]['fft_mag_y_exp']=magnitude
            freq1,freq2,magnitude = self.fft_denoise(d,graph=0,axis='z_exp')
            stats[i]['fft_freq1_z_exp']=freq1
            stats[i]['fft_freq2_z_exp']=freq2
            stats[i]['fft_mag_z_exp']=magnitude
            
        self.stats=pd.DataFrame(stats).T
        X = self.df[['x','y','z']].values
        X1,X2 = self.convert_tensor(X)
        df = self.fast_stats(X1)
        df = pd.concat([df,self.fast_stats(X2)],axis=0).reset_index()
        df = df.drop('index',axis=1)
        self.stats=pd.concat([self.stats,df],axis=1)
        
        return None
    
    def convert_tensor(self,X):
        """
        X is an ndarray (n,3) having the original x,y,z acceleration values
            sampled at the frequency rate. So,
            n = sample_rate * number_of_seconds = number of rows in X
            The 3 columns are x,y,z accelerations.
            
        This function converts from a 2 dimensional numpy array to a 
            3 dimensional tensor in order to speed up the calculations
            (see fast stats).
        
        """
        X = np.c_[X,np.sqrt(X[:,0]**2+X[:,1]**2+X[:,2]**2)]
        rows = self.example_duration*self.freq #axis 1
        groups = len(X)//rows #axis 0
        tot = groups*rows
        X_new = X[:tot].reshape(groups,rows,4)
        rows=len(X[tot:])
        X_final = X[tot:].reshape(1,rows,4)
        
        return X_new,X_final
        
        
    def fast_stats(self,X):
        """
        Input: X is a 3 dimensional numpy array with 
        0th axis representing the observation in the ith set
            for a 60Hz setting and 10 second example duration=>
                0=0:600, 1=600:1200,...
            This axis has a length equal to the total number of groups of data
        1st axis represents the rows in each file (observation)
        2nd axis represents the columns (x,y,z,resultant)
        
        If we sum or find the mean with respect to axis 1, then for each file,
            we sum / find the mean / whatever mathematical operation of all
            the observations in each file, turning each file into 4 numbers.
            
        """
        stats = pd.DataFrame()
        N = len(X)
        
        stats[['x_avg','y_avg','z_avg','res_avg']]=X.mean(axis=1)
        stats[['x_std','y_std','z_std','res_std']]=X.std(axis=1)
        stats[['x_abs','y_abs','z_abs','res_abs']]=np.abs(X-X.mean(axis=1).reshape(N,1,4)).sum(axis=1)
        stats[['x_min','y_min','z_min','res_min']]=X.min(axis=1)
        stats[['x_max','y_max','z_max','res_max']]=X.max(axis=1)
        stats[['x_diff','y_diff','z_diff','res_diff']]=X.max(axis=1)-X.min(axis=1)
        
        #0th axis is the file or the observations 0 = 0:200, 1 = 200:400, etc
        #1st axis is the rows in each file, we are calculating fft by rows, so take the top 50
        #2nd axis controls the columns, 0 = x, 1 = y, 2 = z, 3 = resultant
        X_fft = np.abs(np.fft.fft(X,axis=1))[:,1:51,:]
        stats[['x_fft_mean','y_fft_mean','z_fft_mean','res_fft_mean']]=X_fft.mean(axis=1)
        stats[['x_fft_std','y_fft_std','z_fft_std','res_fft_std']]=X_fft.std(axis=1)
        stats[['x_fft_abs','y_fft_abs','z_fft_abs','res_fft_abs']]=np.abs(X_fft-X_fft.mean(axis=1)
                                                                          .reshape(N,1,4)).sum(axis=1)
        stats[['x_fft_min','y_fft_min','z_fft_min','res_fft_min']]=X_fft.min(axis=1)
        stats[['x_fft_max','y_fft_max','z_fft_max','res_fft_max']]=X_fft.max(axis=1)
        stats[['x_fft_diff','y_fft_diff','z_fft_diff','res_fft_diff']]=X_fft.max(axis=1)-X_fft.min(axis=1)
        
        return stats
        
    
    def ma(self,N):
        self.N = N
        cols = ['x','y','z','resultant']
        ma_cols = [col+'_ma' for col in cols] 
        for i,col in enumerate(cols):
            self.df[ma_cols[i]] = self.df[col].rolling(N).mean()
                
    def exp_smooth(self,alpha):
        self.alpha = alpha
        cols = ['x','y','z','resultant']
        exp_cols = [col+'_exp' for col in cols]
        for i,col in enumerate(cols):
            self.df[exp_cols[i]] = self.df[col].ewm(alpha=alpha).mean()
       
        
        
        
    
    def time_series_plots(self):
        col1, col2,col3 = st.columns(3)
        with col1:
            cols = st.multiselect(":blue[Choose the data to plot:]",
                                  options = self.df.columns)
        with col2:
            N = st.slider(":blue[Moving Average lag:]",value = self.N,
                          min_value = 3,
                          max_value = self.freq)
            self.ma(N)            
            
        with col3:
            alpha = st.number_input(r":blue[Exponential Smooth - $\alpha$]",
                                     value = self.alpha)
            self.exp_smooth(alpha)
            #self.alpha = alpha
        
        data = self.df.copy()

        fig = px.line(data,
                      x='t',
                      y = cols)
        st.plotly_chart(fig)
        st.write("Delete data that is not cyclic in order")
        col1,col2 = st.columns(2)
        with col1:
            t0 = st.number_input(":blue[Select Starting Time]",
                                 value=self.time[0])
        with col2:
            t1 = st.number_input(":blue[Select End Time]",
                                 value=self.time[-1])
        reset_data = st.button("Reset Data")
        if reset_data:
            data=data[(data['t']>t0) & (data['t']<t1)]
            #self.__init__(data['time','x','y','z'],60)
            st.write(data[['time','x','y','z']])
            self.__init__(data[['time','x','y','z']],60)
        
        
        
        
    def critical_pts(self):
        col1,col2 = st.columns(2)
        with col1:
            cols = st.multiselect(":blue[Choose the data:]",
                                  options = self.df.columns,
                                  key = "critical_pts_col")
        with col2:
            t0 = st.slider(":blue[Start Time:]", 
                           value = int(self.time[0]),
                           min_value = int(self.time[0]),
                           max_value = int(self.time[-1])) #dfd
            t1 = st.slider("End Time:", 
                           value = int(self.time[-1]),
                           min_value = int(self.time[0]),
                           max_value = int(self.time[-1]))

        
        data = self.seconds_data(t0,t1).dropna()
        st.divider()
        fft={}
        if len(cols)>1:
            for col in cols:
                fft[col]={}
                fft[col]['freq1'],fft[col]['freq2'],fft[col]['mag']=self.fft_denoise(data,graph=0,axis=col)
            fft_df = pd.DataFrame(fft).T
            #st.write(tot_time)
            st.write(fft_df)
    
    def model_settings(self):
        col1,col2,col3 = st.columns(3)
        with col1:
            ed = st.number_input(":red[Example Duration:] ",
                                 min_value = 5, 
                                 max_value = 60, 
                                 step=5,
                                 value = self.example_duration)
        with col2:
            N = st.number_input(":red[Moving Average Period (N):]",
                                    min_value = 5, 
                                    max_value = self.freq,
                                    value = self.N)
        with col3:
            alpha = st.number_input(r":red[Exponential Smooth ($\alpha$):]",
                                    min_value = 0.05,
                                    max_value = 0.5,
                                    step=0.05,
                                    value = self.alpha)
        btn_settings = st.button("Change")
        
        if btn_settings:
            if N != self.N:
                self.ma(N)
            if alpha != self.alpha:
                self.exp_smooth(alpha)
            if ed != self.example_duration:
                self.example_duration = ed
                self.ed_groups = int(self.time[-1]/self.example_duration)
            self.calculate_group_stats()
        st.write(self.stats)
            
    def step_model(self):
        def arg_max(x):
            magnitudes = [x['fft_mag_res_ma'],x['fft_mag_x_ma'],
                       x['fft_mag_y_ma'],x['fft_mag_z_ma'],
                       x['fft_mag_res_exp'],x['fft_mag_x_exp'],
                       x['fft_mag_y_exp'],x['fft_mag_z_exp']]
            idx = np.argmax(magnitudes)
            cols = [['fft_freq1_res_ma','fft_freq2_res_ma'],
                    ['fft_freq1_x_ma','fft_freq2_x_ma'],
                    ['fft_freq1_y_ma','fft_freq2_y_ma'],
                    ['fft_freq1_z_ma','fft_freq2_z_ma'],
                    ['fft_freq1_res_exp','fft_freq2_res_exp'],
                    ['fft_freq1_x_exp','fft_freq2_x_exp'],
                    ['fft_freq1_y_exp','fft_freq2_y_exp'],
                    ['fft_freq1_z_exp','fft_freq2_z_exp']]
            
            col = cols[idx]
            #vals = np.array([x[col[0]],x[col[1]]])
            steps = x[col[0]]*x['total_time']
            return steps
        
        
        
        stats = self.stats.copy()
        stats['steps'] =stats.apply(arg_max,axis=1)
        st.write(stats)
        st.write(f"Total steps: {stats['steps'].sum():0.0f}")
        
    def step_graphic(self):
        pass
            

 
        
        
            


st.title("Acceleration Data Tool")
## Global session state variables###########################
if 'current' not in st.session_state:
    st.session_state['current'] = None
if 'calc' not in st.session_state:
    st.session_state['calc']=True
    
## Side bar radio button and options #######################
options = ["Files",
           "Statistics"
           "Graph",
           "Models"]
select = st.sidebar.radio(label = "Select the tool:",
                      options = options,
                      key='sb_select')
if select == options[0]:
    tab1, tab2 = st.tabs(["Import Data",
                               "Statistical Measures"])
    acc = st.session_state['current']
    with tab1:
        st.subheader("Import csv file")
        if acc is None:
            uploaded_file = st.file_uploader("Select .csv survey file.",type='csv')
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file,index_col=0,keep_default_na=True)
                acc = AccelerationData(df,57)
                st.session_state['current']=acc
                st.write(acc.df)
    with tab2:
        st.subheader("Statistical Measurements")
        acc = st.session_state['current']
        if acc is not None:
            st.write(acc.stats)
if select == options[1]:
    tab1,tab2,tab3 = st.tabs(["X,Y,Z, Resultant",
                         "Frequencies","Dataframe"])
    acc = st.session_state['current']
    with tab1:
        st.subheader(":blue[X, Y, Z, Resultant Time Series Plots]")
        if acc is not None:
            acc.time_series_plots()
            
    with tab2:
        st.subheader(":blue[Frequencies]")
        if acc is not None:
            acc.critical_pts()
    with tab3:
        if acc is not None:
            st.write(acc.stats)
            st.write(acc.stats.shape)
            
            
if select == options[2]:
    tab1,tab2,tab3 = st.tabs(["Settings",
                         "Model","Graphic"])
    acc = st.session_state['current']
    with tab1:
        st.subheader(":red[Settings]")
        if acc is not None:
            acc.model_settings()
    with tab2:
        if acc is not None:
            acc.step_model()
