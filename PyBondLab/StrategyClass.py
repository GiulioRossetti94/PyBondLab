# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 13:09:08 2024

@authors: Giulio Rossetti & Alex Dickerson
"""

import numpy as np

#==============================================================================
#   Abstract Strategy Class
#==============================================================================

class Strategy:
    def __init__(self, nport, K, J=None, skip=None,DoubleSort=0):
        self.DoubleSort = DoubleSort
        self.nport = nport
        self.K = K
        self.J = J
        self.skip = skip
        self.str_name = f"{self.K}"
        self.__strategy_name__ = None
        if self.J is not None:
            self.str_name += f"_{self.J}"
        if self.skip is not None:
            self.str_name += f"_{self.skip}"            

    def compute_signal(self, data):
        raise NotImplementedError("Each strategy must implement the compute_signal method.")
        
    def get_sort_var(self, adj=None):
        raise NotImplementedError("Each strategy must implement the get_sort_var method.")


#==============================================================================
#   SINGLE SORTING
#==============================================================================

class SingleSort(Strategy):   
    def __init__(self, K, sort_var,nport, J=None, skip=None):
        """
        Initializes the SingleSort strategy with required parameters.

        Parameters
        ----------
        nport : int
            The number of portfolios to sort into.
        K : int
            The holding period for the primary strategy.
        sort_var : str
            The variable to use for sorting.
        J : any, optional
            An additional parameter for strategy customization.
        skip : any, optional
            An additional parameter for strategy customization.
        """
        super().__init__(nport, K, J, skip)
        self.__strategy_name__ = "Single Sorting"
        
        self.sort_var = sort_var  # Sorting variable
        
        print("-" * 35)
        print(f"Initializing strategy (single sort):\nHolding period: {self.K} \nNumber of portfolios: {self.nport} \nSorting on: {self.sort_var}")
        print("-" * 35) 
        
    def compute_signal(self, data):
        return data
    
    def get_sort_var(self, adj=None):
        """
        Returns the variable used for sorting.
        -------
        str
            The name of the variable used for sorting.
        """
        return self.sort_var
    
    def set_sort_var(self, sort_var):
        """
        Sets the variable to be used for sorting.

        Parameters
        ----------
        sort_var : str
            The name of the variable to use for sorting.
        """
        self.sort_var = sort_var


#==============================================================================
#   DOUBLE SORTING
#==============================================================================
class DoubleSort(Strategy):
   
    def __init__(self, K,sort_var,nport, sort_var2,nport2,J=None, skip=None):
        """
        Initializes the DoubleSort strategy with required parameters.

        Parameters
        ----------
        nport : int
            The number of portfolios to sort into.
        K : int
            The holding period for the primary strategy.
        sort_var : str
            The variable to use for primary sorting.
        sort_var2 : str
            The variable to use for secondary sorting.
        nport2 : int
            The number of portfolios for secondary sorting.
        J : any, optional
            An additional parameter for strategy customization.
        skip : any, optional
            An additional parameter for strategy customization.
        """
        super().__init__(nport, K, J, skip)
        self.__strategy_name__ = "Double Sorting"
        
        self.DoubleSort = 1
        self.sort_var = sort_var  # Primary sorting variable
        self.sort_var2 = sort_var2  # Secondary sorting variable
        self.nport2 = nport2
        
        print("-" * 35)
        print(f"Initializing strategy (double sorts): \nHolding period: {self.K} \nNumber of portfolios: {self.nport}x{self.nport2} \nSorting on: {self.sort_var} and {self.sort_var2}")
        print("-" * 35) 
        
    def compute_signal(self, data):
        return data
    
    def get_sort_var(self, adj=None):
        """
        Returns
        -------
        str
            The name of the primary sorting variable.
        """
        return self.sort_var
    
    def get_sort_var2(self, adj=None):
        """
          Returns
        -------
        str
            The name of the secondary sorting variable.
        """
        return self.sort_var2
    
    def set_sort_var(self, sort_var):
        """
        Sets the primary sorting variable.

        Parameters
        ----------
        sort_var : str
            The name of the variable to use for primary sorting.
        """
        self.sort_var = sort_var
        
    
    def set_sort_var2(self, sort_var2):
        """
        Sets the secondary sorting variable.

        Parameters
        ----------
        sort_var2 : str
            The name of the variable to use for secondary sorting.
        """
        self.sort_var2 = sort_var2
        
    
    def set_nport2(self, nport2):
        """
        Sets the number of portfolios for secondary sorting.

        Parameters
        ----------
        nport2 : int
            The number of portfolios for secondary sorting.
        """
        self.nport2 = nport2
        
#==============================================================================
#   MOMENTUM: SingleSorting
#==============================================================================

class Momentum(Strategy):   
    def __init__(self, K,nport, J=None, skip=None):
        """
        Initializes the SingleSort strategy with required parameters.

        Parameters
        ----------
        nport : int
            The number of portfolios to sort into.
        K : int
            The holding period for the primary strategy.
        sort_var : str
            The variable to use for sorting.
        J : any, optional
            An additional parameter for strategy customization.
        skip : any, optional
            An additional parameter for strategy customization.
        """
        super().__init__(nport, K, J, skip)
        self.__strategy_name__ = "MOMENTUM"
        #self.sort_var = None  # Sorting variable
        
        print("-" * 35)
        print(f"Initializing Momentum ({self.J},{self.K}) strategy (single sort):\nHolding period: {self.K} \nNumber of portfolios: {self.nport} \nSorting on: past returns")
        print("-" * 35) 
        
    def compute_signal(self, data):
        varname = "ret"
        data['logret'] = np.log(data[varname] + 1)
        data['signal'] = data.groupby(['ID'], group_keys=False)['logret']\
            .rolling(self.J, min_periods=self.J).sum().values
        data['signal'] = np.exp(data['signal']) - 1    
        data['signal'] = data.groupby("ID")['signal'].shift(self.skip)
        return data
    
    def get_sort_var(self, adj=None):
        """
        Returns the variable used for sorting.
        -------
        str
            The name of the variable used for sorting.
        """
        self.sort_var = f'signal_{adj}' if adj else 'signal'
        return f'signal_{adj}' if adj else 'signal'
        
    def set_sort_var(self, sort_var):
        """
        Sets the variable to be used for sorting.

        Parameters
        ----------
        sort_var : str
            The name of the variable to use for sorting.
        """
        pass

#==============================================================================
#   Long Term Reversal: SingleSorting
#==============================================================================

class LTreversal(Strategy):   
    def __init__(self, K,nport, J=None, skip=None):
        """
        Initializes the SingleSort strategy with required parameters.

        Parameters
        ----------
        nport : int
            The number of portfolios to sort into.
        K : int
            The holding period for the primary strategy.
        sort_var : str
            The variable to use for sorting.
        J : any, optional
            An additional parameter for strategy customization.
        skip : any, optional
            An additional parameter for strategy customization.
        """
        super().__init__(nport, K, J, skip)
        self.__strategy_name__ = "LT-REVERSAL"
        #self.sort_var = None  # Sorting variable
        
        print("-" * 35)
        print(f"Initializing Long Term reversal ({self.J},{self.skip}) strategy (single sort):\nHolding period: {self.K} \nNumber of portfolios: {self.nport} \nSorting on: past returns")
        print("-" * 35) 
        
    def compute_signal(self, data):
        varname = "ret"
        # data['logret'] = np.log(data[varname] + 1)
        data['signal'] = data.groupby(['ID'], group_keys=False)[varname]\
            .apply(lambda x:  x.rolling(window = self.J).sum()) - \
                   data.groupby(['ID'], group_keys=False)[varname]\
            .apply(lambda x:  x.rolling(window = self.skip).sum())
                   
        return data
    
    def get_sort_var(self, adj=None):
        """
        Returns the variable used for sorting.
        -------
        str
            The name of the variable used for sorting.
        """
        self.sort_var = f'signal_{adj}' if adj else 'signal'
        return f'signal_{adj}' if adj else 'signal'
        
    def set_sort_var(self, sort_var):
        """
        Sets the variable to be used for sorting.

        Parameters
        ----------
        sort_var : str
            The name of the variable to use for sorting.
        """
        pass