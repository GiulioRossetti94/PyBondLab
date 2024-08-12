# -*- coding: utf-8 -*-
"""
Created on Sun Aug  12 09:40:34 2024

@author: Giulio Rossetti

Adapted from statsmodels so the code is heavily borrowed from statsmodels
see etc
"""
from PyBondLab.iotools.table import SimpleTable


class StrategyResults():
    """
    This class handles the results of a strategy. It is used to store the results of a strategy .fit() and is used as a base class for io operations.    

    """
    def __init__(self, params: dict, results: dict):
        """
        Initializes the StrategyResults object with parameters and results.

        :param params: A dictionary of parameters used in the strategy.
        :param results: A dictionary containing the results from the strategy fit.
        """
        self.params = params
        self.results = results 
    
    def _getnames(self, strategy_name=None, sortvar_name=None):
        """Extract names from parameters or construct names if not provided."""

        # Extract or construct the strategy name
        if strategy_name is None:
            strategy_name = self.params.get("strategy").__name__ if self.params.get("strategy") else ''

        # Extract or construct the sorting variable name
        if sortvar_name is None:
            sortvar_name = self.params.get("sort_var")
            if sortvar_name is None:
                # Fallback if sort_var is not in params; assume a default naming convention
                sortvar_name = ['var_%d' % i for i in range(len(self.results))]
        
        return strategy_name, sortvar_name

    def summary(
            self,
            sort_by: str | None = None,
            title: str | None = None,
            alpha: float = 0.05,
    ):
        """
        Generates a summary of the results of the strategy.

        :param title: The title of the summary.
        :param alpha: The alpha level for the confidence intervals.
        """
        # Extract the strategy name and sorting variable name
        strategy_name, sortvar_name = self._getnames()
        
        left_side = [('Strategy Name:', None),
                     ('Sorting Variable:', None),
                     ('Date:',None),
                     ('Time',None),
                     ('Start Date',None),
                     ('End Date',None),
                     ('Number of periods',None) ,
                     ]
        right_side = []

        if title is None:
            title = self.params['strategy'].__strategy_name__ + ' Strategy Results Summary'

        smry = Summary()
        smry.add_table_2cols(self, title=title, gleft=left_side, gright = right_side, xname = sortvar_name)


        raise NotImplementedError
    

class Summary:
    """
    Result summary

    Construction does not take any parameters. Tables and text can be added
    with the `add_` methods.

    Attributes
    ----------
    tables : list of tables
        Contains the list of SimpleTable instances, horizontally concatenated
        tables are not saved separately.
    extra_txt : str
        extra lines that are added to the text output, used for warnings
        and explanations.
    """
    def __init__(self):
        self.tables = []
        self.extra_txt = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        """Display as HTML in IPython notebook."""
        return self.as_html()

    def _repr_latex_(self):
        """Display as LaTeX when converting IPython notebook to PDF."""
        return self.as_latex()

    def add_table_2cols(self, res,  title=None, gleft=None, gright=None,
                        xname=None):
        """
        Add a double table, 2 tables with one column merged horizontally

        Parameters
        ----------
        res : results instance
            some required information is directly taken from the result
            instance
        title : str, optional
            if None, then a default title is used.
        gleft : list[tuple], optional
            elements for the left table, tuples are (name, value) pairs
            If gleft is None, then a default table is created
        gright : list[tuple], optional
            elements for the right table, tuples are (name, value) pairs
        yname : str, optional
            optional name for the endogenous variable, default is "y"
        xname : list[str], optional
            optional names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        """

        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            yname=yname, xname=xname)
        self.tables.append(table)

#     def add_table_params(self, res, yname=None, xname=None, alpha=.05,
#                          use_t=True):
#         """create and add a table for the parameter estimates

#         Parameters
#         ----------
#         res : results instance
#             some required information is directly taken from the result
#             instance
#         yname : {str, None}
#             optional name for the endogenous variable, default is "y"
#         xname : {list[str], None}
#             optional names for the exogenous variables, default is "var_xx"
#         alpha : float
#             significance level for the confidence intervals
#         use_t : bool
#             indicator whether the p-values are based on the Student-t
#             distribution (if True) or on the normal distribution (if False)

#         Returns
#         -------
#         None : table is attached

#         """
#         if res.params.ndim == 1:
#             table = summary_params(res, yname=yname, xname=xname, alpha=alpha,
#                                    use_t=use_t)
#         elif res.params.ndim == 2:
#             _, table = summary_params_2dflat(res, endog_names=yname,
#                                              exog_names=xname,
#                                              alpha=alpha, use_t=use_t)
#         else:
#             raise ValueError('params has to be 1d or 2d')
#         self.tables.append(table)

#     def add_extra_txt(self, etext):
#         """add additional text that will be added at the end in text format

#         Parameters
#         ----------
#         etext : list[str]
#             string with lines that are added to the text output.

#         """
#         self.extra_txt = '\n'.join(etext)

#     def as_text(self):
#         """return tables as string

#         Returns
#         -------
#         txt : str
#             summary tables and extra text as one string

#         """
#         txt = summary_return(self.tables, return_fmt='text')
#         if self.extra_txt is not None:
#             txt = txt + '\n\n' + self.extra_txt
#         return txt

#     def as_latex(self):
#         """return tables as string

#         Returns
#         -------
#         latex : str
#             summary tables and extra text as string of Latex

#         Notes
#         -----
#         This currently merges tables with different number of columns.
#         It is recommended to use `as_latex_tabular` directly on the individual
#         tables.

#         """
#         latex = summary_return(self.tables, return_fmt='latex')
#         if self.extra_txt is not None:
#             latex = latex + '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
#         return latex

#     def as_csv(self):
#         """return tables as string

#         Returns
#         -------
#         csv : str
#             concatenated summary tables in comma delimited format

#         """
#         csv = summary_return(self.tables, return_fmt='csv')
#         if self.extra_txt is not None:
#             csv = csv + '\n\n' + self.extra_txt
#         return csv

#     def as_html(self):
#         """return tables as string

#         Returns
#         -------
#         html : str
#             concatenated summary tables in HTML format

#         """
#         html = summary_return(self.tables, return_fmt='html')
#         if self.extra_txt is not None:
#             html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
#         return html
    
# def _getnames(self, strategy_name=None, sortvar_name=None):
#     '''extract names from model or construct names
#     '''
#     if strategy_name is None:
#         if getattr(self.model, 'endog_names', None) is not None:
#             strategy_name = self.model.endog_names
#         else:
#             strategy_name = ''

#     if sortvar_name is None:
#         if getattr(self.model, 'exog_names', None) is not None:
#             sortvar_name = self.model.exog_names
#         else:
#             sortvar_name = ['var_%d' % i for i in range(len(self.params))]

#     return strategy_name, sortvar_name

# def summary_top(results, title=None, gleft=None, gright=None, yname=None, xname=None):
#     '''generate top table(s)
#     '''
#     #change of names ?
#     gen_left, gen_right = gleft, gright

#     # time and names are always included
#     time_now = time.localtime()
#     time_of_day = [time.strftime("%H:%M:%S", time_now)]
#     date = time.strftime("%a, %d %b %Y", time_now)

#     yname, xname = _getnames(results, yname=yname, xname=xname)

#     # create dictionary with default
#     # use lambdas because some values raise exception if they are not available
#     default_items = dict([
#           ('Strategy Name:', lambda: [yname]),
#           ('Sorting Variable:', lambda: [yname]),
#           ('Date:', lambda: [date]),
#           ('Time:', lambda: time_of_day),
#           ('Start Date:', lambda: [start_date]),
#           ('End Date:', lambda: [end_date]),
#           ('No. Periods:', lambda: [d_or_f(results.nobs)]),
#     ])

#     if title is None:
#         title = results.model.__class__.__name__ + 'Regression Results'

#     if gen_left is None:
#         # default: General part of the summary table, Applicable to all? models
#         gen_left = [('Dep. Variable:', None),
#                     ('Model type:', None),
#                     ('Date:', None),
#                     ('No. Observations:', None),
#                     ('Df model:', None),
#                     ('Df resid:', None)]

#         try:
#             llf = results.llf  # noqa: F841
#             gen_left.append(('Log-Likelihood', None))
#         except: # AttributeError, NotImplementedError
#             pass

#         gen_right = []

#     gen_title = title
#     gen_header = None

#     # replace missing (None) values with default values
#     gen_left_ = []
#     for item, value in gen_left:
#         if value is None:
#             value = default_items[item]()  # let KeyErrors raise exception
#         gen_left_.append((item, value))
#     gen_left = gen_left_

#     if gen_right:
#         gen_right_ = []
#         for item, value in gen_right:
#             if value is None:
#                 value = default_items[item]()  # let KeyErrors raise exception
#             gen_right_.append((item, value))
#         gen_right = gen_right_

#     # check nothing was missed
#     missing_values = [k for k,v in gen_left + gen_right if v is None]
#     assert missing_values == [], missing_values

#     # pad both tables to equal number of rows
#     if gen_right:
#         if len(gen_right) < len(gen_left):
#             # fill up with blank lines to same length
#             gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
#         elif len(gen_right) > len(gen_left):
#             # fill up with blank lines to same length, just to keep it symmetric
#             gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

#         # padding in SimpleTable does not work like I want
#         #force extra spacing and exact string length in right table
#         gen_right = [('%-21s' % ('  '+k), v) for k,v in gen_right]
#         gen_stubs_right, gen_data_right = zip_longest(*gen_right) #transpose row col
#         gen_table_right = SimpleTable(gen_data_right,
#                                       gen_header,
#                                       gen_stubs_right,
#                                       title=gen_title,
#                                       txt_fmt=fmt_2cols
#                                       )
#     else:
#         gen_table_right = []  #because .extend_right seems works with []

#     #moved below so that we can pad if needed to match length of gen_right
#     #transpose rows and columns, `unzip`
#     gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

#     gen_table_left = SimpleTable(gen_data_left,
#                                  gen_header,
#                                  gen_stubs_left,
#                                  title=gen_title,
#                                  txt_fmt=fmt_2cols
#                                  )

#     gen_table_left.extend_right(gen_table_right)
#     general_table = gen_table_left

#     return general_table
