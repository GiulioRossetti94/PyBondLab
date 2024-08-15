# -*- coding: utf-8 -*-
"""
Created on Sun Aug  12 09:40:34 2024

@author: Giulio Rossetti

Adapted from statsmodels so the code is heavily borrowed from statsmodels
see https://github.com/statsmodels/statsmodels for more info
"""
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
from itertools import zip_longest
from PyBondLab.iotools.table import SimpleTable


class StrategyResults():
    """
    This class handles the results of a strategy. It is used to store the results of a strategy .fit() and is used as a base class for io operations.    

    """
    def __init__(self, params_input: dict, results: dict):
        """
        Initializes the StrategyResults object with parameters and results.

        :param params: A dictionary of parameters used in the strategy.
        :param results: A dictionary containing the results from the strategy fit.
        """
        self.params_input = params_input
        self.results = results 
        self.combined_res = {**params_input, **results}
        self.test()

    def compute_tstat_pval(self, df, nw_lag=0):
        """"""
        coln = df.shape[1]

        store_res = pd.DataFrame(np.zeros((4,coln)),columns = df.columns,index=['Avg','std err','t','P>|t|'])
        for i in range(coln):
            mod = sm.OLS(df.iloc[:,i]*100,np.ones_like(df.iloc[:,i]),missing='drop').fit(cov_type='HAC',
                             cov_kwds={'maxlags': nw_lag})
            store_res.iloc[0,i] = mod.params.iloc[0]
            store_res.iloc[1,i] = mod.bse.iloc[0]
            store_res.iloc[2,i] = mod.tvalues.iloc[0]
            store_res.iloc[3,i] = mod.pvalues.iloc[0]
        return store_res

    def test(self):
        """
        Returns the se t-values and pval of the parameters.
        """
        target_keys = ['ewls', 'vwls', 'ewl', 'ews', 'vwl', 'vws','ewport','vwport']

        if self.combined_res.get("filters") != {}:
            target_keys += [f"{x}_ep" for x in target_keys]

        stats_res = {}
        
        # Iterate over the predefined keys
        for key in target_keys:
            df = self.combined_res.get(key)
            if isinstance(df, pd.DataFrame):  # Ensure it's a DataFrame
                stats_res[key] = self.compute_tstat_pval(df)
            else:
                if key in ['ewport','vwport']:
                    df = pd.DataFrame(df)
                    stats_res[key] = self.compute_tstat_pval(df)
        self.stats_res = stats_res
        return self
    
    def summary(self, type = "ew"):
        if type == 'ew':
            typename = '(Equal Weighted)'
            which_w =  ['ewls', 'ewl', 'ews']   
            full_port = 'ewport'
            chars_type = 'ew_chars'
            turnover_type = 'ew_turnover'
        elif type == 'vw':
            typename = '(Value Weighted)'
            which_w =  ['vwls', 'vwl', 'vws']
            full_port = 'vwport'
            chars_type = 'vw_chars'
            turnover_type = 'vw_turnover'

        left_side = [('Strategy Name:', None),
                ('Sorting Variable:', None),
                ('Holding Period:', None),
                ('Date:',None),
                ('Time:',None),
                ('Start Date:',None),
                ('End Date:',None),
                ('No. of Periods:',None) ,
                ]
        
        right_side = [('No. Unique Bonds:', None),
                ('Rating:', None),
                ('Turnover Banding:',None),
                ('Filtering Choice:',None),
                ('Level:',None),
                ('Location:',None),
                ]
        
        title = self.params_input['strategy'].__strategy_name__ + ' Strategy Results Summary ' + typename

        smry = Summary()
        smry.add_table_2cols(self, title=title, gleft=left_side, gright = right_side, xname = None)

        # add information about long-short, long, short portfolios
        colname = ['Long-Short', 'Long', 'Short']
        df = pd.concat([self.stats_res[key] for key in which_w], axis=1)
        df.columns = colname
        # now add the ex post results if we have them
        if self.combined_res.get("filters") != {}:
            # augment the which_w list
            which_w += [f"{x}_ep" for x in which_w]

            df_ep = pd.concat([self.stats_res[key] for key in which_w], axis=1)
            df_ep.columns = colname + [f"{x}*" for x in colname]
            df = pd.concat([df,df_ep], axis=1)
        
        self.params = df.loc["Avg"]
        self.bse    = df.loc["std err"]
        self.tvalues= df.loc["t"]
        self.pvalues= df.loc["P>|t|"]
        smry.add_table_params(self, xname=None)
        if self.combined_res.get("filters") != {}:
            # add the ex post results
            smry.add_extra_txt(['* Ex post (unfeasible) results'])

        # add information about all the portfolios
        df_full = self.stats_res[full_port]
        df_full.columns = [f"Q{x}" for x in range(1,df_full.shape[1]+1)]
        self.params = df_full.loc["Avg"]
        self.bse    = df_full.loc["std err"]
        self.tvalues= df_full.loc["t"]
        self.pvalues= df_full.loc["P>|t|"]

        smry.add_table_params(self, xname=None)

        # if chars and turnover are not None, we add that text as well
        add_info = pd.DataFrame()
        if self.combined_res.get("turnover") is not None:
            turn  = _getturnover(self,which_turnover = turnover_type)
            add_info = pd.concat([add_info, turn])
        if self.combined_res.get("chars") is not None:
            chars = _getchars(self, which_char = chars_type)
            add_info = pd.concat([add_info, chars])

        if not add_info.empty:
            self.additional_info = add_info
            smry.add_table_info(self, title='Additional Information',)
        
        if self.combined_res.get("filters") != {}:
            add_info = pd.DataFrame()
            if self.combined_res.get("turnover") is not None:
                turn  = _getturnover(self,which_turnover = turnover_type+"_ep")
                add_info = pd.concat([add_info, turn])
            if self.combined_res.get("chars") is not None:
                chars = _getchars(self, which_char = chars_type+"_ep")
                add_info = pd.concat([add_info, chars])

            if not add_info.empty:
                self.additional_info = add_info
                smry.add_table_info(self, title='Additional Information (Ex Post)*',)

        
        return smry
    
def _getnames(res,strategy_name=None, sortvar_name=None,hp=None):
    """Extract names from parameters or construct names if not provided."""
    # Extract or construct the strategy name
    if strategy_name is None:
        strategy = res.combined_res.get("strategy")
        strategy_name = strategy.__strategy_name__ if strategy else ''

    # Extract or construct the sorting variable name
    if sortvar_name is None:
        strategy = res.combined_res.get("strategy")
        sortvar_name = strategy.sort_var if strategy else None
        if sortvar_name is None:
            # Fallback if sort_var is not in params; assume a default naming convention
            sortvar_name = ['UNDEFINED']
    
    # Extract or construct the holding period
    if hp is None:
        strategy = res.combined_res.get("strategy")
        hp = strategy.K if strategy else None
        
    return strategy_name, sortvar_name,hp

def _getchars(res, which_char = None):
    """Extract chars from parameters or construct chars if not provided."""
    # Extract or construct the chars
    if which_char is None:
        chars = None
    else:
        chars_dict = res.combined_res.get(which_char) # Retrieve chars_dict for the current which_char
        char_list = []
        for key, df in chars_dict.items():
            avg = df.mean(axis=0).to_frame().T  # Calculate the mean
            avg.index = [key]
            char_list.append(avg)  # Append the result to char_list
        chars = pd.concat(char_list)
    return chars

def _getturnover(res,which_turnover = None):
    """Extract turnover from parameters or construct turnover if not provided."""
    # Extract or construct the turnover
    if which_turnover is None:
        turnover = None
    else:
        turnover = res.combined_res.get(which_turnover)
        avg_turn = turnover.mean(axis=0).to_frame().T
        avg_turn.index = ['Turnover(%)']
    return avg_turn * 100

def _getdates(res, start_date=None, end_date=None, tot_periods=None):
    """Extract dates from parameters or construct dates if not provided."""
    # Extract or construct the start date
    if start_date is None:
        start_date = res.combined_res.get("start_date", None)
    if end_date is None:
        end_date = res.combined_res.get("end_date", None)
    if tot_periods is None:
        tot_periods = res.combined_res.get("tot_periods", None)
    return start_date, end_date, tot_periods

def _getratings(res, ratings=None):
    """Extract ratings from parameters or construct ratings if not provided."""
    # Extract or construct the ratings
    if ratings is None:
        ratings = res.combined_res.get("rating", None)
        if ratings is None:
            ratings = "All Bonds Included"
        elif ratings == "NIG":
            ratings = "Non Investment Grade"
        elif ratings == "IG":
            ratings = "Investment Grade"
    return ratings

def _getfilters(res, filters=None):
    """Extract filters from parameters or construct filters if not provided."""
    # Extract or construct the filters
    nameadj = {"trim":"Return Excl. (Trim)",
               "price":"Price Excl. (Trim)",
               "bounce":"Bounce-Back Excl.(Trim)",
               "wins":"Wins."}
    if filters is None:
        filters = res.combined_res.get("filters", None)
        if filters == {}:
            filters = "No filters applied"
            level = None
            location = None
        else:
            level = filters.get("level", None)
            location = filters.get("location", None)
            filters = filters.get("adj", None)
            if filters is not None:
                filters = nameadj[filters]
            # map to better name

    return filters, level, location
    
class Summary:
    """
    This code is from statsmodels and is used to create a summary table. It is used to create a summary table for the results of a strategy.
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
        xname : list[str], optional
            optional names for the exogenous variables, default is "var_xx".
            Must match the number of parameters in the model.
        """

        table = summary_top(res, title=title, gleft=gleft, gright=gright,
                            xname=xname)
        self.tables.append(table)

    def add_table_params(self, res, xname=None):
        """create and add a table for the parameter estimates

        Parameters
        ----------
        res : results instance
        xname : {list[str], None}
            optional names for the exogenous variables, default is "var_xx"
        """
        table = summary_params(res, xname=xname)
        self.tables.append(table)

    def add_table_info(self,res, title=None,xname=None ):
        """create and add a table for additional information  """
        table = summary_info(res, title=title, xname=xname)
        self.tables.append(table)

    def add_extra_txt(self, etext):
        """add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.

        """
        self.extra_txt = '\n'.join(etext)

    def as_text(self):
        """return tables as string

        Returns
        -------
        txt : str
            summary tables and extra text as one string

        """
        txt = summary_return(self.tables, return_fmt='text')
        if self.extra_txt is not None:
            txt = txt + '\n\n' + self.extra_txt
        return txt

    def as_latex(self):
        """return tables as string

        Returns
        -------
        latex : str
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        """
        latex = summary_return(self.tables, return_fmt='latex')
        if self.extra_txt is not None:
            latex = latex + '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
        return latex

    def as_csv(self):
        """return tables as string

        Returns
        -------
        csv : str
            concatenated summary tables in comma delimited format

        """
        csv = summary_return(self.tables, return_fmt='csv')
        if self.extra_txt is not None:
            csv = csv + '\n\n' + self.extra_txt
        return csv

    def as_html(self):
        """return tables as string

        Returns
        -------
        html : str
            concatenated summary tables in HTML format

        """
        html = summary_return(self.tables, return_fmt='html')
        if self.extra_txt is not None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html

def summary_top(results, title=None, gleft=None, gright=None, xname=None):
    '''generate top table(s)
    '''
    #change of names ?
    gen_left, gen_right = gleft, gright

    # time and names are always included
    time_now = time.localtime()
    time_of_day = [time.strftime("%H:%M:%S", time_now)]
    date = time.strftime("%a, %d %b %Y", time_now)

    # get params for left side
    strategy_name, sortvar_name,hp = _getnames(results,strategy_name=None, sortvar_name=None,hp=None)
    start_date,end_date, tot_periods = _getdates(results, start_date=None, end_date=None, tot_periods=None)

    start_date = start_date.strftime("%m/%d/%Y") if start_date is not None else ''
    end_date = end_date.strftime("%m/%d/%Y") if end_date is not None else ''

    # create dictionary with default
    # use lambdas because some values raise exception if they are not available
    default_items = dict([
          ('Strategy Name:', lambda: [strategy_name]),
          ('Sorting Variable:', lambda: [sortvar_name]),
          ('Holding Period:', lambda: [hp]),
          ('Date:', lambda: [date]),
          ('Time:', lambda: time_of_day),
          ('Start Date:', lambda: [start_date]),
          ('End Date:', lambda: [end_date]),
          ('No. of Periods:', lambda: [tot_periods]),
    ])

    if title is None:
        title =  'Strategy Results Summary'

    if gen_left is None:
        # default: General part of the summary table, Applicable to all? models
        gen_left = [('Strategy Name:', None),
                    ('Sorting Variable:', None),
                    ('Holding Period:', lambda: [hp]), 
                    ('Date:', None),
                    ('No. of Periods:', None),
                    ('Start Date:', None),
                    ('End Date:', None)]
        
    # get params for right side
    ratings = _getratings(results, ratings=None)
    filters, level, location = _getfilters(results, filters=None)
    nbonds = results.combined_res.get('nbonds')
    banding = results.combined_res.get('banding_threshold', None)
        
    default_items_right = dict([
          ('No. Unique Bonds:', lambda: [nbonds]),
          ('Rating:', lambda: [ratings]),
          ('Turnover Banding:', lambda: [banding]),
          ('Filtering Choice:', lambda: [filters]),
          ('Level:', lambda: [level]),
          ('Location:', lambda: [location]),
    ])

    if gen_right is None:
        # default: General part of the summary table, Applicable to all? models
        gen_right = [('No. Unique Bonds:', None),
                    ('Rating:', None),
                    ('Turnover Banding:', None),
                    ('Filtering Choice:', None),
                    ]

    gen_title = title
    gen_header = None

    # replace missing (None) values with default values
    gen_left_ = []
    for item, value in gen_left:
        if value is None:
            value = default_items[item]()  # let KeyErrors raise exception
        gen_left_.append((item, value))
    gen_left = gen_left_

    if gen_right:
        gen_right_ = []
        for item, value in gen_right:
            if value is None:
                value = default_items_right[item]()  # let KeyErrors raise exception
            gen_right_.append((item, value))
        gen_right = gen_right_

    # check nothing was missed
    missing_values = [k for k,v in gen_left + gen_right if v is None]
    assert missing_values == [], missing_values

    # pad both tables to equal number of rows
    if gen_right:
        if len(gen_right) < len(gen_left):
            # fill up with blank lines to same length
            gen_right += [(' ', ' ')] * (len(gen_left) - len(gen_right))
        elif len(gen_right) > len(gen_left):
            # fill up with blank lines to same length, just to keep it symmetric
            gen_left += [(' ', ' ')] * (len(gen_right) - len(gen_left))

        # padding in SimpleTable does not work like I want
        #force extra spacing and exact string length in right table
        gen_right = [('%-21s' % ('  '+k), v) for k,v in gen_right]
        gen_stubs_right, gen_data_right = zip_longest(*gen_right) #transpose row col
        gen_table_right = SimpleTable(gen_data_right,
                                      gen_header,
                                      gen_stubs_right,
                                      title=gen_title,
                                      txt_fmt=fmt2
                                      )
    else:
        gen_table_right = []  #because .extend_right seems works with []

    #moved below so that we can pad if needed to match length of gen_right
    #transpose rows and columns, `unzip`
    gen_stubs_left, gen_data_left = zip_longest(*gen_left) #transpose row col

    gen_table_left = SimpleTable(gen_data_left,
                                 gen_header,
                                 gen_stubs_left,
                                 title=gen_title,
                                 txt_fmt=fmt_2cols
                                 )

    gen_table_left.extend_right(gen_table_right)
    general_table = gen_table_left

    return general_table

def summary_params(results, xname=None, skip_header=False, title=None):
    '''create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    xname : {list[str], None}
        optional names for the exogenous variables, default is "var_xx"
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    '''

    # Parameters part of the summary table
    params = np.asarray(results.params)
    std_err = np.asarray(results.bse)
    tvalues = np.asarray(results.tvalues)  # is this sometimes called zvalues
    pvalues = np.asarray(results.pvalues)

    param_header = ['coef(%)', 'std err', 't', 'P>|t|']
    if skip_header:
        param_header = None

    xname = results.params.index.tolist()
    params_stubs = results.params.index.tolist()

    exog_idx = lrange(len(xname))
    params = np.asarray(params)
    std_err = np.asarray(std_err)
    tvalues = np.asarray(tvalues)
    pvalues = np.asarray(pvalues)
    params_data = lzip([forg(params[i], prec=4) for i in exog_idx],
                       [forg(std_err[i]) for i in exog_idx],
                       [forg(tvalues[i]) for i in exog_idx],
                       ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                       )
    # parameter_table = SimpleTable(params_data,
    #                               param_header,
    #                               params_stubs,
    #                               title=title,
    #                               txt_fmt=fmt_params
    #                               )
    # transpose this is a bit of a hack but it works for now
    params_data = list(zip(*params_data))
    parameter_table = SimpleTable(params_data,
                                  params_stubs,
                                  param_header,
                                  title=title,
                                  txt_fmt=fmt_params
                                  )
    return parameter_table

def summary_info(results, title=None, xname=None,skip_header=False):
    '''create a summary table for additional information

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    title : str, optional
        if None, then a default title is used.
    xname : list[str], optional
        optional names for the exogenous variables, default is "var_xx"
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    info_table : SimpleTable instance
    '''

    # get the info
    params = np.asarray(results.additional_info)
    # use as header the index of the DataFrame eg turnover and chars
    # remember the header is actually the index in the summary. sloppy. TODO
    param_header = results.additional_info.index.tolist()

    if skip_header:
        param_header = None

    params_stubs = results.additional_info.columns.tolist() # this is the number of ptfs
    ninfo = lrange(len(results.additional_info.index.tolist()))
    nportf = lrange(len(results.additional_info.columns.tolist()))
    params = np.asarray(params)

    params_data = [[forg(params[i, j],prec=4) for j in nportf] for i in ninfo]

    parameter_table = SimpleTable(params_data,
                                params_stubs,
                                param_header,
                                title=title,
                                txt_fmt=fmt_params
                                )
    return parameter_table

def table_extend(tables, keep_headers=True):
    """extend a list of SimpleTables, adding titles to header of subtables

    This function returns the merged table as a deepcopy, in contrast to the
    SimpleTable extend method.

    Parameters
    ----------
    tables : list of SimpleTable instances
    keep_headers : bool
        If true, then all headers are kept. If falls, then the headers of
        subtables are blanked out.

    Returns
    -------
    table_all : SimpleTable
        merged tables as a single SimpleTable instance

    """
    from copy import deepcopy
    for ii, t in enumerate(tables[:]): #[1:]:
        t = deepcopy(t)

        #move title to first cell of header
        # TODO: check if we have multiline headers
        if t[0].datatype == 'header':
            t[0][0].data = t.title
            t[0][0]._datatype = None
            t[0][0].row = t[0][1].row
            if not keep_headers and (ii > 0):
                for c in t[0][1:]:
                    c.data = ''

        # add separating line and extend tables
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format('txt', row_dec_below='-')
            table_all.extend(t)

    table_all.title = None
    return table_all

def summary_return(tables, return_fmt='text'):
    # join table parts then print
    if return_fmt == 'text':
        strdrop = lambda x: str(x).rsplit('\n',1)[0]
        # convert to string drop last line
        return '\n'.join(lmap(strdrop, tables[:-1]) + [str(tables[-1])])
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join(x.as_csv() for x in tables)
    elif return_fmt == 'latex':
        # TODO: insert \hline after updating SimpleTable
        table = copy.deepcopy(tables[0])
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()
    elif return_fmt == 'html':
        return "\n".join(table.as_html() for table in tables)
    else:
        raise ValueError('available output formats are text, csv, latex, html')

# the following code is verbatim (or with minor modifications) from statsmodels. Credits to statsmodels
def forg(x, prec=3):
    x = np.squeeze(x)
    if prec == 3:
        # for 3 decimals
        if (x) or (abs(x) < 1e-4):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    else:
        raise ValueError("`prec` argument must be either 3 or 4, not {prec}"
                         .format(prec=prec))

def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))

def lrange(*args, **kwargs):
    return list(range(*args, **kwargs))

def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))

gen_fmt = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 7,
    "colsep": '   ',
    "row_pre": '  ',
    "row_post": '  ',
    "table_dec_above": '": ',
    "table_dec_below": None,
    "header_dec_below": None,
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": "r",
    "stubs_align": "l",
    "fmt": 'txt'
}

# Note table_1l_fmt over rides the below formating unless it is not
# appended to table_1l
fmt_1_right = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 16,
    "colsep": '   ',
    "row_pre": '',
    "row_post": '',
    "table_dec_above": '": ',
    "table_dec_below": None,
    "header_dec_below": None,
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": "r",
    "stubs_align": "l",
    "fmt": 'txt'
}

fmt_2 = {
    "data_fmts": ["%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 10,
    "colsep": ' ',
    "row_pre": '  ',
    "row_post": '   ',
    "table_dec_above": '": ',
    "table_dec_below": '": ',
    "header_dec_below": '-',
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": 'r',
    "stubs_align": 'l',
    "fmt": 'txt'
}

# new version  # TODO: as of when?  compared to what?  is old version needed?
fmt_base = {
    "data_fmts": ["%s", "%s", "%s", "%s", "%s"],
    "empty_cell": '',
    "colwidths": 10,
    "colsep": ' ',
    "row_pre": '',
    "row_post": '',
    "table_dec_above": '=',
    "table_dec_below": '=',  # TODO need '=' at the last subtable
    "header_dec_below": '-',
    "header_fmt": '%s',
    "stub_fmt": '%s',
    "title_align": 'c',
    "header_align": 'r',
    "data_aligns": 'r',
    "stubs_align": 'l',
    "fmt": 'txt'
}

fmt_2cols = copy.deepcopy(fmt_base)

fmt2 = {
    "data_fmts": ["%18s", "-%19s", "%18s", "%19s"],  # TODO: TODO: what?
    "colsep": ' ',
    "colwidths": 18,
    "stub_fmt": '-%21s',
}
fmt_2cols.update(fmt2)

fmt_params = copy.deepcopy(fmt_base)

fmt3 = {
    "data_fmts": ["%s", "%s", "%8s", "%s", "%11s", "%11s"],
}
fmt_params.update(fmt3)

"""
Summary Table formating
This is here to help keep the formating consistent across the different models
"""
fmt_latex = {
    'colsep': ' & ',
    'colwidths': None,
    'data_aligns': 'r',
    'data_fmt': '%s',
    'data_fmts': ['%s'],
    'empty': '',
    'empty_cell': '',
    'fmt': 'ltx',
    'header': '%s',
    'header_align': 'c',
    'header_dec_below': '\\hline',
    'header_fmt': '%s',
    'missing': '--',
    'row_dec_below': None,
    'row_post': '  \\\\',
    'strip_backslash': True,
    'stub': '%s',
    'stub_align': 'l',
    'stub_fmt': '%s',
    'table_dec_above': '\\hline',
    'table_dec_below': '\\hline'}

fmt_txt = {
    'colsep': ' ',
    'colwidths': None,
    'data_aligns': 'r',
    'data_fmts': ['%s'],
    'empty': '',
    'empty_cell': '',
    'fmt': 'txt',
    'header': '%s',
    'header_align': 'c',
    'header_dec_below': '-',
    'header_fmt': '%s',
    'missing': '--',
    'row_dec_below': None,
    'row_post': '',
    'row_pre': '',
    'stub': '%s',
    'stub_align': 'l',
    'stub_fmt': '%s',
    'table_dec_above': '-',
    'table_dec_below': None,
    'title_align': 'c'}