import pandas as pd
import numpy as np
from scipy import linalg, stats
import json

class CA:
    def __init__(self, data, active_row, active_col, supp_cols=None, 
                 row_names=None, col_names=None, supp_col_names=None):
        '''
        Construct the base contingency table. 
        Input: 
            data: a Pandas dataframe
            activa_row: string; indicating a (categorical) variable name from the data;
                        assumed to have > 2 distinct values
            activa_col: string; indicating a (categorical) variable name from the data;
                        assumed to have > 2 distinct values
            supp_cols: a list of variables to use as reference 
            row_names: a Python dictionary with `keys`={old names}, `values`={new names}
            col_names: a Python dictionary with `keys`={old names}, `values`={new names}
            supp_col_names: a Python dictionary with `keys`={variable names}, 
                            `values`={Python dictionary with `keys`={old names}, `values={new names}}
        '''
        ## check conditions
        if len(data[active_row].unique()) < 3:
            raise Exception("Invalid row variable (need >=3 distinct categories.")
        elif len(data[active_col].unique()) < 3:
            raise Exception("Invalid col variable (need >=3 distinct categories.")
        data = data.copy(deep=True)

        ## dealing with renaming
        if row_names:
            data[active_row] = data[active_row].apply(lambda x: row_names[str(x)])
        if col_names:
            data[active_col] = data[active_col].apply(lambda x: col_names[str(x)])
        if supp_cols and supp_col_names:
            for col in supp_cols:
                if col in supp_col_names:
                    new_names = supp_col_names[col]
                    data[col] = data[col].apply(lambda x: new_names[str(x)])
        
        ## initiate
        self.data = data
        self.row = active_row
        self.col = active_col
        self.supp_cols = supp_cols

        ## form main contingency table and probability tables
        self.count = pd.crosstab(index=data[self.row], columns=data[self.col])
        self.prob = self.to_P(self.count)
        self.r, self.c = self.get_margins(self.prob)
        self.svd()

        ## if there are supplementary variables, form the contingency and prob tables
        if self.supp_cols:
            self.supp_count = {}
            self.supp_prob = {}
            for supp in self.supp_cols:
                self.supp_count[supp] = pd.crosstab(index=data[self.row], columns=data[supp])
                self.supp_prob[supp] = self.to_P(self.supp_count[supp])


    def to_P(self, table):
        return table.div(table.sum().sum())


    def get_margins(self, table):
        return table.sum(axis=1), table.sum(axis=0)


    def make_diag(self, v, exponent=1, extra_width=0, axis=1):
        if extra_width == 0:
            return np.diag(np.power(v, exponent))
        else:
            diag = np.diag(np.power(v, exponent))
            if axis == 1:
                zeros = np.zeros((diag.shape[0], extra_width))
            else:
                zeros = np.zeros((extra_width, diag.shape[1]))
            return np.concatenate([diag, zeros], axis=axis)


    def svd(self):
        E = np.expand_dims(self.r, axis=1) @ np.expand_dims(self.c, axis=0)
        self.expected = pd.DataFrame(E)
        self.expected.index = self.count.index
        self.expected.columns=self.count.columns
        X = self.make_diag(self.r, -1/2) @ (self.prob - E) @ self.make_diag(self.c, -1/2)
        self.chi2 = pd.DataFrame(X)
        self.chi2.index = self.count.index
        self.chi2.columns=self.count.columns
        self.U, self.s, self.Vh = linalg.svd(X)


    def get_sigma(self, exponent=1):
        if self.c.shape[0] < self.U.shape[0]:
            return self.make_diag(self.s, exponent, self.U.shape[0] - self.c.shape[0], axis=0)
        return self.make_diag(self.s, exponent, self.c.shape[0] - self.U.shape[0], axis=1)


    def get_row_coord(self, mode='principal'):
        F = self.make_diag(self.r, -1/2) @ self.U
        if mode == 'principal':
            F = F @ self.get_sigma()
        return F


    def get_col_coord(self, mode='principal'):
        G = self.make_diag(self.c, -1/2) @ self.Vh.T
        if mode == 'principal':
            G = G @ self.get_sigma().T
        return G


    def get_supp_coord(self, direction='col'):
        if direction == 'col' and self.supp_cols:
            supp_coord = {}
            for supp in self.supp_cols:
                P = self.supp_prob[supp]
                _, c = self.get_margins(P)
                G = self.make_diag(c, -1) @ P.T @ self.get_row_coord() @ self.get_sigma(-1).T
                supp_coord[supp] = G
            return supp_coord


    def get_corr(self, direction='row'):
        '''
        Returns the correlation coefficient matrix.
        Input: 
            M: assumed to be the principal coordinates of row (or column) profiles
            margins: assumed to be the row (or column) margins of the probability matrix
        Output:
            numpy array with dimension like M
        '''
        if direction == 'row':
            return np.power(self.get_row_coord(), 2) * np.expand_dims(self.r, axis=1)
        return np.power(self.get_col_coord(), 2) * np.expand_dims(self.c, axis=1)
    

    def generate_principal_inertia(self, return_dim=2, tojson=True):
        inertia = self.s
        perc_iner = self.s / self.s.sum()
        cum_iner = perc_iner.cumsum()
        df = pd.DataFrame({'inertia': inertia,
                           '% of iner': perc_iner,
                           'cum of iner': cum_iner})
        df = df.rename(lambda x: 'Dim '+str(x), axis=0)
        if tojson:
            return df.iloc[:return_dim, :].to_json()
        return df.iloc[:return_dim, :]
    

    def generate_ca_2d(self, row_mode='principal', col_mode='principal', tojson=True):
        F = self.get_row_coord(mode=row_mode)
        F_iner = self.get_corr(direction='row')
        G = self.get_col_coord(mode=col_mode)
        G_iner = self.get_corr(direction='col')
        row_s, col_s = self.get_margins(self.count)

        variable = [self.row] * len(self.count.index) + [self.col] * len(self.count.columns)
        category = list(self.count.index) + list(self.count.columns)
        is_active = ['True'] * (len(self.count.index) + len(self.count.columns))
        x_coord = list(F[:, 0]) + list(G[:, 0])
        x_iner = list(F_iner[:, 0]) + list(G_iner[:, 0])
        y_coord = list(F[:, 1]) + list(G[:, 1])
        y_iner = list(F_iner[:, 1]) + list(G_iner[:, 1])
        count = list(row_s) + list(col_s)
        
        if self.supp_cols:
            supp_dict = self.get_supp_coord()
            for col in supp_dict:
                num_cols = len(self.supp_count[col].columns)
                _, col_sum = self.get_margins(self.supp_count[col])
                variable = variable + [col] * num_cols
                category = category + list(self.supp_count[col].columns)
                is_active = is_active + ['False'] * num_cols
                x_coord = x_coord + list(supp_dict[col].iloc[:, 0])
                x_iner = x_iner + ['NA'] * num_cols
                y_coord = y_coord + list(supp_dict[col].iloc[:, 1])
                y_iner = y_iner + ['NA'] * num_cols
                count = count + list(col_sum)
        
        df = pd.DataFrame({'variable': variable,
                           'category': category,
                           'is_active': is_active,
                           'x_coord': x_coord,
                           'x_iner': x_iner,
                           'y_coord': y_coord,
                           'y_iner': y_iner,
                           'count': count})
        if tojson:
            json_dict = df.to_json()
            # json_dict['whole_variable'] = [self.row, self.col]
            return json_dict
        return df


    def generate_profile(self, direction, category='mean', tojson=True):
        # print('Backing' in self.count.index)
        # print(self.prob * 4512) 
        if direction == 'row' and category in self.prob.index:
            if tojson:
                return self.prob.loc[category].to_json()
            return self.prob.loc[category]
        elif direction == 'row' and category == 'mean':
            _, c = self.get_margins(self.prob)
            if tojson:
                return c.to_json()
            return c
        elif direction == 'col' and category in self.prob.columns:
            if tojson:
                return self.prob[category].to_json()
            return self.prob[category]
        elif direction == 'col' and category == 'mean':
            r, _ = self.get_margins(self.prob)
            if tojson:
                return r.to_json()
            return r
        else:
            raise Exception('''
            Invalid argument(s)! 
            direction must be either 'row' or 'col';
            category must either be a valid category name or 'mean'.
            ''')


    def generate_stats(self, tojson=True):
        chi2, p, dof, _ = stats.chi2_contingency(self.count, correction=False)
        stats_dict = {'chi2': chi2, 'p': p, 'DoF': dof}
        if tojson:
            return json.dumps(stats_dict)
        return stats_dict