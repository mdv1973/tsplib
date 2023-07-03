"""
Retrieve TSP problem specs obtained from TSPLIB at http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
"""
import os, pickle
import numpy as np

TSPLIB_PATH = os.path.dirname(__file__)

def _euc_2d(node1, node2): #verified
    xd = node1[0]-node2[0]
    yd = node1[1]-node2[1]
    return np.sqrt(xd*xd+yd*yd)

def _ciel_2d(node1, node2):
    xd = node1[0]-node2[0]
    yd = node1[1]-node2[1]
    return 1+int(np.sqrt(xd*xd+yd*yd))

def _man_2d(node1, node2):
    xd = abs(node1[0]-node2[0])
    yd = abs(node1[1]-node2[1])
    return xd+yd

def _max_2d(node1, node2):
    xd = abs(node1[0]-node2[0])
    yd = abs(node1[1]-node2[1])
    return max(xd, yd)

def _geo(node1, node2): #verified
    PI = 3.141592
    deg = int(node1[0])
    min = node1[0]-deg
    latitude1 = PI*(deg+5.*min/3.)/180.
    deg = int(node1[1])
    min = node1[1]-deg
    longitude1 = PI*(deg+5.*min/3.)/180.
    deg = int(node2[0])
    min = node2[0]-deg
    latitude2 = PI*(deg+5.*min/3.)/180.
    deg = int(node2[1])
    min = node2[1]-deg
    longitude2 = PI*(deg+5.*min/3.)/180.
    RRR = 6378.388
    q1 = np.cos(longitude1-longitude2)
    q2 = np.cos(latitude1-latitude2)
    q3 = np.cos(latitude1+latitude2)
    return int(RRR*np.arccos(.5*((1.+q1)*q2-(1.-q1)*q3))+1.)

def _att(node1, node2): #verified
    xd = node1[0]-node2[0]
    yd = node1[1]-node2[1]
    r = np.sqrt((xd*xd+yd*yd)/10.)
    t = int(.5+r)
    return t+1 if t<r else t

_dist_fn_map = {
    'EUC_2D': _euc_2d,
    'CIEL_2D': _ciel_2d,
    'MAN_2D': _man_2d,
    'MAX_2D': _max_2d,
    'GEO': _geo,
    'ATT': _att,
}

class Problem(object):
    def __init__(self, problem_type, name):
        """
        problem_type: string: 'tsp' or 'atsp'
        name: string: problem name

        raises: RuntimeError

        side-effects:
            writes ./{problem_type}/{name}.{problem_type}.dm.pkl file to cache distance matrix for the problem
        """
        self._problem_file = os.path.join(TSPLIB_PATH, problem_type, f'{name}.{problem_type}')
        if not os.path.isfile(self._problem_file): 
            raise RuntimeError(f'cannot find {self._problem_file}')
        self._distance_matrix_file = self._problem_file+'.dm.pkl'
        if os.path.isfile(self._distance_matrix_file):
            with open(self._distance_matrix_file, 'rb') as dmf:
                self.distance_matrix = pickle.load(dmf)
            if not isinstance(self.distance_matrix, np.ndarray):
                raise RuntimeError(f'invalid distance matrix in {self._distance_matrix_file}')
        else:
            self.distance_matrix = None
        self.name = name
        self.problem_type = problem_type.upper()
        self.comment = ''
        self.problem_size = -1
        self._parse_pfile()
        return
    def _parse_pfile(self):
        ew_type = ''
        ew_format = ''
        nc_type = 0
        nc_data = []
        ew_data = []
        with open(self._problem_file, 'rt') as pf:
            in_nc_section = False
            in_ew_section = False
            for line in pf:
                words = line.strip().replace(':',' : ').split()
                w = len(words)
                if not words:
                    continue
                if in_ew_section:
                    try:
                        _ = float(words[0])
                    except:
                        in_ew_section = False
                if not (in_nc_section or in_ew_section):
                    if words[0]=='NAME':
                        self.name = ' '.join(words[2:]) or self.name
                        continue
                    if words[0]=='TYPE' and w>2:
                        self.problem_type = words[2]
                        continue
                    if words[0]=='COMMENT':
                        if not self.comment:
                            self.comment = ' '.join(words[2:])
                        else:
                            self.comment += '\n'+' '.join(words[2:])
                        continue
                    if words[0]=='DIMENSION' and w>2:
                        self.problem_size = int(words[2])
                        continue
                    if words[0]=='EDGE_WEIGHT_TYPE' and w>2:
                        ew_type = words[2]
                        continue
                    if words[0]=='EDGE_WEIGHT_FORMAT' and w>2:
                        ew_format = words[2]
                        continue
                    if words[0]=='NODE_COORD_TYPE' and w>2:
                        nc_type_map = {'TWOD_COORDS':2, 'THREED_COORDS':3, 'NO_COORDS':0}
                        nc_type = nc_type_map.get(words[2], 0)
                        continue
                    if words[0]=='EOF':
                        break
                    if words[0]=='NODE_COORD_SECTION':
                        in_nc_section = True
                        continue
                    if words[0]=='EDGE_WEIGHT_SECTION':
                        in_ew_section = True
                        continue
                elif in_nc_section:
                    if nc_type and w!=nc_type: 
                        raise RuntimeError(f'inconsistent node_coord_type in {self._problem_file}')
                    nc_data.append([ float(x) for x in words[1:] ])
                    if len(nc_data)==self.problem_size:
                        in_nc_section = False
                    continue
                elif in_ew_section:
                    ew_data.extend([ float(x) for x in words ])
                    continue
        if self.distance_matrix is None:
            if ew_data:
                self.distance_matrix=self._make_distance_matrix_from_ew(ew_data, ew_format)
            elif nc_data:
                dist_fn = _dist_fn_map.get(ew_type,'')
                if not dist_fn:
                    raise RuntimeError(f'no distance function available for edge weight type {ew_type}')
                self.distance_matrix = self._make_distance_matrix_from_nc(nc_data, dist_fn)
            else:
                raise RuntimeError(f'no distance matrix can be obtained from the problem file {self._problem_file}')
            with open(self._distance_matrix_file, 'wb') as dmf:
                pickle.dump(self.distance_matrix, dmf)
        return

    def _make_distance_matrix_from_ew(self, edge_weight_list, format, as_int=True):
        N = self.problem_size
        ew = np.array(edge_weight_list, dtype=int if as_int else float)
        if format=='FULL_MATRIX':
            DM = ew.reshape((N, N))
        elif format in ('UPPER_ROW', 'LOWER_COL'):
            DM = np.zeros((N, N), dtype=int if as_int else float)
            ti = np.triu_indices(n=N, k=1)
            DM[ti] = ew
            DM = DM+DM.T
        elif format in ('LOWER_ROW', 'UPPER_COL'):
            DM = np.zeros((N, N), dtype=int if as_int else float)
            ti = np.tril_indices(n=N, k=1)
            DM[ti] = ew
            DM = DM+DM.T
        elif format in ('UPPER_DIAG_ROW', 'LOWER_DIAG_COL'):
            DM = np.zeros((N, N), dtype=int if as_int else float)
            ti = np.triu_indices(n=N, k=0)
            DM[ti] = ew
            DM = DM+DM.T-np.diag(np.diag(DM))
        elif format in ('LOWER_DIAG_ROW', 'UPPER_DIAG_COL'):
            DM = np.zeros((N, N), dtype=int if as_int else float)
            ti = np.tril_indices(n=N, k=0)
            DM[ti] = ew
            DM = DM+DM.T-np.diag(np.diag(DM))
        else:
            raise RuntimeError(f'cannot handle edge weigth format {format}')
        return DM

    def _make_distance_matrix_from_nc(self, node_coord_list, dist_fn, as_int=True):
        N = self.problem_size
        DM = np.zeros((N, N), dtype=int if as_int else float)
        for i in range(N):
            for j in range(i, N):
                dij = dist_fn(node_coord_list[i], node_coord_list[j])
                if as_int:
                    dij = int(.5+dij)
                DM[i, j] = dij
                DM[j, i] = dij
        return DM


if __name__=='__main__':
    #p = Problem('tsp','berlin52') #euc
    #p = Problem('tsp','swiss42') #full matrix tsp
    #p = Problem('tsp','pa561') #triangular
    #p = Problem('tsp','d198') #euc
    #p = Problem('tsp', 'att532') #att
    #p = Problem('tsp', 'bayg29') #upper row
    #p = Problem('tsp', 'dantzig42') #lower diag row
    #p = Problem('tsp', 'si175') #upper diag row
    p = Problem('atsp', 'br17') #full matrix atsp
    print(p.name)
    print(p.comment)
    print(p.problem_type, p.problem_size)
    print(p.distance_matrix[:10, :10], '...')

    N = p.problem_size
    DM = p.distance_matrix
    test_len = sum([ DM[i, i+1] for i in range(N-1) ])+DM[N-1, 0]
    print('canonical length', test_len)