#-*- coding: utf-8 -*-

#**********************************************************************
# Script name: gric.py
# Version: 2018.03.22 !BETA VERSION!
# Description: Python tool for constructing a grid of irregular shape,
# bounded by four points. Creates a masked grid out of Numpy's
# rectangular meshgrid. Cell objects can also be constructed out of the
# masked grid.
#
# Author: Gregor Rajh
# Year: 2018
# Python version: 2
# Dependencies:
#   *geographiclib (optional)
#   *matplotlib
#   *numpy
# github: http://github.com/grajh/gric
# e-mail: rajhgregor@gmail.com
#**********************************************************************


import numpy as np
import numpy.ma as ma

try:
    from geographiclib.geodesic import Geodesic
    gl_import = True
except ImportError, e:
    gl_import = False
from matplotlib.collections import LineCollection


class Cell(object):
    def __init__(self, ul, ur, ll, lr):
        self.ID = ""
        self.nds = (ul, ur, ll, lr)
        self.ul = ul
        self.ur = ur
        self.ll = ll
        self.lr = lr
    
    def contains(self, point_x, point_y):
        ul_x = self.ul[0]
        ul_y = self.ul[1]
        lr_x = self.lr[0]
        lr_y = self.lr[1]
        if point_x >= ul_x and point_x <= lr_x:
            if point_y >= lr_y and point_y <= ul_y:
                return True
            else:
                return False
        else:
            return False


class Grid(object):
    def __init__(self, ul_crnr, ur_crnr, ll_crnr, lr_crnr, grid_spac,
        degree_cor=False, grid_spac_unit='deg'):
        # Add grid_spac_rat argument.
        self.ul_crnr = ul_crnr
        self.ur_crnr = ur_crnr
        self.ll_crnr = ll_crnr
        self.lr_crnr = lr_crnr
        self.grid_spac = grid_spac
        self.fnds = []
        self.dec_placs = len(str(ul_crnr[0]).split('.')[1])
        self.x_min = min(ul_crnr[0], ll_crnr[0], ur_crnr[0], lr_crnr[0])
        self.x_max = max(ul_crnr[0], ll_crnr[0], ur_crnr[0], lr_crnr[0])
        self.y_min = min(ul_crnr[1], ll_crnr[1], ur_crnr[1], lr_crnr[1])
        self.y_max = max(ul_crnr[1], ll_crnr[1], ur_crnr[1], lr_crnr[1])
        self.mask_val = None

        lon_mid = (ul_crnr[0] + ll_crnr[0]) * 0.5
        lat_mid = (ul_crnr[1] + ll_crnr[1]) * 0.5
        delta_deg = 1/10**(self.dec_placs)
        lon_dmid = lon_mid + delta_deg
        lat_dmid = lat_mid + delta_deg

        ecf_WGS84 = 2 * np.pi * 6378137.0
        lon_lat_r = np.cos(np.radians(lat_mid))
        elat_len = ecf_WGS84 / 360

        # Use degree_cor=True only for grid specified in WGS84
        # coordinates.
        if degree_cor == True and gl_import == True:
            lon_len = Geodesic.WGS84.Inverse(
                lat_mid, lon_mid, lat_mid, lon_dmid)['s12']
            lat_len = Geodesic.WGS84.Inverse(
                lat_mid, lon_mid, lat_dmid, lon_mid)['s12']
            lon_lat_r = lon_len / lat_len

            self.grid_spac_lon = grid_spac / lon_lat_r
            self.grid_spac_lat = grid_spac

        elif degree_cor == True and gl_import == False:
            # Approximation for a sphere.
            self.grid_spac_lon = grid_spac / lon_lat_r
            self.grid_spac_lat = grid_spac

        elif grid_spac_unit == 'm' and gl_import == True:
            lon_len = Geodesic.WGS84.Inverse(
                lat_mid, lon_mid, lat_mid, lon_dmid)['s12']
            lat_len = Geodesic.WGS84.Inverse(
                lat_mid, lon_mid, lat_dmid, lon_mid)['s12']

            self.grid_spac_lon = grid_spac * delta_deg / lon_len
            self.grid_spac_lat = grid_spac * delta_deg / lat_len

        elif grid_spac_unit == 'm' and gl_import == False:
            # Approximation for a sphere.
            lon_len = elat_len * lon_lat_r

            self.grid_spac_lon = grid_spac / lon_len
            self.grid_spac_lat = grid_spac / elat_len

        else:
            self.grid_spac_lon = grid_spac
            self.grid_spac_lat = grid_spac


        delta_fg_x = self.x_max - self.x_min
        delta_fg_y = self.y_max - self.y_min
        # Implement module and substract it from x_max and y_max.
        nn_x = delta_fg_x / self.grid_spac_lon
        nn_y = delta_fg_y / self.grid_spac_lat

        space_x = np.linspace(self.x_min, self.x_max, nn_x + 1)
        space_y = np.linspace(self.y_max, self.y_min, nn_y + 1)

        self.xnds, self.ynds = np.meshgrid(
            space_x, space_y, sparse=False, indexing='xy'
            )

    def _slice_gridl(self, x_0, y_0, k):
        for i, row in enumerate(self.mxnds):
            y_value = self.ynds[i][0]
            x_value = x_0 + ((y_value - y_0) / (k))
            f_value = round(x_value, self.dec_placs + 1) \
                - (self.grid_spac / 10)

            rowm = ma.masked_where(row < f_value, row).mask
            self.mxnds[i] = ma.masked_where(rowm, self.mxnds[i])
            self.mynds[i] = ma.masked_where(rowm, self.mynds[i])
            fxnds = ma.filled(self.mxnds[i])
            fynds = ma.filled(self.mynds[i])

            row_coords = zip(fxnds, fynds)
            self.fnds.append(row_coords)

    def _slice_gridm(self, x_0, y_0, k):
        for i, row in enumerate(self.mxnds):
            y_value = self.ynds[i][0]
            x_value = x_0 + ((y_value - y_0) / (k))
            f_value = round(x_value, self.dec_placs + 1) \
                + (self.grid_spac / 10)

            rowm = ma.masked_where(row > f_value, row).mask
            self.mxnds[i] = ma.masked_where(rowm, self.mxnds[i])
            self.mynds[i] = ma.masked_where(rowm, self.mynds[i])
            fxnds = ma.filled(self.mxnds[i])
            fynds = ma.filled(self.mynds[i])

            row_coords = zip(fxnds, fynds)
            self.fnds.append(row_coords)

    def slice_grid(self):
        self.mxnds = ma.masked_array(np.copy(self.xnds))
        self.mynds = ma.masked_array(np.copy(self.ynds))
        self.mask_val = self.mxnds.fill_value

        delta_xbl = self.ll_crnr[0] - self.ul_crnr[0]
        delta_xbr = self.lr_crnr[0] - self.ur_crnr[0]
        delta_xlb = self.lr_crnr[0] - self.ll_crnr[0]
        delta_xub = self.ur_crnr[0] - self.ul_crnr[0]
        delta_yub = self.ur_crnr[1] - self.ul_crnr[1]
        delta_ylb = self.lr_crnr[1] - self.ll_crnr[1]
        delta_ybl = self.ll_crnr[1] - self.ul_crnr[1]
        delta_ybr = self.lr_crnr[1] - self.ur_crnr[1]

        if delta_ylb < 0:
            self.fnds = []
            x_0 = self.ll_crnr[0]
            y_0 = self.ll_crnr[1]
            k = delta_ylb / delta_xlb
            self._slice_gridl(x_0, y_0, k)
        elif delta_ylb > 0:
            self.fnds = []
            x_0 = self.ll_crnr[0]
            y_0 = self.ll_crnr[1]
            k = delta_ylb / delta_xlb
            self._slice_gridm(x_0, y_0, k)
        else:
            pass
        
        if delta_yub < 0:
            self.fnds = []
            x_0 = self.ul_crnr[0]
            y_0 = self.ul_crnr[1]
            k = delta_yub / delta_xub
            self._slice_gridm(x_0, y_0, k)
        elif delta_yub > 0:
            self.fnds = []
            x_0 = self.ul_crnr[0]
            y_0 = self.ul_crnr[1]
            k = delta_yub / delta_xub
            self._slice_gridl(x_0, y_0, k)
        else:
            pass

        if delta_xbr > 0 or delta_xbr < 0:
            self.fnds = []
            x_0 = self.ur_crnr[0]
            y_0 = self.ur_crnr[1]
            k = delta_ybr / delta_xbr
            self._slice_gridm(x_0, y_0, k)
        else:
            pass

        if delta_xbl > 0 or delta_xbl < 0:
            self.fnds = []
            x_0 = self.ul_crnr[0]
            y_0 = self.ul_crnr[1]
            k = delta_ybl / delta_xbl
            self._slice_gridl(x_0, y_0, k)
        else:
            pass

    def build_cells(self):
        self.cell_list = []            

        for rowi, rown in enumerate(self.fnds[1:]):
            row = self.fnds[rowi]

            for ndi, ndn in enumerate(row[1:]):
                ndni = ndi + 1
                nd_ul = row[ndi]
                nd_ur = ndn
                nd_ll = rown[ndi]
                nd_lr = rown[ndni]
                cell_coords = nd_ul + nd_ur + nd_ll + nd_lr

                if self.mask_val not in cell_coords:
                    cell_inst = Cell(nd_ul, nd_ur, nd_ll, nd_lr)
                    self.cell_list.append(cell_inst)
                else:
                    continue

        return self.cell_list

    def build_plot_cells(self, cell_list_in):
        self.cell_plist = []
        
        for cell in cell_list_in:
            cell_ndsu = (cell.ul, cell.ur)
            cell_ndsl = (cell.ll, cell.lr)
            cell_ndslt = (cell.ul, cell.ll)
            cell_ndsrt = (cell.ur, cell.lr)
            self.cell_plist.append(cell_ndsu)
            self.cell_plist.append(cell_ndsl)
            self.cell_plist.append(cell_ndslt)
            self.cell_plist.append(cell_ndsrt)

        self.cell_lins_collect = LineCollection(
            self.cell_plist, linewidths=(0.5), colors=('0.5'),
            linestyle='solid', alpha=1.0, zorder=10
            )
        
        return self.cell_plist, self.cell_lins_collect

    def add_cell(self, ul, ur, ll, lr):
        self.cell_list_mod = self.cell_list[:]
        self.cell_list_mod.append(Cell(ul, ur, ll, lr))
        